import pytorch_lightning as pl
import models
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from datasets import FASDataset
import os
import argparse
from pytorch_lightning import seed_everything
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from loss.label_smoothing import LabelSmoothing
from loss.focal_loss import FocalLoss
from loss.triplet_loss import TripletLoss
from pytorch_lightning.trainer.supporters import CombinedLoader

# Set seed
seed = 42
seed_everything(seed)

parser = argparse.ArgumentParser()



class AntiSpoofing(pl.LightningModule):

    def __init__(self, hparams):
        super(AntiSpoofing, self).__init__()

        self.params = hparams
        self.num_classes = self.params.num_classes
        self.data_dir = self.params.data_dir
        self.train_label = self.params.train_label
        self.val_label = self.params.val_label
        self.batch_size = self.params.batch_size
        self.lr = self.params.lr
        self.loss = self.params.loss
        self.crit1_weight = self.params.crit1_weight
        self.crit2_weight = self.params.crit2_weight

        ########## define the model ########## 
        # arch = torchvision.models.resnet18(pretrained=False)
        self.arch = models.__dict__[self.params.arch]()

        # arch = torchvision.models.mobilenet_v3_large()
        # num_ftrs = arch.fc.in_features
        #
        modules = list(self.arch.children())[:-1]  # ResNet18 has 10 children

        self.backbone = torch.nn.Sequential(*modules)  # [bs, 512, 1, 1]

        self.fea_final = torch.nn.Sequential(
            torch.nn.Linear(1280, 128))

        self.final = torch.nn.Sequential(
            torch.nn.Dropout(0.1),
            torch.nn.Linear(1280, self.num_classes))
        self.crit = None
        self.crit2 = None
        if "label_smoothing" in self.loss:
            self.crit = LabelSmoothing(size=2, smoothing=0.1)
        elif "CE" in self.loss:
            self.crit = F.cross_entropy
        elif "focal_loss" in self.loss:
            self.crit = FocalLoss(2, smoothing=0.1)

        if "triplet_loss" in self.loss:
            self.crit2 = TripletLoss()

    def forward(self, x):
        x = self.backbone(x)
        x = x.reshape(x.size(0), -1)
        y = self.final(x)

        return x, y

    def configure_optimizers(self):
        # REQUIRED
        optimizer = torch.optim.SGD([
            {'params': self.backbone.parameters()},
            {'params': self.final.parameters()}
        ], lr=self.lr)

        exp_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, self.params.cos_T)
        return [optimizer], [exp_lr_scheduler]
        # return optimizer

    def on_train_start(self):
        self.logger.log_hyperparams(vars(self.params))

    def training_step(self, batch, batch_idx):
        # REQUIRED
        x, y = batch
        in_feature, y_hat = self.forward(x)
        loss = 0
        if self.crit and self.crit1_weight:
            loss += self.crit1_weight * self.crit(y_hat, y)
        if self.crit2 and self.crit2_weight:
            norm_fea = F.normalize(in_feature)
            loss += self.crit2_weight * self.crit2(norm_fea, y)

        _, preds = torch.max(y_hat, dim=1)
        acc = torch.sum(preds == y.data) / (y.shape[0] * 1.0)

        self.log('train_loss', loss)
        self.log('train_acc', acc)

        return {'loss': loss, 'train_acc': acc}

    def validation_step(self, batch, batch_idx):
        # OPTIONALl
        val_batch = batch
        x, y = val_batch
        _, y_hat = self.forward(x)
        softmax = torch.softmax(y_hat, 1)
        ind = torch.ones([softmax.shape[0], 1], dtype=torch.int64).cuda()
        score = torch.gather(softmax, 1, ind)

        loss = F.cross_entropy(y_hat, y)
        _, preds = torch.max(y_hat, 1)
        acc = torch.sum(preds == y.data) / (y.shape[0] * 1.0)
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        return {'val_loss': loss, 'val_acc': acc, 'score': score.squeeze(1), 'y': y}

    def predict_step(self, batch, batch_idx: int, dataloader_idx=None):
        x, y, paths = batch
        fea, y_hat = self.forward(x)
        softmax = torch.softmax(y_hat, 1)
        ind = torch.ones([softmax.shape[0], 1], dtype=torch.int64).cuda()
        score = torch.gather(softmax, 1, ind)

        loss = F.cross_entropy(y_hat, y)
        _, preds = torch.max(y_hat, 1)
        acc = torch.sum(preds == y.data) / (y.shape[0] * 1.0)

        return {'predict_loss': loss, 'predict_acc': acc, 'score': score.squeeze(1), 'y': y, 'paths': paths, 'fea': fea}

    def validation_epoch_end(self, outputs) -> None:
        score = torch.hstack([x["score"] for x in outputs])
        y = torch.hstack([x["y"] for x in outputs])

        min_acer = 1
        min_apcer = 1
        min_bpcer = 1
        min_th = 0

        for th in torch.range(0.5, 1, 0.001):
            th_score = torch.where(score > th, 1, 0)
            TP = torch.sum(torch.where((th_score == 0) & (y == 0), 1, 0))
            TN = torch.sum(torch.where((th_score == 1) & (y == 1), 1, 0))
            FP = torch.sum(torch.where((th_score == 0) & (y == 1), 1, 0))
            FN = torch.sum(torch.where((th_score == 1) & (y == 0), 1, 0))

            apcer = FN / (FN + TP)
            bpcer = FP / (TN + FP)
            acer = (apcer + bpcer) / 2
            if acer < min_acer:
                min_bpcer = bpcer
                min_apcer = apcer
                min_acer = acer
                min_th = th

        logs = {"acer": min_acer, "apcer": min_apcer, "bpcer": min_bpcer, "th": min_th}

        self.log_dict(logs)

    def train_dataloader(self):
        # REQUIRED

        trans_list = []
        real_trans_list = []

        if "RandomResizedCrop" in self.params.trans:
            trans_list.append(transforms.RandomResizedCrop(224, scale=self.params.scale))
            real_trans_list.append(transforms.RandomResizedCrop(224, scale=self.params.scale))
        else:
            trans_list.append(transforms.Resize(224))
            real_trans_list.append(transforms.Resize(224))

        if "HorizontalFlip" in self.params.trans:
            trans_list.append(transforms.RandomHorizontalFlip())
            real_trans_list.append(transforms.RandomHorizontalFlip())
        if "sharpness" in self.params.trans:
            trans_list.append(transforms.RandomChoice([transforms.RandomAdjustSharpness(1.5, 0.2),
                                                       transforms.RandomAdjustSharpness(2, 0.2),
                                                       transforms.RandomAdjustSharpness(2.5, 0.2),
                                                       transforms.RandomAdjustSharpness(3, 0.2),
                                                       transforms.RandomAdjustSharpness(3.5, 0.2),
                                                       transforms.RandomAdjustSharpness(4, 0.2)]
                                                      ))

        tmp_trans = []
        if "ColorJitter" in self.params.trans:
            tmp_trans.append(transforms.ColorJitter(0.2, 0.2, 0.2, 0.2))
        if "Blur" in self.params.trans:
            tmp_trans.append(transforms.GaussianBlur(3))
        if len(tmp_trans) > 0:
            trans_list.append(transforms.RandomApply(torch.nn.ModuleList(tmp_trans), self.params.cj_blur_ratio))
            real_trans_list.append(transforms.RandomApply(torch.nn.ModuleList(tmp_trans), self.params.cj_blur_ratio))

        trans_list.append(transforms.ToTensor())
        real_trans_list.append(transforms.ToTensor())
        if "RandomErasing" in self.params.trans:
            trans_list.append(transforms.RandomErasing(0.2))
            real_trans_list.append(transforms.RandomErasing(0.2))
        elif "RandomErasingWhite" in self.params.trans:
            trans_list.append(transforms.RandomErasing(0.2, value=1))
            real_trans_list.append(transforms.RandomErasing(0.2, value=1))
        elif "RandomErasingRandom" in self.params.trans:
            trans_list.append(transforms.RandomErasing(0.2, value="random"))
            real_trans_list.append(transforms.RandomErasing(0.2, value="random"))

        if "Grayscale" in self.params.trans:
            trans_list.append(transforms.Grayscale(num_output_channels=3))
            real_trans_list.append(transforms.Grayscale(num_output_channels=3))

        trans_list.append(transforms.Normalize([0.5, 0.5, 0.5], [1, 1, 1]))
        real_trans_list.append(transforms.Normalize([0.5, 0.5, 0.5], [1, 1, 1]))

        transform = transforms.Compose(trans_list)
        real_transform = transforms.Compose(real_trans_list)

        train_set = FASDataset(self.train_label, self.data_dir, transform, real_transform)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=24)

        return train_loader

    def val_dataloader(self):
        trans_list = [transforms.Resize(256),
                      transforms.CenterCrop(224)]
        if "Grayscale" in self.params.trans:
            trans_list.append(transforms.Grayscale(num_output_channels=3))
        trans_list.append(transforms.ToTensor())
        trans_list.append(transforms.Normalize([0.5, 0.5, 0.5], [1, 1, 1]))
        transform = transforms.Compose(trans_list)

        val_set = FASDataset(self.val_label, self.data_dir, transform)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=int(self.batch_size / 4), shuffle=True,
                                                 num_workers=4)

        return val_loader


def main(hparams):
    model = AntiSpoofing(hparams)
    tb_logger = pl_loggers.TensorBoardLogger('logs/')
    tb_logger.log_hyperparams(hparams)

    checkpoint_callback = ModelCheckpoint(monitor='acer', save_top_k=20)
    kwargs = {}
    if hparams.__contains__("resume"):
        kwargs = {"resume_from_checkpoint": hparams.resume}

    trainer = pl.Trainer(
        max_epochs=hparams.epochs,
        gpus=hparams.gpus,
        accelerator='ddp',
        logger=tb_logger,
        stochastic_weight_avg=True,
        callbacks=[checkpoint_callback],

        **kwargs
    )
    trainer.fit(model)


from argparse import Namespace

args = {
    'arch': "mbv2_ca",
    'num_classes': 2,
    'epochs': 240,
    'batch_size': 64,
    "cos_T": 60,
    "data_dir": "./box_process/",
    "train_label": "./crop_label_train.txt",
    "val_label": "./crop_label_val.txt",
    'gpus': 2,
    'lr': 0.1,
    'loss': "CE",
    'crit1_weight': 1,
    'crit2_weight': 0.6,
    'trans': "RandomResizedCrop,HorizontalFlip,Blur,ColorJitter,RandomErasingRandom,sharpness",
    'scale': (0.08, 1),
    "cj_blur_ratio": 0.15,
    # 'resume': ""
}

hyperparams = Namespace(**args)

if __name__ == '__main__':
    main(hyperparams)
