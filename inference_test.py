import os
import torch
from train import AntiSpoofing
import pytorch_lightning as pl
from torchvision import transforms
from datasets import FASDataset

import argparse

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--version', type=str, default='0', help='model version')
args = parser.parse_args()

checkpoint_dir = os.path.join('./logs/default', "version_{}".format(args.version), "checkpoints")
checkpoint_paths = os.listdir(checkpoint_dir)
checkpoint_paths = [checkpoint_paths[-1]]
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

for checkpoint_path in checkpoint_paths:
    submit_file = open("submit.txt", 'w')
    print(checkpoint_path)
    checkpoint_path = checkpoint_dir + checkpoint_path
    hyperparams = Namespace(**args)

    checkpoint = torch.load(checkpoint_path)
    model_infer = AntiSpoofing(hyperparams)
    model_infer.load_state_dict(checkpoint['state_dict'])

    model_infer.eval()
    trainer = pl.Trainer(
        max_epochs=1,
        gpus=1,
    )

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [1, 1, 1])
    ])

    test_set = FASDataset("./crop_label_val.txt", "./box_process/", transform,
                          phase='test')

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False, num_workers=8)

    results = trainer.predict(model_infer, test_loader)
    score = torch.hstack([x["score"] for x in results])

    all_path = []
    for x in results:
        all_path += x['paths']

    for ind in range(len(all_path)):
        submit_file.write("{} {}\n".format(os.path.split(all_path[ind])[-1], score[ind]))

    test_set = FASDataset("./crop_label_test.txt", "./box_process/", transform,
                          phase='test')

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False, num_workers=8)

    results = trainer.predict(model_infer, test_loader)

    score = torch.hstack([x["score"] for x in results])

    all_path = []
    for x in results:
        all_path += x['paths']

    for ind in range(len(all_path)):
        r_path = os.path.split(all_path[ind])[-1]
        name = r_path.split('.')[0]
        new_name = int(name) + 4645
        new_name = str(new_name) + ".png"
        submit_file.write("{} {}\n".format(new_name, score[ind]))

    # trainer.test(model_infer)

# try_dataloader = model_infer.test_dataloader()
#
# inputs, labels = next(iter(try_dataloader))
# print("read img")
# print(len(inputs))
# # inference
# outputs = model_infer(inputs)
# print(outputs[0])
# _, preds = torch.max(outputs, dim=1)
# print(preds)
# print(torch.sum(preds == labels.data) / (labels.shape[0] * 1.0))

# print('Predicted: ', ' '.join('%5s' % classes[preds[j]] for j in range(8)))
