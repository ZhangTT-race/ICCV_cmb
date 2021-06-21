import cv2
import os
import argparse

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--path', type=str, default='/tmp-data/sys/HiFiMask-Challenge', help='path to dir:HiFiMask-Challenge')


def crop_img(save_path, in_bbox_info, out_crop_info, out_label, replace_path, with_label=False):
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    crop_info = open(out_crop_info, "w")
    label_file = open(out_label, 'w')

    with open(in_bbox_info, 'r') as T:
        for line in T.readlines():
            line = line.strip()
            path, label = line.split(" ")[:2]
            path = path.replace('/tmp-data/sys/HiFiMask-Challenge', replace_path)
            img = cv2.imread(path)
            H, W, C = img.shape
            print(H, W, C)
            if "None" not in line:
                bbox = line.split('[')[-1]
                bbox = bbox.split("]")[0]
                bbox = bbox.split(" ")
                while ("" in bbox):
                    bbox.remove("")
                bbox = bbox[:4]
                print(bbox)
                print(path)
                assert len(bbox) == 4
                bbox = list(map(int, bbox))
                # img = cv2.imread(path)
                v_len = bbox[2] - bbox[0]
                h_len = bbox[3] - bbox[1]
                left = int(max(0, bbox[0] - v_len / 8))
                right = int(min(W, bbox[2] + v_len / 8))
                up = int(max(0, bbox[1] - h_len / 8))
                down = int(min(H, bbox[3] + h_len / 8))
            else:
                left = up = 0
                right = W
                down = H
            print(left, up, right, down)
            crop = img[up:down, left:right, :]
            img_name = path.split("/")[-1]
            dir_name = path.split("/")[-2]
            save_dir = os.path.join(save_path, dir_name)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            img_path = os.path.join(save_dir, img_name)
            cv2.imwrite(img_path, crop)
            if not with_label:
                crop_info.write("{} {}\n".format(img_path, str([left, up, right, down])))
                label_file.write("{} {}\n".format(img_path, 0))
            else:
                crop_info.write("{} {} {}\n".format(img_path, label, str([left, up, right, down])))
                label_file.write("{} {}\n".format(img_path, label))
            # break
    label_file.close()
    crop_info.close()

if __name__ == '__main__':
    args = parser.parse_args()

    replace_path = args.path

    save_path = os.path.join(replace_path, "train_cropped")
    crop_info = "crop_info_train.txt"
    bbox_info = "bbox_info_train.txt"
    label_file = "crop_label_train.txt"
    crop_img(save_path, bbox_info, crop_info, label_file, replace_path, True)
    save_path = os.path.join(replace_path, "val_cropped")
    crop_info = "crop_info_val.txt"
    bbox_info = "bbox_info_val.txt"
    label_file = "crop_label_val.txt"
    crop_img(save_path, bbox_info, crop_info, label_file, replace_path, True)
    save_path = os.path.join(replace_path, "test_cropped")
    crop_info = "crop_info_test.txt"
    bbox_info = "bbox_info_test.txt"
    label_file = "crop_label_test.txt"
    crop_img(save_path, bbox_info, crop_info, label_file, replace_path, False)
