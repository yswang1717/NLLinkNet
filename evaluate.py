import os
import random
import json

import cv2
import numpy as np
import torch

from tqdm import tqdm


SMOOTH_IOU = 1e-6


def BatchInU_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    """
    :param outputs: Tensor. [BATCH, H, W].
    :param labels: Tensor. [BATCH, H, W].
    :param i_counts: List. len() -> C.
    :param u_counts: List. len() -> C.
    """
    if len(labels.size()) == 4 and labels.size(1) == 1:
        labels = labels[:, 0]

    assert outputs.size() == labels.size()

    outputs_c = outputs.cuda().detach() == 1
    labels_c = labels.cuda().detach() == 1
    intersection = (outputs_c & labels_c).float().sum()  # Will be zero if Truth=0 or Prediction=0
    union = (outputs_c | labels_c).float().sum()  # Will be zero if both are 0

    return int(intersection.cpu()) / int(union.cpu())


def test(gt_base_path, prediction_base_path, imagelist):
    mious = []

    for image_basename in tqdm(imagelist):
        gt_path = os.path.join(gt_base_path, '{}_mask.png').format(image_basename)
        mask_path = os.path.join(prediction_base_path, '{}_mask.png').format(image_basename)

        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        gt = torch.from_numpy(gt).cuda().detach()
        mask = torch.from_numpy(mask).cuda().detach()

        gt[gt >= 0.5] = 1
        gt[gt <= 0.5] = 0
        gt = gt[None]

        mask[mask >= 0.5] = 1
        mask[mask <= 0.5] = 0
        mask = mask[None]

        mious.append(BatchInU_pytorch(mask, gt))

    mious = np.array(mious)

    return np.mean(mious)


def main(dataset_path, result_path):
    imagelist = list(filter(lambda x: x.find('sat') != -1, os.listdir(dataset_path)))
    random.Random(2020).shuffle(imagelist)

    train_imagelist = imagelist[:-1200]
    train_imagelist = list(map(lambda x: x[:-8], train_imagelist))

    val_imagelist = imagelist[-1200:]
    val_imagelist = list(map(lambda x: x[:-8], val_imagelist))

    result_epoch_digit = [f for f in os.listdir(result_path)
                          if os.path.isdir(os.path.join(result_path, f)) and f.isdigit() and int(f) % 5 == 0]

    result_epoch_digit = result_epoch_digit[:50]

    train_mious = {}
    val_mious = {}

    for epoch in tqdm(result_epoch_digit):
        prediction_base_path = os.path.join(result_path, epoch)

        train_mious[int(epoch)] = test(gt_base_path=dataset_path, prediction_base_path=prediction_base_path,
                                       imagelist=train_imagelist)
        val_mious[int(epoch)] = test(gt_base_path=dataset_path, prediction_base_path=prediction_base_path,
                                     imagelist=val_imagelist)
        print("[%s] Train: %s, Val: %s" % (epoch, train_mious[int(epoch)], val_mious[int(epoch)]))

    return train_mious, val_mious


if __name__ == '__main__':
    dataset_path = "/mnt/Dataset/DeepGlobe/Road/train"
    result_path = "/mnt/Checkpoints/NL-LinkNet/DLinkNet/Results"
    # result_path = "/mnt/Checkpoints/NL-LinkNet/NLLinkNet34/Results"

    train_mious, val_mious = main(dataset_path, result_path)

    with open('DLinkNet_Train.json', 'w') as fp:
        json.dump(train_mious, fp)
    with open('DLinkNet_Val.json', 'w') as fp:
        json.dump(val_mious, fp)
    """
    with open('NLLinkNet34_Train.json', 'w') as fp:
        json.dump(train_mious, fp)
    with open('NLLinkNet34_Val.json', 'w') as fp:
        json.dump(val_mious, fp)
    """
