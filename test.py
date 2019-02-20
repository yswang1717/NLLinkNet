import argparse
import os

import cv2
import numpy as np
from tqdm import tqdm

from networks.dinknet import LinkNet34, DinkNet34
from networks.nllinknet_location import NL3_LinkNet, NL4_LinkNet, NL34_LinkNet, Baseline
from networks.nllinknet_pairwise_func import NL_LinkNet_DotProduct, NL_LinkNet_Gaussian, NL_LinkNet_EGaussian
from networks.unet import Unet
from test_framework import TTAFramework


def test_models(model, name, source='../dataset/Road/valid', scales=(1.0,), target=''):
    if type(scales) == tuple:
        scales = list(scales)
    print(model, name, source, scales, target)

    solver = TTAFramework(model)
    solver.load('weights/' + name + '.th')

    if target == '':
        target = 'submits/' + name + '/'
    else:
        target = 'submits/' + target + '/'

    source = '../dataset/Road/valid/'

    val = os.listdir(source)
    if not os.path.exists(target):
        try:
            os.makedirs(target)
        except OSError as e:
            import errno
            if e.errno != errno.EEXIST:
                raise
    len_scales = int(len(scales))
    if len_scales > 1:
        print('multi-scaled test : ', scales)

    for i, name in tqdm(enumerate(val), ncols=10, desc="Testing "):
        mask = solver.test_one_img_from_path(source + name, scales)
        mask[mask > 4.0 * len_scales] = 255  # 4.0
        mask[mask <= 4.0 * len_scales] = 0
        mask = mask[:, :, None]
        mask = np.concatenate([mask, mask, mask], axis=2)
        cv2.imwrite(target + name[:-7] + 'mask.png', mask.astype(np.uint8))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="set model name")
    parser.add_argument("--name", help="set path of weights")
    parser.add_argument("--source", help="path of test datasets", default='../dataset/Road/valid')
    parser.add_argument("--scales", help="set scales for MST", default=[1.0], type=float, nargs='*')
    parser.add_argument("--target", help="path of submit files", default='')

    args = parser.parse_args()

    models = {'NL3_LinkNet': NL3_LinkNet, 'NL4_LinkNet': NL4_LinkNet, 'NL34_LinkNet': NL34_LinkNet,
              'Baseline': Baseline,
              'NL_LinkNet_DotProduct': NL_LinkNet_DotProduct, 'NL_LinkNet_Gaussian': NL_LinkNet_Gaussian,
              'NL_LinkNet_EGaussian': NL_LinkNet_EGaussian,
              'UNet': Unet, 'LinkNet': LinkNet34, 'DLinkNet': DinkNet34}

    model = models[args.model]
    name = args.name
    scales = args.scales
    target = args.target
    source = args.source

    test_models(model=model, name=name, source=source, scales=scales, target=target)


if __name__ == "__main__":
    main()
