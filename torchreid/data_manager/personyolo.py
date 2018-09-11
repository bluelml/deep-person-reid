from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import re
import sys
import urllib
import tarfile
import zipfile
import os.path as osp
from scipy.io import loadmat
import numpy as np
import h5py
from scipy.misc import imsave
import json


class PersonYolo(object):
    """

    """
    dataset_dir = 'personyolo'

    def __init__(self, root='data', verbose=True, **kwargs):
        super(PersonYolo, self).__init__()

        self.json_file = '../api/matches-ralphdemovball_yolo.json.line'

        yolo, num_fns= self._process_json(self.json_file, relabel=True)

        if verbose:
            print("=> PersonYolo loaded")
            print("Dataset statistics:")
            print("  ------------------------------")
            print(yolo[:5])
            print("Number of iamges {}".format(num_fns))
            print("  ------------------------------")

        self.yolo = yolo
        self.num_fns = num_fns

    def _process_json(self, file_path, relabel=False):
        dataset = []
        with open(file_path, 'r') as f:
            for line in f:
                info = json.loads(line.strip())
                img_path = info['input_fn']
                for item in info['result']:
                    bbox = [int(item[2][0]), int(item[2][1]), int(item[2][2]), int(item[2][3])]
                    dataset.append((img_path, bbox))

        num_fns = len(dataset)
        return dataset, num_fns