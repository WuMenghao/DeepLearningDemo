# -*- coding: utf-8 -*-
# Created by: WU MENGHAO
# Created on: 2019/12/10
from __future__ import print_function

import os
from tensorflow.python.client import device_lib

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
local_device_protos = device_lib.list_local_devices()
names = [x.name for x in local_device_protos if x.device_type == 'GPU']
print('devices: %s' % local_device_protos)
print('GPUs: %s' % names)
