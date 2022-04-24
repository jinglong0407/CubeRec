'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
'''

import os
from os.path import join
import torch
from enum import Enum
import multiprocessing

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# ROOT_PATH = "/Users/gus/Desktop/light-gcn"
ROOT_PATH = "C:/Users/wangz/Desktop/LightGCN-PyTorch-master/LightGCN-PyTorch-master"
CODE_PATH = join(ROOT_PATH, 'code')
DATA_PATH = join(ROOT_PATH, 'data')
BOARD_PATH = join(CODE_PATH, 'runs')
FILE_PATH = join(CODE_PATH, 'checkpoints')
import sys
sys.path.append(join(CODE_PATH, 'sources'))
if not os.path.exists(FILE_PATH):
    os.makedirs(FILE_PATH, exist_ok=True)


config = {}
config['bpr_batch_size'] = 64
config['latent_dim_rec'] = 64
config['lightGCN_n_layers']= 3
config['dropout'] = 0
config['keep_prob']  = 0.6
config['A_n_fold'] = 100
config['test_u_batch_size'] = 100
config['multicore'] = 0
config['lr'] = 0.001
config['decay'] = 1e-4
config['pretrain'] = 0
config['A_split'] = False
config['bigdata'] = False

GPU = torch.cuda.is_available()
device = torch.device('cuda' if GPU else "cpu")
CORES = multiprocessing.cpu_count() // 2


seed = 2020
dataset = 'Yelp'
LOAD = 0
PATH = "./checkpoints"
topks = eval("[20]")
tensorboard = 1
comment = "lgn"

# group loss & ssl loss
mu = 0

# group representations
group_rep = 'geometric'
# group_rep = 'attentive'


# let pandas shut up
from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)


def cprint(words : str):
    print(f"\033[0;30;43m{words}\033[0m")

