import sys
import os
import os.path
import torch 
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
from sparse_gp import SparseGP
import scipy.stats as sps
import random
import numpy as np
import json 
import time 
import scipy.io
from scipy.io import loadmat
from scipy.stats import pearsonr
sys.path.append('%s/..' % os.path.dirname(os.path.realpath(__file__))) 
sys.path.insert(0, '../')
from shutil import copy

from ConfigSpace.read_and_write import json as config_space_json_r_w
# from nasbench import api
# from nasbench.lib import graph_util
from models import GNNVAE_fb
from utils import util 

### Parsed Arguements
saved_log_dir ='experiments/svge/nb101/test/' ##180820
BO_rounds =10
BO_batch_size =5  ##50
sample_dist='uniform'
random_baseline = True
log_dir ='bayesian_optimization/test_nb101_{}/'.format(time.strftime("%Y%m%d-%H%M%S"))
model_name='GNNVAE_fb'
checkpoint=300
data_type='NB101'
data_name='NB101'
device='cuda'

iteration=0
rand_idx=0
save_dir = '{}results_{}_{}/'.format(log_dir, model_name, rand_idx)  # where to save the BO results
if not os.path.exists(save_dir):
        os.makedirs(save_dir)
best_arc=('0 1 1 0 0 0 1 0 0 0 1 0 0 1 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0', '1 3 2 4 3 2 0')
g_best=util.parse_graph_to_nx(best_arc[0], best_arc[1], flat=True)
util.plot_DAG(g_best, save_dir, 'best_arc_iter_{}'.format(iteration), data_type=data_type, pdf=True)

#Nasbench 101 Settings
