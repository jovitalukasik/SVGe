######################################################################################
# Parts based on
# Copyright (c) Shen Yan, arch2vec, 
# https://github.com/MSU-MLSys-Lab/arch2vec
# modified
######################################################################################

import sys, pathlib
import os, glob, json
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import torch
import numpy as np 
import itertools
import tqdm
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_dense_adj
import torch.nn.functional as F

from collections import OrderedDict

from nasbench201 import api
from utils_data import prep_data


if __name__ == "__main__":
    
    # Set package for relative imports
    if __package__ is None or len(__package__) == 0:                  
        DIR = pathlib.Path(__file__).resolve().parent.parent
        print(DIR)
        sys.path.insert(0, str(DIR.parent))
        __package__ = DIR.name

##############################################################################
#
#                              Dataset Code
#
##############################################################################


# Useful constants
OP_PRIMITIVES_201 = [
    'output',
    'input',
    'nor_conv_1x1',
    'nor_conv_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'none',
]

OPS_by_IDX_201 = {OP_PRIMITIVES_201.index(i):i for i in OP_PRIMITIVES_201}
OPS_201 = {i:OP_PRIMITIVES_201.index(i) for i in OP_PRIMITIVES_201}

ADJACENCY = np.array([[0, 1, 1, 0, 1, 0, 0, 0],
                    [0, 0, 0, 1, 0, 1 ,0 ,0],
                    [0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0]])


nasbench = api.NASBench201API('datasets/nasbench201/NAS-Bench-201-v1_0-e61699.pth')

class Dataset:
            
    ##########################################################################
    def __init__(
            self,
            batch_size,
        ):

        if __name__ == "__main__":
            path = os.path.join(".","nasbench201") 
        else:
            path = os.path.join(".","datasets","nasbench201") #for debugging
            
        file_cache_train = os.path.join(path, "cache_train")
        file_cache_test = os.path.join(path, "cache_test")
        file_cache = os.path.join(path, "cache")
        ############################################        

        if not os.path.isfile(file_cache):
            self.data = []
            for index in tqdm.tqdm(range(len(nasbench))):
                item = nasbench.query_meta_info_by_index(index)
                self.data.append(Dataset.map_item(index))
                self.data[-1].edge_index =  Dataset.map_network(item)[0]
                self.data[-1].node_atts = Dataset.map_network(item)[1]
                self.data[-1].num_nodes = 8
            
            print(f"Saving data to cache: {file_cache}")
            torch.save(self.data, file_cache)
        
        ############################################
        if not os.path.isfile(file_cache_train):
            print(f"Loading data from cache: {file_cache}")
            self.data = torch.load(file_cache)

            self.train_data, self.test_data = Dataset.sample(self.data)
            print('prepare data for Autoencoder Training')
            self.train_data = prep_data(self.train_data, max_num_nodes=8, NB201=True)

            print(f"Saving train data to cache: {file_cache_train}")
            torch.save(self.train_data, file_cache_train)

            print(f"Saving test data to cache: {file_cache_test}")
            torch.save(self.test_data, file_cache_test)
        
        else:
            print(f"Loading train data from cache: {file_cache_train}")
            self.train_data = torch.load(file_cache_train)

            print(f"Loading test data from cache: {file_cache_test}")
            self.test_data = torch.load(file_cache_test)
        
        ############################################

        self.length = len(self.train_data)+ len(self.test_data)

        self.train_dataloader = DataLoader(
            self.train_data,
            shuffle = True,
            num_workers = 4,
            pin_memory = True,
            batch_size = batch_size
        )

        self.test_dataloader = DataLoader(
            self.test_data,
            shuffle = False,
            num_workers = 4,
            pin_memory = False,
            batch_size = batch_size
        )

        # self.dataloader = DataLoader(
        #     self.data,
        #     shuffle = True,
        #     num_workers = 4,
        #     pin_memory = True,
        #     batch_size = batch_size
        # )


    ##########################################################################
    @staticmethod
    def map_item(item, dataset='cifar10_valid_converged'):
        valid_acc, val_acc_avg, time_cost, test_acc, test_acc_avg = Dataset.train_and_eval(item, nepoch=None, dataname='cifar10-valid', use_converged_LR=True)
        acc = torch.FloatTensor([valid_acc/100.0 ])
        test_acc = torch.FloatTensor([test_acc/100.0 ])
        val_acc_avg = torch.FloatTensor([val_acc_avg/100.0])
        test_acc_avg = torch.FloatTensor([test_acc_avg/100.0 ])
        training_time = torch.FloatTensor([time_cost/100.0])
        
        return Data(acc=acc, test_acc=test_acc, acc_avg=val_acc_avg, test_acc_avg=test_acc_avg, training_time=training_time)

    ##########################################################################
    @staticmethod
    def map_network(item):

        nodes = ['input']
        steps = item.arch_str.split('+')
        steps_coding = ['0', '0', '1', '0', '1', '2']
        cont = 0
        for step in steps:
            step = step.strip('|').split('|')
            for node in step:
                n, idx = node.split('~')
                assert idx == steps_coding[cont]
                cont += 1
                nodes.append(n)
        nodes.append('output')

        ops = [OPS_201[k] for k in nodes]

        node_attr = torch.LongTensor(ops)
        edge_index = torch.tensor(np.nonzero(ADJACENCY))
        
        return edge_index.long(), node_attr
            
    ##########################################################################
    @staticmethod
    def train_and_eval(arch_index, nepoch=None, dataname=None, use_converged_LR=True):
        assert dataname !='cifar10', 'Do not allow cifar10 dataset'
        if use_converged_LR and dataname=='cifar10-valid':
            assert nepoch == None, 'When using use_converged_LR=True, please set nepoch=None, use 12-converged-epoch by default.'


            info = nasbench.get_more_info(arch_index, dataname, None, True)
            valid_acc, time_cost = info['valid-accuracy'], info['train-all-time'] + info['valid-per-time']
            valid_acc_avg = nasbench.get_more_info(arch_index, 'cifar10-valid', None, False, False)['valid-accuracy']
            test_acc = nasbench.get_more_info(arch_index, 'cifar10', None, False, True)['test-accuracy']
            test_acc_avg = nasbench.get_more_info(arch_index, 'cifar10', None, False, False)['test-accuracy']

        elif not use_converged_LR:

            assert isinstance(nepoch, int), 'nepoch should be int'
            xoinfo = nasbench.get_more_info(arch_index, 'cifar10-valid', None, True)
            xocost = nasbench.get_cost_info(arch_index, 'cifar10-valid', False)
            info = nasbench.get_more_info(arch_index, dataname, nepoch, False, True)
            cost = nasbench.get_cost_info(arch_index, dataname, False)
            # The following codes are used to estimate the time cost.
            # When we build NAS-Bench-201, architectures are trained on different machines and we can not use that time record.
            # When we create checkpoints for converged_LR, we run all experiments on 1080Ti, and thus the time for each architecture can be fairly compared.
            nums = {'ImageNet16-120-train': 151700, 'ImageNet16-120-valid': 3000,
                    'cifar10-valid-train' : 25000,  'cifar10-valid-valid' : 25000,
                    'cifar100-train'      : 50000,  'cifar100-valid'      : 5000}
            estimated_train_cost = xoinfo['train-per-time'] / nums['cifar10-valid-train'] * nums['{:}-train'.format(dataname)] / xocost['latency'] * cost['latency'] * nepoch
            estimated_valid_cost = xoinfo['valid-per-time'] / nums['cifar10-valid-valid'] * nums['{:}-valid'.format(dataname)] / xocost['latency'] * cost['latency']
            try:
                valid_acc, time_cost = info['valid-accuracy'], estimated_train_cost + estimated_valid_cost
            except:
                valid_acc, time_cost = info['est-valid-accuracy'], estimated_train_cost + estimated_valid_cost
            test_acc = info['test-accuracy']
            test_acc_avg = nasbench.get_more_info(arch_index, dataname, None, False, False)['test-accuracy']
            valid_acc_avg = nasbench.get_more_info(arch_index, dataname, None, False, False)['valid-accuracy']
        else:
            # train a model from scratch.
            raise ValueError('NOT IMPLEMENT YET')
        return valid_acc, valid_acc_avg, time_cost, test_acc, test_acc_avg
        
    ##########################################################################
    @staticmethod
    def sample(dataset,
        seed = 999        
    ):
        random_shuffle = np.random.permutation(range(len(dataset)))

        train_data = [dataset[i] for i in random_shuffle[:int(len(dataset)*0.9)]]
        test_data = [dataset[i] for i in random_shuffle[int(len(dataset)*0.9):]]

        return train_data, test_data

    ##########################################################################
    @staticmethod
    def pg_graph_to_nb201(pg_graph):
        # first tensor node attributes, second is the edge list
        ops = [OPS_by_IDX_201[i] for i in pg_graph.x.cpu().numpy()]
        matrix = np.array(to_dense_adj(pg_graph.edge_index)[0].cpu().numpy())
        try: 
            if (matrix == ADJACENCY).all():
                steps_coding = ['0', '0', '1', '0', '1', '2']

                node_1='|'+ops[1]+'~'+steps_coding[0]+'|'
                node_2='|'+ops[2]+'~'+steps_coding[1]+'|'+ops[3]+'~'+steps_coding[2]+'|'
                node_3='|'+ops[4]+'~'+steps_coding[3]+'|'+ops[5]+'~'+steps_coding[4]+'|'+ops[6]+'~'+steps_coding[5]+'|'
                nodes_nb201=node_1+'+'+node_2+'+'+node_3
                index = nasbench.query_index_by_arch(nodes_nb201)
                acc = Dataset.map_item(index).acc
            else:
                acc = torch.zeros(1)
        except:
            acc = torch.zeros(1)
        
        return acc

##############################################################################
#
#                              Debugging
#
##############################################################################

if __name__ == "__main__":
    
    def print_keys(d, k=None, lvl=0):
        if k is not None:
            print(f"{'---'*(lvl)}{k}")
        if type(d) == list and len(d) == 1:
            d = d[0]
        if type(d) == dict:
            for k in d:
                print_keys(d[k], k, lvl+1)
                
    ds = Dataset(10)
    for batch in ds.dataloader:
        print(batch)
        break
