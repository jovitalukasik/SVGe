import sys, pathlib
import torch
import numpy as np 
import itertools
import copy
import tqdm
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_dense_adj

import torch.nn.functional as F


from nasbench import api
from nasbench.lib import graph_util



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

##############################################################################
# NAS-Bench-101 Data STRUCTURE .tfrecord
##############################################################################
# ---nasbench.hash_iterator() : individual hash for each graph in the whole .tfrecord dataset
# ------nasbench.get_metrics_from_hash(unique_hash): metrics of data sample given by the hash
# ---------fixed_metrics: {'module_adjacency': array([[0, 1, 0, 0, ...type=int8),
#                         'module_operations': ['input', 'conv3x3-bn-relu', 'maxpool3x3', 'conv3x3-bn-relu', 'conv3x3-bn-relu', 'conv1x1-bn-relu', 'output'], 
#                         'trainable_parameters': 8555530}
# ---------computed_metrics : dict including the epoch and all metrics which are computed for each architecture{108: [{...}, {...}, {...}]}
# ------------computed_metrics[108]: [{'final_test_accuracy': 0.9211738705635071, 
#                                  'final_train_accuracy': 1.0, 
#                                  'final_training_time': 1769.1279296875,
#                                  'final_validation_accuracy': 0.9241786599159241,
#                                  'halfway_test_accuracy': 0.7740384340286255, 
#                                  'halfway_train_accuracy': 0.8282251358032227,
#                                  'halfway_training_time': 883.4580078125, 
#                                  'halfway_validation_accuracy': 0.7776442170143127},
#                                  {...},{...}]
##############################################################################

import os, glob, json
if __name__ == "__main__":
    from utils_data import prep_data
else:
    from .utils_data import prep_data


OP_PRIMITIVES_NB101 = [
    'output',
    'input',
    'conv1x1-bn-relu',
    'conv3x3-bn-relu',
    'maxpool3x3'
]

OPS_by_IDX_NB101 = {OP_PRIMITIVES_NB101.index(i):i for i in OP_PRIMITIVES_NB101}
OPS_NB101 = {i:OP_PRIMITIVES_NB101.index(i) for i in OP_PRIMITIVES_NB101}

MATRIX_NB101 = torch.zeros(7, 7)

# Useful constants
INPUT = 'input'
OUTPUT = 'output'
CONV3X3 = 'conv3x3-bn-relu'
CONV1X1 = 'conv1x1-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'
NUM_VERTICES = 7
MAX_EDGES = 9
EDGE_SPOTS = NUM_VERTICES * (NUM_VERTICES - 1) / 2   # Upper triangular matrix
OP_SPOTS = NUM_VERTICES - 2   # Input/output vertices are fixed
ALLOWED_OPS = [CONV3X3, CONV1X1, MAXPOOL3X3]
ALLOWED_EDGES = [0, 1]   # Binary adjacency matrix

epoch=108

nasbench = api.NASBench('datasets/nasbench101/nasbench_only108.tfrecord')
class Dataset:
            
    ##########################################################################
    def __init__(
            self,
            batch_size,
        ):

        if __name__ == "__main__":
            path = os.path.join(".","nasbench101") 
        else:
            path = os.path.join(".","datasets","nasbench101")
            
        file_cache = os.path.join(path, "cache")
        # file_cache_train = os.path.join(path, "cache_train")
        file_cache_train = os.path.join(path, "cache_train_small")
        file_cache_test = os.path.join(path, "cache_test")
        ############################################        

        if not os.path.isfile(file_cache):
            self.data = []
            for unique_hash in tqdm.tqdm(nasbench.hash_iterator()):
                fixed_metrics, computed_metrics = nasbench.get_metrics_from_hash(unique_hash)
                self.data.append(Dataset.map_network(fixed_metrics))
                self.data[-1].val_acc = Dataset.map_item(computed_metrics)[0]
                self.data[-1].acc = Dataset.map_item(computed_metrics)[1]  
                self.data[-1].training_time = Dataset.map_item(computed_metrics)[2]  

            print(f"Saving data to cache: {file_cache}")
            torch.save(self.data, file_cache)
        


        if not os.path.isfile(file_cache_train):
            print(f"Loading data from cache: {file_cache}")
            self.data = torch.load(file_cache)

            self.train_data, self.test_data = Dataset.sample(self.data)

            print('prepare data for Autoencoder Training')
            self.train_data = prep_data(self.train_data[:10_000], max_num_nodes=7, NB101=True) ### small training data only contrains 10k data (faster for checking code)
            print(f"Saving train data to cache: {file_cache_train}")
            torch.save(self.train_data, file_cache_train)

            print(f"Saving test data to cache: {file_cache_test}")
            torch.save(self.test_data, file_cache_test)
        
        else:
            print(f"Loading train data from cache: {file_cache_train}")
            self.train_data = torch.load(file_cache_train)

            print(f"Loading test data from cache: {file_cache_test}")
            self.test_data = torch.load(file_cache_test)[:10_000]

        self.length = len(self.train_data)+ len(self.test_data)

        self.train_dataloader = DataLoader(
            self.train_data,
            shuffle = True,
            num_workers = 0,
            pin_memory = True,
            batch_size = batch_size
        )

        self.test_dataloader = DataLoader(
            self.test_data, 
            shuffle = False,
            num_workers = 0,
            pin_memory = False,
            batch_size = batch_size
        )
        
        # self.dataloader = DataLoader(
        #     self.data,
        #     shuffle = True,
        #     num_workers = 0,
        #     pin_memory = True,
        #     batch_size = batch_size
        # )

    ##########################################################################
    @staticmethod
    def map_item(item):
        test_acc = 0.0
        val_acc = 0.0
        training_time = 0.0
        for repeat_index in range(len(item[epoch])):
            assert len(item[epoch])==3, 'len(computed_metrics[epoch]) should be 3'
            data_point = item[epoch][repeat_index]
            val_acc += data_point['final_validation_accuracy']
            test_acc += data_point['final_test_accuracy']
            training_time += data_point['final_training_time']
        val_acc = val_acc/3.0
        test_acc = test_acc/3.0
        training_time_avg = training_time/3.0
        
        return torch.FloatTensor([val_acc]), torch.FloatTensor([test_acc]), torch.Tensor([training_time_avg])
        
    ##########################################################################
    @staticmethod
    def map_network(item):
        matrix= item['module_adjacency']

        node_operations = item['module_operations']

        node_attr = [OPS_NB101[attr] for attr in node_operations]
        num_nodes = len(node_attr)

        
        edge_index = torch.tensor(np.nonzero(matrix))
        node_attr  = torch.tensor(node_attr)
        scores = torch.tensor(matrix).flatten()
        
        return Data(edge_index=edge_index.long(), node_atts=node_attr, num_nodes=num_nodes)

    
    ##########################################################################
    @staticmethod
    def sample(dataset,
        seed = 999,
        k = None
    ):
        random_shuffle = np.random.permutation(range(len(dataset)))
        train_data = [dataset[i] for i in random_shuffle[:int(len(dataset)*0.9)]]
        test_data = [dataset[i] for i in random_shuffle[int(len(dataset)*0.9):]]

        return train_data, test_data
    ##########################################################################
    @staticmethod
    def get_info_generated_graph(item):
        
        if isinstance(item , list):
            data = []
            for graph in item:
                adjacency_matrix = to_dense_adj(graph.edge_index)[0].cpu().numpy().astype(int)
                ops = [OPS_by_IDX_NB101[attr.item()] for attr in graph.node_atts.cpu()]
                try:
                    spec = api.ModelSpec(matrix = adjacency_matrix, ops= ops)
                    fixed_metrics, computed_metrics = nasbench.get_metrics_from_spec(spec)
                    graph.val_acc = Dataset.map_item(computed_metrics)[0]
                    graph.acc = Dataset.map_item(computed_metrics)[1]
                    graph.training_time = Dataset.map_item(computed_metrics)[2]
                    data.append(graph)  
                except:
                    continue

        else:
            adjacency_matrix = to_dense_adj(item.edge_index)[0].cpu().numpy().astype(int)
            ops = [OPS_by_IDX_NB101[attr.item()] for attr in item.node_atts.cpu()]
            try:
                spec = api.ModelSpec(matrix = adjacency_matrix, ops = ops)
                fixed_metrics, computed_metrics = nasbench.get_metrics_from_spec(spec)
                item.val_acc = Dataset.map_item(computed_metrics)[0]
                item.acc = Dataset.map_item(computed_metrics)[1] 
                item.training_time = Dataset.map_item(computed_metrics)[2] 
                data = item  
            except:
                pass
        return data

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