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

import os, glob, json
if __name__ == "__main__":
    from utils_data import prep_data
else:
    from .utils_data import prep_data


OP_PRIMITIVES_ENAS= [
        'output',
        'input',
        'conv3',
        'sep3',
        'conv5',
        'sep5',
        'avg3',
        'max3',
]        


class Dataset:
            
    ##########################################################################
    def __init__(
            self,
            batch_size,
        ):

        if __name__ == "__main__":
            path = os.path.join(".","ENAS") 
        else:
            path = os.path.join(".","datasets","ENAS") 
            
        file_txt = os.path.join(path, "final_structures6.txt")
        file_cache = os.path.join(path, "cache")
        file_cache_train = os.path.join(path, "cache_train")
        file_cache_test = os.path.join(path, "cache_test")


        ############################################        
        burn_in=1000 

        if not os.path.isfile(file_cache):
            self.data = []
            with open(file_txt , 'r') as f:
                for i, row in enumerate(tqdm.tqdm(f)):
                    if i < burn_in:
                            continue
                    if row is None:
                            break
                    row, y = eval(row)
                    self.data.append(Dataset.map_network(row))
                    self.data[-1].acc = torch.FloatTensor([y])  

            print(f"Saving data to cache: {file_cache}")
            torch.save(self.data, file_cache)
        

        ############################################
        if not os.path.isfile(file_cache_train):
            print(f"Loading data from cache: {file_cache}")
            self.data = torch.load(file_cache)

            self.train_data, self.test_data = Dataset.sample(self.data)
            print('prepare data for Autoencoder Training')
            self.train_data = prep_data(self.train_data, max_num_nodes=8)

            print(f"Saving train data to cache: {file_cache_train}")
            torch.save(self.train_data, file_cache_train)

            print(f"Saving test data to cache: {file_cache_test}")
            torch.save(self.test_data, file_cache_test)
        
        else:
            print(f"Loading train data from cache: {file_cache_train}")
            self.train_data = torch.load(file_cache_train)

            print(f"Loading test data from cache: {file_cache_test}")
            self.test_data = torch.load(file_cache_test)

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
    def map_network(row):
        edge_list=[]
        node_atts=[]
        if type(row) == str:
            row = eval(row)  # convert string to list of lists
        n = len(row)
        node_atts.append(1)
        for i, node in enumerate(row):
            node_type=node[0]+2
            node_atts.append(node_type)
            edges=(i,i+1)
            edge_list.append(edges)
            for j, edge in enumerate(node[1:]):
                if edge == 1:
                    edges=(j,i+1)
                    edge_list.append(edges)
        node_type=0
        node_atts.append(node_type)
        edges=(n,n+1)
        edge_list.append(edges)

        edge_index  = torch.tensor(edge_list).t()
        node_attr  = torch.tensor(node_atts)
        num_nodes =  len(node_attr)

        return Data(edge_index=edge_index.long(), node_atts=node_attr, num_nodes=num_nodes)
        
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
    def get_info_generated_graph(item):
        pass

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