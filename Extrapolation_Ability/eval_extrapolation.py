import logging
logging.basicConfig(format="%(asctime)s %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from tqdm import tqdm
import numpy as np
import json 
import os
import sys
import torch
import networkx as nx
import operator
from datetime import datetime
from numpy.random import shuffle as shffle

from torch_geometric.data import Data, DataLoader

from ConfigSpace.read_and_write import json as config_space_json_r_w

from models.SVGe import SVGE_acc



import argparse
parser = argparse.ArgumentParser(description='Extrapolation-Model-Evaluation')
parser.add_argument('--model',                  type=str ,default='SVGE_acc')
parser.add_argument('--name',                   type=str ,default='extrapol')
parser.add_argument('--data_search_space',      type=str, default='ENAS', help= 'Choice between NB101 and ENAS search space, change also path_state_dict')
parser.add_argument("--device",                 type=str, default="cpu")
parser.add_argument('--path_state_dict',        type=str, help='directory to saved model', default='state_dicts/SVGE_acc_ENAS/')
parser.add_argument('--checkpoint',             type=int, default=100, help='Which checkpoint of trained model to load')

args = parser.parse_args()

##############################################################################
#
#                              Runfolder
#
##############################################################################
#Create Log Directory
now = datetime.now()
runfolder = now.strftime("%Y_%m_%d_%H_%M_%S")
runfolder = f"{args.data_search_space}/{args.model}/{runfolder}_{args.name}"

FOLDER_EXPERIMENTS = os.path.join(os.getcwd(), 'Experiments/Extrapolation/')
log_dir = os.path.join(FOLDER_EXPERIMENTS, runfolder)
os.makedirs(log_dir)

# save command line input
cmd_input = 'python ' + ' '.join(sys.argv) + '\n'
with open(os.path.join(log_dir, 'cmd_input.txt'), 'a') as f:
    f.write(cmd_input)
print('Command line input: ' + cmd_input + ' is saved.')



for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(log_dir, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

def main(args):

    ##############################################################################
    #
    #                           Data Config
    #
    ##############################################################################
    if args.data_search_space=='ENAS':
        data_config_path='configs/data_configs/ENAS_configspace.json'
    elif args.data_search_space=='NB101':
        data_config_path='configs/data_configs/NB101_configspace.json'
    else:
        raise TypeError("Unknow Seach Space : {:}".format(args.data_search_space))
    #Get Data specific configs
    data_config = json.load(open(data_config_path, 'r'))

    ##############################################################################
    #
    #                           Model Config
    #
    ##############################################################################
    #Get Model configs
    model_config_path='configs/model_configs/svge_configspace.json'
    model_configspace = config_space_json_r_w.read(open(model_config_path, 'r').read())
    model_config = model_configspace.get_default_configuration().get_dictionary()

    ##############################################################################
    #
    #                              Model
    #
    ##############################################################################
    model=eval(args.model)(model_config=model_config, data_config=data_config).to(args.device)

    path_state_dict=args.path_state_dict
    checkpoint = args.checkpoint

    model_dict = model.state_dict()

    m = torch.load(os.path.join(path_state_dict, f"model_checkpoint{checkpoint}.obj"), map_location=args.device)
    m = {k: v for k, v in m.items() if k in model_dict}

    model_dict.update(m)
    model.load_state_dict(model_dict)
    ##############################################################################
    #
    #                        Load Data
    #
    ##############################################################################
    if args.data_search_space=='ENAS':
        data_loader=torch.load('datasets/ENAS/final_structures12.pth')
    elif args.data_search_space=='NB101':
        data_loader=torch.load('datasets/nasbench101/graphs_8.pth')

    acc=evaluate(model, data_loader , 128, args.device)
    b=torch.topk(acc,5)
    print(b)
    for ind in b.indices.tolist():
        adj = generate_adj(data_loader[ind])
        node_atts = data_loader[ind].node_atts.cpu().numpy()
        if args.data_search_space=='ENAS':
            enas_str = decode_NASdata_to_ENAS(data_loader[ind])
            config_dict = {
                'index': ind,
                'node_atts': node_atts,
                'adj': adj,
                'enas_str': enas_str
                }
        else:
            config_dict = {
                'index': ind,
                'node_atts': node_atts,
                'adj': adj,
                }
        with open(os.path.join(log_dir, 'results.txt'), 'a') as file:
                    json.dump(str(config_dict), file)
                    file.write('\n')


def evaluate(model, data_loader,batch_size,  device):
    model.eval()
    pred_acc=torch.Tensor().to(device)
    data_loader = DataLoader(data_loader, shuffle = False, num_workers = 0, pin_memory = False, batch_size = batch_size)
    for graph in tqdm(data_loader):
        with torch.no_grad():
            graph.to(device)
            output = model.inference(graph, only_acc=True)
        pred_acc=torch.cat([pred_acc, output.view(-1)])
    return pred_acc

def generate_adj(pth_graph):     
    edge_u = pth_graph.edge_index[0]
    edge_v = pth_graph.edge_index[1]

    # our graph has n nodes
    n = pth_graph.num_nodes

    # adjacency matrix - initialize with 0
    adjMatrix = [[0 for i in range(n)] for k in range(n)]


    # scan the arrays edge_u and edge_v
    for i in range(len(edge_u)):
        u = edge_u[i]
        v = edge_v[i]
        adjMatrix[u][v] = 1
        
    return(np.array(adjMatrix))


def parse_graph_to_nx(graph_name):
    G=nx.DiGraph()
    edges=np.transpose(np.asarray( graph_name.edge_index))
    label=np.asarray( graph_name.node_atts)-2
    val_acc= np.asarray( graph_name.acc)
    for i in range(len(edges)):
          G.add_edge(edges[i,0], edges[i,1])
    for i in range(len(G)):
        G.nodes[i]['Label']= label[i]
        G.nodes[i]['label']=str(i)
    return G


def decode_NASdata_to_ENAS(g):
    G=parse_graph_to_nx(g)
    node_type=list(G.nodes(data='Label'))
    node_type.sort(key = operator.itemgetter(0))
    n=nx.adjacency_matrix(G).todense().shape[0]
    res = []
    adjlist=[list(np.nonzero(idx)[1]) for idx in np.transpose(nx.adjacency_matrix(G, nodelist=range(n)).todense()) ]  
    for i in range(1, n-1):
        res.append(node_type[i][1])
        row = [0] * (i-1)
        for j in adjlist[i]:
            if j < i-1:
                row[j] = 1
        res += row
    return ' '.join(str(x) for x in res)


if __name__ == '__main__':
    main(args)