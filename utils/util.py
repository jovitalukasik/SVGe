######################################################################################
# Parts based on
# Copyright (c) Muhan Zhang, D-VAE, 
# https://github.com/muhanzhang/D-VAE
# modified
######################################################################################

import networkx as nx
from tqdm import tqdm
import numpy as np
from scipy.stats import norm, spearmanr, kendalltau
from sklearn.metrics import mean_squared_error
import torch
import sys
import os
import pickle
import scipy 
import gzip
import collections
import pygraphviz as pgv


sys.path.append('%s/..' % os.path.dirname(os.path.realpath(__file__))) 
sys.path.insert(0, '../')

from torch_geometric.data import Data, DataLoader

from nasbench import api

'''load and save objects'''
def save_object(obj, filename):
    result = pickle.dumps(obj)
    with gzip.GzipFile(filename, 'wb') as dest: dest.write(result)
    dest.close()

def load_object(filename):
    with gzip.GzipFile(filename, 'rb') as source: result = source.read()
    ret = pickle.loads(result)
    source.close()
    return ret

def load_module_state(model, state_name, device):
    pretrained_dict = torch.load(state_name, map_location=device)
    model_dict = model.state_dict()

    # to delete, to correct grud names
    '''
    new_dict = {}
    for k, v in pretrained_dict.items():
        if k.startswith('grud_forward'):
            new_dict['grud'+k[12:]] = v
        else:
            new_dict[k] = v
    pretrained_dict = new_dict
    '''

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict) 
    # 3. load the new state dict
    model.load_state_dict(pretrained_dict)
    return

    
def evaluate_metrics(y_true, y_pred, prediction_is_first_arg):
    """
    Create a dict with all evaluation metrics
    """

    if prediction_is_first_arg:
        y_true, y_pred = y_pred, y_true

    metrics_dict = dict()
    metrics_dict["mse"] = mean_squared_error(y_true, y_pred)
    metrics_dict["rmse"] = np.sqrt(mean_squared_error(y_true, y_pred))
    metrics_dict["kendall_tau"], p_val = kendalltau(y_true, y_pred)
    metrics_dict["spearmanr"] = spearmanr(y_true, y_pred).correlation

    return metrics_dict

class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

            
def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6

def load_module_state(model, state_name):
    pretrained_dict = torch.load(state_name)
    model_dict = model.state_dict()

    # to delete, to correct grud names
    '''
    new_dict = {}
    for k, v in pretrained_dict.items():
        if k.startswith('grud_forward'):
            new_dict['grud'+k[12:]] = v
        else:
            new_dict[k] = v
    pretrained_dict = new_dict
    '''

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict) 
    # 3. load the new state dict
    model.load_state_dict(pretrained_dict)
    return

##############################################################################
#
#                         VAE Abilities
#
##############################################################################


def parse_graph_to_nx(e, label, flat=False):
    try:
        G=nx.DiGraph()
        for i in range(e.shape[1]):
            G.add_edge(e[0][i].item(),e[1][i].item())
        for i in range(len(G)):
            G.nodes[i]['Label']= label[i].item()
    except: 
        G=nx.empty_graph()
    if flat==True:
        adj_flatten=np.array([int(s) for s in e.split(' ')])
        nodes=([int(s) for s in label.split(' ')])
        n=len(nodes)
        matrix=adj_flatten.reshape(n,n)
        G=nx.DiGraph()
        edges=np.transpose(np.transpose(np.nonzero(matrix)))        
        for i in range(edges.shape[1]):
            G.add_edge(edges[0][i].item(),edges[1][i].item())
        for i in range(len(G)):
            G.nodes[i]['Label']= nodes[i]
    return G

def is_same_DAG(g0, g1):
    attr0=(nx.get_node_attributes(g0, 'Label'))
    attr1=(nx.get_node_attributes(g1, 'Label'))
    # note that it does not check isomorphism
    if g0.__len__() != g1.__len__():
        return False
    for vi in range(g0.__len__()):
        if attr0[vi] != attr1[vi]:
            return False
        if set(g0.pred[vi]) != set(g1.pred[vi]):
            return False
    return True

def ratio_same_DAG(G0, G1):
    # how many G1 are in G0
    res = 0
    for g1 in tqdm(G1):
        for g0 in G0:
            if is_same_DAG(g1, g0):
                res += 1
                break
    return res / len(G1)


def is_valid_DAG(g, START_TYPE=0, END_TYPE=1):
    # Check if the given igraph g is a valid DAG computation graph
    # first need to have no directed cycles
    # second need to have no zero-indegree nodes except input
    # third need to have no zero-outdegree nodes except output
    # i.e., ensure nodes are connected
    # fourth need to have exactly one input node
    # finally need to have exactly one output node
    attr=(nx.get_node_attributes(g, 'Label'))
    res = nx.is_directed_acyclic_graph(g)
    n_start, n_end = 0, 0
    for vi in range(g.__len__()):
        if attr[vi] == START_TYPE:
            n_start += 1
        elif attr[vi] == END_TYPE:
            n_end += 1
        if g.in_degree(vi) == 0 and attr[vi] != START_TYPE:
            return False
        if g.out_degree(vi) == 0 and attr[vi] != END_TYPE:
            return False
    return res and n_start == 1 and n_end == 1

def is_valid_ENAS(g, START_TYPE=0, END_TYPE=1):
    # first need to be a valid DAG computation graph
    res = is_valid_DAG(g, START_TYPE, END_TYPE)
    # in addition, node i must connect to node i+1
    for i in range(g.number_of_nodes()-2):
        res=res and nx.node_connectivity(g,i,i+1)!=0
        if not res:
            return res
    # the output node n must not have edges other than from n-1
    res = res and  g.in_degree(g.number_of_nodes()-1)
    return res

def is_valid_NAS(g,nasbench,  START_TYPE=1, END_TYPE=0):
    L = {1:'input', 0:'output', 2:'conv1x1-bn-relu', 3:'conv3x3-bn-relu', 4:'maxpool3x3'}

    # first need to be a valid DAG computation graph
    res = is_valid_DAG(g, START_TYPE, END_TYPE)
    
    node_atts=nx.get_node_attributes(g, 'Label')
    labels=[node_atts[i] for i in range(len(node_atts))]
    node_list = [L[x] for x in np.array(labels)]

    adjacency_matrix=nx.to_numpy_array(g).astype(int)
    # Check for nasbench validity
    try:
        model_spec = api.ModelSpec(matrix=adjacency_matrix, ops=node_list)
        # data = nasbench.query(model_spec)
        if nasbench.is_valid(model_spec):
            return res
        else:
            return False
    except:
        return False
    return res

def is_valid_NAS201(g, START_TYPE=1, END_TYPE=0):
    L = {1:'input' ,0:'output', 2:'nor_conv_1x1', 3:'nor_conv_3x3' , 4: 'avg_pool_3x3', 5:  'skip_connect', 6:'none' }

    # first need to be a valid DAG computation graph
    res = is_valid_DAG(g, START_TYPE, END_TYPE)
    
    node_atts=nx.get_node_attributes(g, 'Label')
    labels=[node_atts[i] for i in range(len(node_atts))]

    try:
        node_list = [L[x] for x in np.array(labels)]
    except:
        return False
    if node_list[0] !='input' or node_list[-1]!='output':
        return False
    return res

def recon_accuracy(test_set_loader, model, device):
    encode_times = 10 # sample embedding 10 times for each Graph
    decode_times = 10 # decode each embedding 10 times 
    n_perfect = 0
    pbar = tqdm(test_set_loader) #10% of Dataset
    for i, graph in enumerate(pbar):
        g_batch=graph.to_data_list()
        g=[parse_graph_to_nx(graphs.edge_index, graphs.node_atts)  for graphs in  g_batch]
        graph=graph.to(device)
        _, _, _, mean, log_var, _ = model.inference(graph, sample=True)
        for _ in range(encode_times):
            _,_,_,_,_, z =  model.inference(mean, sample=True, log_var=log_var)
            for _ in range(decode_times):
                graph_out,_,_,_,_,_ =model.inference(z)
                g_recon=[parse_graph_to_nx(edges, label) for (label, edges) in graph_out]
                n_perfect += sum(is_same_DAG(g0, g1) for g0, g1 in zip(g,g_recon))
    acc = n_perfect / (len(test_set_loader)* test_set_loader.batch_size * encode_times * decode_times)
    print('Recon accuracy from Test Set: {:.5f}'.format(acc))
    return acc


def extract_latent(data_loader, model, device, ENAS=False, NB101=False, NB201=False):
    print('Scaling to Training Data Range')
    Z = []
    Y=[]
    Y_test=[]
    Y_avg=[]
    Y_test_avg=[]
    Y_time=[]
    g_batch = []
    pbar = tqdm(data_loader) 
    for i, graph in enumerate(pbar):
        if isinstance(graph, list):
            graph = graph[0]
        graph.to(device)        
        _, _, _, mean, _, _ = model.inference(graph, sample=True) 
        mean = mean.cpu().detach().numpy()
        Z.append(mean)
        if ENAS==True:
            Y.append(graph.acc.cpu()) 
        elif NB101==True:
            Y.append(graph.acc.cpu()) 
            Y_test.append(graph.test_acc.cpu()) 
            Y_time.append(graph.training_time.cpu()) 
        elif NB201==True:
            Y.append(graph.acc.cpu()) 
            Y_test.append(graph.test_acc.cpu()) 
            Y_time.append(graph.training_time.cpu()) 
            Y_avg.append(graph.acc_avg.cpu())
            Y_test_avg.append(graph.test_acc_avg.cpu())
    if ENAS==True:
        return np.concatenate(Z, 0), torch.cat(Y,0).numpy()
    elif NB101==True:
         return np.concatenate(Z, 0), torch.cat(Y,0).numpy(),  torch.cat(Y_test,0).numpy(),  torch.cat(Y_time,0).numpy()
    elif NB201==True:
        return np.concatenate(Z, 0), torch.cat(Y,0).numpy(),  torch.cat(Y_test,0).numpy(),  torch.cat(Y_time,0).numpy(),  torch.cat(Y_avg,0).numpy(),  torch.cat(Y_test_avg,0).numpy()
            

def save_latent_representations(Dataset, model, device, epoch, path, data_name, ENAS=False, NB101=False, NB201=False):
    data = Dataset(batch_size = 1024)
    if ENAS==True:
        Z_train, Y_train = extract_latent(data.train_dataloader, model, device, ENAS=ENAS)
        Z_test, Y_test = extract_latent(data.test_dataloader, model, device, ENAS=ENAS)
    elif NB101==True:
        Z_train, Y_val_train, Y_test_train, Y_time_train = extract_latent(data.train_dataloader, model, device, NB101=NB101)
        Z_test, Y_val_test, Y_test_test, Y_time_test =  extract_latent(data.test_dataloader, model, device, NB101=NB101)
    else:
        Z_train, Y_val_train, Y_test_train, Y_time_train, Y_val_train_avg, Y_test_train_avg = extract_latent(data.train_dataloader, model, device, NB201=NB201)
        Z_test, Y_val_test, Y_test_test, Y_time_test, Y_val_test_avg, Y_test_test_avg  = extract_latent(data.test_dataloader, model, device, NB201=NB201)

    latent_pkl_name = os.path.join(path,  data_name+
                                   '_latent_epoch{}.pkl'.format(epoch))
    latent_mat_name = os.path.join(path, data_name + 
                                   '_latent_epoch{}.mat'.format(epoch))
    if ENAS==True:
        with open(latent_pkl_name, 'wb') as f:
            pickle.dump((Z_train, Y_train, Z_test, Y_test), f)
        print('Saved latent representations to ' + latent_pkl_name)
        scipy.io.savemat(latent_mat_name, 
                        mdict={
                            'Z_train': Z_train, 
                            'Z_test': Z_test, 
                            'Y_train': Y_train, 
                            'Y_test': Y_test
                            }
                        )
    elif NB101==True:
        with open(latent_pkl_name, 'wb') as f:
            pickle.dump(( Z_train, Y_val_train, Y_test_train, Y_time_train, Z_test, Y_val_test, Y_test_test, Y_time_test ), f)
        print('Saved latent representations to ' + latent_pkl_name)
        scipy.io.savemat(latent_mat_name, 
                        mdict={
                            'Z_train': Z_train, 
                            'Z_test': Z_test, 
                            'Y_val_train': Y_val_train, 
                            'Y_val_test': Y_val_test, 
                            'Y_test_train': Y_test_train,
                            'Y_test_test': Y_test_test,
                            'Y_time_train': Y_time_train, 
                            'Y_time_test': Y_time_test
                            }
                        )
    else:
        with open(latent_pkl_name, 'wb') as f:
            pickle.dump(( Z_train, Y_val_train, Y_test_train, Y_time_train,Y_val_train_avg, Y_test_train_avg , Z_test, Y_val_test, Y_test_test, Y_time_test,Y_val_test_avg, Y_test_test_avg  ), f)
        print('Saved latent representations to ' + latent_pkl_name)
        scipy.io.savemat(latent_mat_name, 
                        mdict={
                            'Z_train': Z_train, 
                            'Z_test': Z_test, 
                            'Y_val_train': Y_val_train, 
                            'Y_val_test': Y_val_test, 
                            'Y_val_train_avg': Y_val_train_avg,
                            'Y_val_test_avg': Y_val_test_avg,
                            'Y_test_train': Y_test_train,
                            'Y_test_test': Y_test_test,
                            'Y_test_train_avg': Y_test_train_avg,
                             'Y_test_test_avg':Y_test_test_avg, 
                            'Y_time_train': Y_time_train, 
                            'Y_time_test': Y_time_test
                            }
                        ) 

def prior_validity(Dataset, model, Z_train , n_latent_points, device, scale_to_train_range=False, ENAS=False, NB101=False, NB201=False):
    dataset = Dataset(batch_size=1)
    if NB101==True:
        from datasets.NASBench101 import nasbench
    if scale_to_train_range:
        z_mean, z_std = Z_train.mean(0), Z_train.std(0)
        z_mean, z_std = torch.FloatTensor(z_mean).to(device), torch.FloatTensor(z_std).to(device)
    nz=z_mean.size(0)
    decode_times = 10
    n_valid = 0
    amount=0
    print('Prior validity experiment begins...')
    G = []
    G_valid = []
    G_valid_str=[]
    pbar = tqdm(range(n_latent_points))
    for i in pbar:
        z = torch.randn(1, nz).to(device)
        z = z * z_std + z_mean  # move to train's latent range
        for j in range(decode_times):
            try:
                g_str, _, _, _, _, _ = model.inference(z)
                for graph in g_str:
                    g=parse_graph_to_nx(graph[1], graph[0])
                    G.extend(g)
                    amount+=1
                    if ENAS==True:
                        if is_valid_ENAS(g, START_TYPE=1, END_TYPE=0):
                            n_valid += 1
                            G_valid_str.append(g_str)
                            G_valid.append(g)
                    elif NB101==True:
                        if is_valid_NAS(g,nasbench, START_TYPE=1, END_TYPE=0):
                            n_valid += 1
                            G_valid_str.append(g_str)
                            G_valid.append(g)
                    elif NB201==True:
                        if is_valid_NAS201(g,  START_TYPE=1, END_TYPE=0):
                            n_valid += 1
                            G_valid_str.append(g_str)
                            G_valid.append(g)
            except:
                continue

    r_valid = n_valid / (n_latent_points * decode_times)
    print('Ratio of valid decodings from the prior: {:.4f}'.format(r_valid))
    print('amount /n:',amount)
    
    
    G_list=[list(g[0]) for g in G_valid_str]
    G_tensor=[torch.cat([g[0].view(-1),g[1].view(-1)]) for g in G_list]
    G_str = [str(g.cpu().numpy()) for g in G_tensor]
    r_unique = len(set(G_str)) / len(G_str) if len(G_str)!=0 else 0.0
    print('Ratio of unique decodings from the prior: {:.4f}'.format(r_unique))
    
    G_train=[]
    for graph in tqdm(dataset.train_dataloader):
        g=parse_graph_to_nx(graph[0].edge_index, graph[0].node_atts)
        G_train.append(g)

    if G_valid == []:
        r_novel = 0
    else:
        r_novel = 1 - ratio_same_DAG(G_train, G_valid)
    print('Ratio of novel graphs out of training data: {:.4f}'.format(r_novel))
    
    return r_valid, r_unique, r_novel

def decode_nx_to_NAS(g):
    #decode an NAS nx to its adajcency matrix and its node_atts
    node_list=nx.get_node_attributes(g, 'Label')
    types = [node_list[i] for i in range(len(node_list))]
    matrix= nx.to_numpy_array(g).astype(int)
    return ( ' '.join(str(x) for x in matrix.reshape(-1)), ' '.join(str(x) for x in types))

def decode_nx_to_ENAS(g):
    # decode an nx to a flattend ENAS string
    n=len(g)
    res = []
    adjlist=[list(g.predecessors(i)) for i in range(len(g))]+ [list(g.successors(len(g)-2))]
    for i in range(1, n-1):
        res.append(nx.get_node_attributes(g, 'Label')[1]-2)
        row = [0] * (i-1)
        for j in adjlist[i]:
            if j < i-1:
                row[j] = 1
        res += row
    return ' '.join(str(x) for x in res)

def decode_from_latent_space(
        latent_points, model, nasbench, decode_attempts=500, n_nodes='variable', return_nx=False, 
        data_type='ENAS'):
    # decode points from the VAE model's latent space multiple attempts
    # and return the most common decoded graphs
    if n_nodes != 'variable':
        check_n_nodes = True  # check whether the decoded graphs have exactly n nodes
    else:
        check_n_nodes = False
    decoded_arcs = []  # a list of lists of igraphs
    pbar = tqdm(range(decode_attempts))
    for i in pbar:
        graph_out,_,_,_,_,_ =model.inference(latent_points)  
        current_decoded_arcs=[parse_graph_to_nx(edges, label) for (label, edges) in graph_out]      
        decoded_arcs.append(current_decoded_arcs)
        pbar.set_description("Decoding attempts {}/{}".format(i, decode_attempts))

    # We see which ones are decoded to be valid architectures
    valid_arcs = []  # a list of lists of strings
    if return_nx:
        str2nxgraph = {}  # map strings to igraphs
    pbar = tqdm(range(latent_points.shape[0]))
    for i in pbar:
        valid_arcs.append([])
        for j in range(decode_attempts):
            arc = decoded_arcs[j][i] # arc is an nx
            if data_type == 'ENAS':
                if is_valid_ENAS(arc, model.START_TYPE, model.END_TYPE):
                    if not check_n_nodes or check_n_nodes and arc.number_of_nodes() == n_nodes:
                        cur = decode_nx_to_ENAS(arc)  # a flat ENAS string
                        if return_nx:
                            str2nxgraph[cur] = arc
                        valid_arcs[i].append(cur)
            elif data_type == 'NB101':
                if is_valid_NAS(arc, nasbench, model.START_TYPE, model.END_TYPE) == True:
                    cur=decode_nx_to_NAS(arc)
                    valid_arcs[i].append(cur)    
                    if return_nx:
                            str2nxgraph[cur] = arc  
            elif data_type== 'NB201':
                if is_valid_NAS201(arc, model.START_TYPE, model.END_TYPE):
                    cur=decode_nx_to_NAS(arc)
                    valid_arcs[i].append(cur)    
                    if return_nx:
                            str2nxgraph[cur] = arc     
        pbar.set_description("Check validity for {}/{}".format(i, latent_points.shape[0]))

    # select the most common decoding as the final architecture
    final_arcs = []  # a list of lists of strings
    pbar = tqdm(range(latent_points.shape[ 0 ]))
    for i in pbar:
        valid_curs = valid_arcs[i]
        aux = collections.Counter(valid_curs)
        if len(aux) > 0:
            arc, num_arc = list(aux.items())[np.argmax(aux.values())]
        else:
            arc = None
            num_arc = 0
        final_arcs.append(arc)
        pbar.set_description("Latent point {}'s most common decoding ratio: {}/{}".format(
                             i, num_arc, len(valid_curs)))

    if return_nx:
        final_arcs_nx = [str2nxgraph[x] if x is not None else None for x in final_arcs]
        return final_arcs_nx, final_arcs
    return final_arcs

##############################################################################
#
#                          Network visualization
#
##############################################################################

def plot_DAG(g, res_dir, name, backbone=False, data_type='ENAS', pdf=False):
    file_name = os.path.join(res_dir, name+'.png')
    if pdf:
        file_name = os.path.join(res_dir, name+'.pdf')
    if data_type == 'ENAS':
        draw_network(g, file_name, backbone)
    elif  data_type=='NB101':
        draw_network_NAS(g, file_name, backbone)
    elif data_type=='NB201':
        draw_network_NAS201(g, file_name, backbone)
    return file_name


def draw_network(g, path, backbone=False):
    graph = pgv.AGraph(directed=True, strict=True, fontname='fixed', arrowtype='open')
    adjlist=[list(g.predecessors(i)) for i in range(len(g))]+ [list(g.successors(len(g)-2))]

    if g is None:
        add_node(graph, 0, 0)
        graph.layout(prog='dot')
        graph.draw(path)
        return
    for idx in range(len(g)):
        add_node(graph, idx, nx.get_node_attributes(g, 'Label')[idx])
    for idx in range(len(g)):
        for node in adjlist[idx]:
            if node == idx-1 and backbone:
                graph.add_edge(node, idx, weight=1)
            else:
                graph.add_edge(node, idx, weight=0)
    graph.layout(prog='dot')
    graph.draw(path)


def add_node(graph, node_id, label, shape='box', style='filled'):
    if label == 1:  
        label = 'input'
        color = '#ff79c6'
    elif label == 0:
        label = 'output'
        color = '#bd93f9'
    elif label == 2:
        label = 'conv3'
        color = '#ffb86c'
    elif label == 3:
        label = 'sep3'
        color = '#e57373'
    elif label == 4:
        label = 'conv5'
        color = '#66d9ef'
    elif label == 5:
        label = 'sep5'
        color = '#ffe082'
    elif label == 6:
        label = 'avg3'
        color = '#1565c0'
    elif label == 7:
        label = 'max3'
        color = '#6272a4'
    else:
        label = ''
        color = 'aliceblue'
    label = f"{label}"
    graph.add_node(
            node_id, label=label, color='black', fillcolor=color,
            shape=shape, style=style, fontsize=24)

def draw_network_NAS(g, path, backbone=False):
    graph = pgv.AGraph(directed=True, strict=True, fontname='Helvetica', arrowtype='open')
    adjlist=[list(g.predecessors(i)) for i in range(len(g))]+ [list(g.successors(len(g)-2))]
    if g is None:
        add_node_NAS(graph, 0, 0)
        graph.layout(prog='dot')
        graph.draw(path)
        return
    for idx in range(len(g)):
        add_node_NAS(graph, idx, nx.get_node_attributes(g, 'Label')[idx])
    for idx in range(len(g)):
        for node in adjlist[idx]:
            if node == idx-1 and backbone:
                graph.add_edge(node, idx, weight=1)
            else:
                graph.add_edge(node, idx, weight=0)
    graph.layout(prog='dot')
    graph.draw(path)
    
    
def add_node_NAS(graph, node_id, label, shape='box', style='filled'):
    if label == 0:  
        label = 'output'
        color='#bd93f9'
    elif label == 1:
        label = 'input'
        color = '#ff79c6'    
    elif label == 2:
        label = 'conv1'
        color = '#66d9ef'
    elif label == 3:
        label = 'conv3'
        color = '#ffb86c'
    elif label == 4:
        label = 'max3'
        color = '#6272a4'
    else:
        label = ''
        color = 'aliceblue'
    label = f"{label}"
    graph.add_node(
            node_id, label=label, color='black', fillcolor=color,
            shape=shape, style=style, fontsize=24)


def draw_network_NAS201(g, path, backbone=False):
    graph = pgv.AGraph(directed=True, strict=True, fontname='fixed', arrowtype='open')
    adjlist=[list(g.predecessors(i)) for i in range(len(g))]+ [list(g.successors(len(g)-2))]

    if g is None:
        add_node_NAS201(graph, 0, 0)
        graph.layout(prog='dot')
        graph.draw(path)
        return
    for idx in range(len(g)):
        add_node_NAS201(graph, idx, nx.get_node_attributes(g, 'Label')[idx])
    for idx in range(len(g)):
        for node in adjlist[idx]:
            if node == idx-1 and backbone:
                graph.add_edge(node, idx, weight=1)
            else:
                graph.add_edge(node, idx, weight=0)
    graph.layout(prog='dot')
    graph.draw(path)


def add_node_NAS201(graph, node_id, label, shape='box', style='filled'):
    if label == 1:  
        label = 'input'
        color = '#ff79c6'
    elif label == 0:
        label = 'output'
        color = '#bd93f9'
    elif label == 2:
        label = 'nor_conv_1x1'
        color = '#ffb86c'
    elif label == 3:
        label = 'nor_conv_3x3'
        color = '#e57373'
    elif label == 4:
        label = 'avg_pool_3x3'
        color = '#66d9ef'
    elif label == 5:
        label = 'skip_connect'
        color = '#ffe082'
    elif label == 6:
        label = 'none'
        color = '#1565c0'
    else:
        label = ''
        color = 'aliceblue'
    label = f"{label}"
    graph.add_node(
            node_id, label=label, color='black', fillcolor=color,
            shape=shape, style=style, fontsize=24)
