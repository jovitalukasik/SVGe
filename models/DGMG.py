import numpy as np
import igraph
import torch
import pdb

from torch.autograd import Variable
from torch.distributions import Bernoulli, Categorical

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn.init import xavier_uniform

from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import softmax, scatter_



def edges2index(edges, finish=False):
    device = edges.device
    batch_size, size = edges.size()
    edge_index = torch.LongTensor(2, 0).to(device)
    num_nodes = int(np.sqrt(2*size+1/4)+.5)
    for idx, batch_edge in enumerate(edges):
        trans = idx*num_nodes
        if finish:
            trans = 0
        i = np.inf
        j = 0
        for k, edge in enumerate(batch_edge):
            if j >= i:
                j -= i
                
            if edge.item() == 0:
                j += 1
                i = int(np.ceil((np.sqrt(2*k+9/4)-.5)))
            else:
                i = int(np.ceil((np.sqrt(2*k+9/4)-.5)))
                edge_index = torch.cat([edge_index, torch.LongTensor([[j+trans], [i+trans]]).to(device)], 1)
                j += 1
    return edge_index


def batch2graph(graphs, stop=10, backward=False):
    graphs = graphs.cpu().numpy()
    output = list()
    for graph in graphs:
        graph = list(graph)
        node_att = graph[0]
        if backward:
            node_atts=[0,node_att]
        else:
            node_atts = [1, node_att]
        edges = list([1])
        idx = 1
        run = 3
        if backward:
           treshold=1
        else:
            treshold=0
        while node_att != treshold:
            if run >= stop:
                break
            node_att = graph[idx]
            node_atts += [node_att]
            edges += graph[idx+1:idx+run]
            idx += run
            run += 1
        edge_index = edges2index(torch.tensor(edges).unsqueeze(0))
        output.append((torch.tensor(node_atts), edge_index))
    return output

### Node embedding

class GNNLayer(MessagePassing): 
    def __init__(self, ndim):
        super(GNNLayer, self).__init__(aggr='add') 
        self.msg = nn.Linear(ndim*2, ndim*2)  
        self.msg_rev = nn.Linear(ndim*2, ndim*2) 
        self.upd = nn.GRUCell(2*ndim, ndim)
        
    def forward(self, edge_index, h):
        return self.propagate(edge_index, h=h)

    def message(self, h_j, h_i):
        m = torch.cat([h_j, h_i], dim=1)
        m, m_reverse = torch.split(m, m.size(0)//2, 0) 
        a = torch.cat([self.msg(m), self.msg_rev(m_reverse)], dim=0)
        return a 
    
    def update(self, aggr_out, h):
        h = self.upd(aggr_out, h)
        return h


class GNNinit(MessagePassing):   
    def forward(self, edge_index, h, h_prev):
        h_diff=self.propagate(edge_index, x=h_prev)
        h=h+h_diff
        return h,h_diff
    
        
    
    
class NodeEmb(nn.Module):
    def __init__(self, ndim, num_layers, num_node_atts, node_dropout, dropout):
        super(NodeEmb,self).__init__()
        
        self.ndim=ndim
        self.num_node_atts=num_node_atts
        self.num_layers = num_layers
        self.dropout = dropout
        self.Dropout = nn.Dropout(dropout)
        self.NodeInit = nn.Embedding(num_node_atts, ndim)
        self.GNNLayers = nn.ModuleList([GNNLayer(ndim) for _ in range(num_layers)])
        self.GNNinit = GNNinit()
         
        
        
    def forward(self, edge_index, node_atts):
        h = self.NodeInit(node_atts)
        edge_index = torch.cat([edge_index, torch.index_select(edge_index, 0, torch.tensor([1, 0]).to(h.device))], 1)   
        for layer in self.GNNLayers: 
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = layer(edge_index, h)
        return h
    
### Graph embedding

class GraphEmb(nn.Module):
    def __init__(self, ndim, gdim, aggr='gsum'):
        super(GraphEmb, self).__init__()
        self.ndim = ndim
        self.gdim = gdim
        self.aggr = aggr
        self.f_m = nn.Linear(ndim, gdim) 
        if aggr == 'gsum': 
            self.g_m = nn.Linear(ndim, 1) 
            self.sigm = nn.Sigmoid()

    def forward(self, h, batch):
        if self.aggr == 'mean':
            h = self.f_m(h)
            return scatter_('mean', h, batch)
        elif self.aggr == 'gsum':
            h_vG = self.f_m(h)  
            g_vG = self.sigm(self.g_m(h)) 
            h_G = torch.mul(h_vG, g_vG)   
            return scatter_('add', h_G, batch)
        

        
class GNNEncoder(nn.Module):
    def __init__(self, ndim, gdim, num_gnn_layers, num_node_atts, node_dropout=.0, g_aggr='gsum', dropout=.0):
        super().__init__()
        self.NodeEmb = NodeEmb(ndim, num_gnn_layers, num_node_atts, node_dropout, dropout)
        self.GraphEmb_mean = GraphEmb(ndim, gdim, g_aggr) 
        self.GraphEmb_var = GraphEmb(ndim, gdim, g_aggr)        
        
    def forward(self, edge_index, node_atts, batch):
        h = self.NodeEmb(edge_index, node_atts)
        h_G_mean = self.GraphEmb_mean(h, batch) 
        h_G_var = self.GraphEmb_var(h, batch) 
        return h_G_mean, h_G_var

    def number_of_parameters(self):
        return(sum(p.numel() for p in self.parameters() if p.requires_grad))  
    
## GNNDecoder
    
class NodeEmbUpd(nn.Module):
    def __init__(self,
                 ndim,
                 num_layers,
                 node_dropout=.0,
                 dropout=.0):
        super().__init__()
        self.ndim = ndim
        self.num_layers = num_layers
        self.dropout = dropout
        self.Dropout = nn.Dropout(dropout)
        self.GNNLayers = nn.ModuleList([GNNLayer(ndim) for _ in range(num_layers)])
        
    def forward(self, h, edge_index):
        edge_index = torch.cat([edge_index, torch.index_select(edge_index, 0, torch.tensor([1, 0]).to(h.device))], 1) 
        for layer in self.GNNLayers:
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = layer(edge_index, h)
        return h
    
    
    
class GraphAggr(nn.Module):
    def __init__(self, ndim, gdim, aggr='gsum'):
        super().__init__()
        self.ndim = ndim
        self.gdim = gdim
        self.aggr = aggr
        self.f_m = nn.Linear(ndim, gdim)
        if aggr == 'gsum':
            self.g_m = nn.Linear(ndim, 1)
            self.sigm = nn.Sigmoid()

    def forward(self, h, idx):
        if self.aggr == 'mean':
            h = self.f_m(h).view(-1, idx, self.gdim)
            return torch.mean(h, 1)
        elif self.aggr == 'gsum':
            h_vG = self.f_m(h)
            g_vG = self.sigm(self.g_m(h))
            h = torch.mul(h_vG, g_vG).view(-1, idx, self.gdim) 
            return torch.sum(h, 1)
        
        
        
class GraphEmbed(nn.Module):
    def __init__(self, ndim, gdim, num_gnn_layers):
        super().__init__()
        self.ndim = ndim
        self.gdim = gdim
        self.num_gnn_layers=num_gnn_layers
        self.NodeEmb = NodeEmbUpd(ndim, num_gnn_layers)
        self.GraphEmb = GraphAggr(ndim, gdim)
        self.GraphEmb_init = GraphAggr(ndim, gdim)
        
    def forward(self, h, edge_index):
        idx = h.size(1)
        h = h.view(-1, self.ndim)
        if idx == 1:
            return h.unsqueeze(1), self.GraphEmb.f_m(h), self.GraphEmb_init.f_m(h)
        else:
            h = self.NodeEmb(h, edge_index)
            h_G = self.GraphEmb(h, idx)
            h_G_init = self.GraphEmb_init(h, idx)
            return h.view(-1, idx, self.ndim), h_G, h_G_init
        

class NodeAdd(nn.Module):
    def __init__(self, gdim, num_node_atts):
        super().__init__()
        self.gdim = gdim
        self.num_node_atts=num_node_atts
        self.f_an = nn.Linear(gdim*2, gdim)
        self.f_an_2 = nn.Linear(gdim, num_node_atts)
        
    def forward(self, h_G, c):
        s = self.f_an(torch.cat([h_G, c], 1))        
        return self.f_an_2(F.relu(s))  
        
        
        
class NodeInit(nn.Module):
    def __init__(self, ndim, gdim, num_node_atts):
        super().__init__()
        self.ndim = ndim
        self.gdim = gdim
        self.num_node_atts=num_node_atts
        self.NodeInits = nn.Embedding(num_node_atts, ndim)
        self.f_init = nn.Linear(ndim+gdim*2, ndim+gdim)
        self.f_init_2 = nn.Linear(ndim+gdim, ndim)
        self.f_start = nn.Linear(ndim+gdim, ndim+gdim) 
        self.f_start_2 = nn.Linear(ndim+gdim, ndim)
        
    def forward(self, h_G_init, node_atts, c):
        e = self.NodeInits(node_atts)
        if isinstance(h_G_init, str):
            h_inp = self.f_start(torch.cat([e, c], 1))
            return self.f_start_2(F.relu(h_inp))
        h_v = self.f_init(torch.cat([e, h_G_init, c], 1))
        return self.f_init_2(F.relu(h_v))
    
    
    
class Nodes(nn.Module): 
    def __init__(self, ndim, gdim):
        super().__init__()
        self.ndim = ndim
        self.gdim = gdim
        self.f_s_1 = nn.Linear(ndim*2+gdim*2, ndim+gdim)
        self.f_s_2 = nn.Linear(ndim+gdim, 1)
    
    def forward(self, h, h_v, h_G, c):
        idx = h.size(1)
        s = self.f_s_1(torch.cat([h.view(-1, self.ndim),
                                  h_v.unsqueeze(1).repeat(1, idx, 1).view(-1, self.ndim),
                                  h_G.repeat(idx, 1),
                                  c.repeat(idx, 1)], dim=1)) 
        return self.f_s_2(F.relu(s)).view(-1, idx)
    
    
    
    
class Generator(nn.Module):
    def __init__(self, ndim, gdim, num_gnn_layers, num_node_atts, alpha=.5, stop=20):
        super().__init__()
        self.ndim = ndim
        self.gdim = gdim
        self.num_gnn_layers=num_gnn_layers
        self.num_node_atts=num_node_atts
        self.alpha = alpha
        self.prop = GraphEmbed(ndim, gdim,num_gnn_layers) 
        self.nodeAdd = NodeAdd(gdim, num_node_atts) 
        self.nodeInit = NodeInit(ndim, gdim, num_node_atts) 
        self.nodes = Nodes(ndim, gdim)
        self.node_criterion = torch.nn.CrossEntropyLoss(reduction='none') 
        self.edge_criterion = torch.nn.BCEWithLogitsLoss(reduction='none') 
        self.stop = stop

        
    def forward(self, h, c, edge_index, node_atts, edges):
        h, h_G, h_G_init = self.prop(h, edge_index) 
        node_score = self.nodeAdd(h_G, c)
        node_loss = self.node_criterion(node_score, node_atts)    
        h_v = self.nodeInit(h_G_init, node_atts, c) 
    
        if h.size(1) == 1: 
            h = torch.cat([h, h_v.unsqueeze(1)], 1)
            return h, 2*(1-self.alpha)*node_loss
        edge_score = self.nodes(h, h_v, h_G, c) 
        edge_loss = torch.mean(self.edge_criterion(edge_score, edges), 1)
        h = torch.cat([h, h_v.unsqueeze(1)], 1)
        return h, 2*((1-self.alpha)*node_loss + self.alpha*edge_loss)
    
    def inference(self, h, c, edge_index):
        h, h_G, h_G_init = self.prop(h, edge_index)
        node_logit = self.nodeAdd(h_G, c)
        node_atts = Categorical(logits=node_logit).sample().long()
        non_zero = (node_atts != 0)
        h_v = self.nodeInit(h_G_init, node_atts, c)
        
        if h.size(1) == 1:
            edges = torch.ones_like(node_atts).unsqueeze(1)
            h = torch.cat([h, h_v.unsqueeze(1)], 1)
            return h, node_atts.unsqueeze(1), edges, non_zero
        
        edge_logit = self.nodes(h, h_v, h_G, c)
        edges = Bernoulli(logits=edge_logit).sample().long()
        h = torch.cat([h, h_v.unsqueeze(1)], 1)
        
        if h.size(1) >= self.stop:
            print('stop generating, since max size ({}) was reached'.format(self.stop))
            non_zero = torch.zeros_like(non_zero)
             
        
        return h, node_atts.unsqueeze(1), edges, non_zero

    
    
    
class GNNDecoder(nn.Module):  
    def __init__(self, ndim, gdim, num_gnn_layers, num_node_atts, max_n):
        super().__init__()
        self.ndim = ndim
        self.gdim = gdim
        self.num_gnn_layers=num_gnn_layers
        self.num_node_atts=num_node_atts
        self.max_n=max_n
        self.generator = Generator(ndim, gdim, num_gnn_layers, num_node_atts)

    def forward(self, batch_list, c, nodes):
        h = self.generator.nodeInit('start', torch.ones_like(batch_list[1].node_atts), c).unsqueeze(1) 
        loss_list = torch.Tensor().to(c.device)
        edge_index = 0
        for batch in batch_list[:self.max_n-1]:
            h, loss = self.generator(h,
                                     c,
                                     edge_index,
                                     batch.node_atts,
                                     batch.edges
                                    )
            loss_list = torch.cat([loss_list, loss.unsqueeze(1)], 1)
            edge_index = batch.edge_index
        return torch.mean(torch.sum(torch.mul(loss_list, nodes), 1), 0)

    
    def inference(self, c):
        batch_size = c.size(0)
        h = self.generator.nodeInit('start', torch.ones(batch_size, dtype=torch.long).to(c.device), c).unsqueeze(1)        
        h, node_atts, edges, non_zeros = self.generator.inference(h, c, None) 
        graph = node_atts.clone()
        node_atts = torch.cat([edges, node_atts], 1) 
        num_zeros = (non_zeros == 0).sum().item()
        while num_zeros < batch_size:
            edge_index = edges2index(edges)
            h, node_atts_new, edges_new, non_zero = self.generator.inference(h, c, edge_index)
            graph = torch.cat([graph, node_atts_new, edges_new], 1)
            node_atts = torch.cat([node_atts, node_atts_new], 1) 
            edges = torch.cat([edges, edges_new], 1)
            non_zeros = torch.mul(non_zeros, non_zero)
            num_zeros = (non_zeros == 0).sum().item()
        return batch2graph(graph), node_atts.view(batch_size, -1), edges2index(edges, finish=True)
    

class DGMG(nn.Module):
    def __init__(self, model_config, data_config,  START_TYPE=1 , END_TYPE=0):

        super().__init__()
        self.ndim= model_config['node_embedding_dim']
        self.gdim=model_config['graph_embedding_dim']
        self.num_gnn_layers=model_config['gnn_iteration_layers']
        self.num_node_atts=data_config['num_node_atts']
        self.beta=model_config['beta']
        self.config = model_config
        self.max_n=data_config['max_num_nodes']

        self.Encoder = GNNEncoder( model_config['node_embedding_dim'], model_config['graph_embedding_dim'], model_config['gnn_iteration_layers'], data_config['num_node_atts'])
        self.Decoder = GNNDecoder( model_config['node_embedding_dim'], model_config['graph_embedding_dim'], model_config['gnn_iteration_layers'], data_config['num_node_atts'], data_config['max_num_nodes'])
        self.START_TYPE=START_TYPE
        self.END_TYPE=END_TYPE
        
        
    def sample(self, mean, log_var, eps_scale=0.01):      
        if self.training:
            std = log_var.mul(0.5).exp_()
            eps = torch.randn_like(std) * eps_scale
            return eps.mul(std).add_(mean)
        else:
            return mean        
        
    def forward(self, batch_list):
        h_G_mean, h_G_var = self.Encoder(batch_list[0].edge_index,
                                         batch_list[0].node_atts,
                                         batch_list[0].batch)
        c = self.sample(h_G_mean, h_G_var)
        kl_loss = -0.5 * torch.sum(1 + h_G_var - h_G_mean.pow(2) - h_G_var.exp())
        recon_loss = self.Decoder(batch_list[1:], c, batch_list[0].nodes) 

        return recon_loss+ self.beta*kl_loss , recon_loss, kl_loss
    
    
    
    def inference(self, data, sample=False, log_var=None):
        if isinstance(data, torch.Tensor): 
            if data.size(-1) != self.gdim:
                raise Exception('Size of input is {}, must be {}'.format(data.size(0), self.ndim*2))
            if data.dim() == 1:
                mean = data.unsqueeze(0)
            else:
                mean = data
        elif isinstance(data, Data):
            if not data.__contains__('batch'):
                data.batch = torch.LongTensor([0]).to(data.edge_index.device)
            mean, log_var = self.Encoder(data.edge_index, data.node_atts, data.batch)
           
        if sample:
            c = self.sample(mean, log_var)
        else:
            c = mean
            log_var = 0
       
        edges, node_atts, edge_list = self.Decoder.inference(c)
               
        return edges, node_atts, edge_list, mean, log_var, c
    
    def number_of_parameters(self):
        return(sum(p.numel() for p in self.parameters() if p.requires_grad))
    
   
# Accuracy Prediction
class GetAcc(nn.Module): 
    def __init__(self,  model_config, dim_target=1):
        super(GetAcc, self).__init__()
        self.gdim = model_config['graph_embedding_dim']
        self.num_layers = model_config['num_regression_layers']
        self.dropout = model_config['dropout']
        self.dim_target=dim_target
        self.lin_layers = nn.ModuleList([nn.Linear(self.gdim//(2**num), self.gdim//(2**(num+1))) for num in range(self.num_layers-1)]) 
        self.lin_layers.append(nn.Linear(self.gdim//(2**(self.num_layers-1)), dim_target))
        
    def forward(self, h):
        for layer in self.lin_layers[:-1]:
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = F.relu(layer(h)) 
        h = self.lin_layers[-1](h)
        return h
    
    def __repr__(self):
        return '{}({}x Linear) Dropout(p={})'.format(self.__class__.__name__,
                                                self.num_layers,
                                                self.dropout
                                              )  

class DGMG_acc(nn.Module):
    def __init__(self, model_config, data_config,  START_TYPE=1 , END_TYPE=0, dim_target=1):
        super().__init__()
        self.ndim= model_config['node_embedding_dim']
        self.gdim= model_config['graph_embedding_dim']
        self.num_gnn_layers=model_config['gnn_iteration_layers']
        self.num_node_atts=data_config['num_node_atts']
        self.beta=model_config['beta']
        self.model_config = model_config
        self.max_n=data_config['max_num_nodes']
        self.dim_target= dim_target

        self.Encoder = GNNEncoder( model_config['node_embedding_dim'], model_config['graph_embedding_dim'], model_config['gnn_iteration_layers'], data_config['num_node_atts'])
        self.Decoder = GNNDecoder( model_config['node_embedding_dim'], model_config['graph_embedding_dim'], model_config['gnn_iteration_layers'], data_config['num_node_atts'], data_config['max_num_nodes'])
        self.START_TYPE=START_TYPE
        self.END_TYPE=END_TYPE
        
        self.Accuracy = GetAcc(model_config, dim_target)

        self.START_TYPE=START_TYPE
        self.END_TYPE=END_TYPE
        
        
    def sample(self, mean, log_var, eps_scale=0.01):      
        if self.training:
            std = log_var.mul(0.5).exp_()
            eps = torch.randn_like(std) * eps_scale
            return eps.mul(std).add_(mean)
        else:
            return mean        
        
    def forward(self, batch_list):
        h_G_mean, h_G_var = self.Encoder(batch_list[0].edge_index,
                                         batch_list[0].node_atts,
                                         batch_list[0].batch)
        acc = self.Accuracy(h_G_mean) 
        c = self.sample(h_G_mean, h_G_var)
        kl_loss = -0.5 * torch.sum(1 + h_G_var - h_G_mean.pow(2) - h_G_var.exp())
        recon_loss = self.Decoder(batch_list[1:], c, batch_list[0].nodes) 

        return recon_loss+ self.beta*kl_loss , recon_loss, kl_loss, acc
    
    
    def inference(self, data, sample=False, log_var=None, only_acc=False):
        if isinstance(data, torch.Tensor): 
            if data.size(-1) != self.gdim:
                raise Exception('Size of input is {}, must be {}'.format(data.size(0), self.ndim*2))
            if data.dim() == 1:
                mean = data.unsqueeze(0)
            else:
                mean = data
        elif isinstance(data, Data):
            if not data.__contains__('batch'):
                data.batch = torch.LongTensor([0]).to(data.edge_index.device)
            mean, log_var = self.Encoder(data.edge_index, data.node_atts, data.batch)
           
        acc = self.Accuracy(mean) 
        if sample:
            c = self.sample(mean, log_var)
        else:
            c = mean
            log_var = 0

        if only_acc:
            return  acc
       
        edges, node_atts, edge_list = self.Decoder.inference(c)
               
        return edges, node_atts, edge_list, mean, log_var, c, acc
    
    
    def number_of_parameters(self):
        return(sum(p.numel() for p in self.parameters() if p.requires_grad))