import torch 
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, subgraph
from tqdm import tqdm 
    
def sort_edge_index(edge_index, num_nodes):
    idx = edge_index[0]  + edge_index[1]* num_nodes
    perm = idx.argsort()

    return edge_index[:, perm]

def prep_data(data, max_num_nodes, aggr='sum', device='cpu', NB201=False, NB101=False):
    device = torch.device(device)
    data_list=[]
    for graph in tqdm(data):
        node_atts=graph.node_atts.numpy()
        node_atts_reverse=np.flip(graph.node_atts.numpy(),0)
        num_nodes=node_atts.size
        L_list=list(range(num_nodes-1,-1,-1))
        L= { i : L_list[i] for i in range(0,len(L_list)) }
        edge_list= sort_edge_index(graph.edge_index,num_nodes)
        edge_index_reverse= torch.flip(edge_list,[0,1])
        edge_list_reverse=torch.LongTensor(np.stack(([L[x] for x in edge_index_reverse[0].numpy()],[L[x] for x in edge_index_reverse[1].numpy()])))
        edge_list_reverse= sort_edge_index(edge_list_reverse,num_nodes)
        nodes=np.zeros(max_num_nodes-1, dtype=int)
        nodes[:num_nodes-1]=1    
        acc=graph.acc.numpy().item()
        if NB201: 
            test_acc=graph.test_acc.numpy().item()
            acc_avg=graph.acc_avg.numpy().item()
            test_acc_avg=graph.test_acc_avg.numpy().item()
            training_time=graph.training_time.numpy().item()
            data=Data(edge_index=edge_list.to(device),
                        num_nodes=num_nodes, 
                        node_atts=torch.LongTensor(node_atts).to(device),
                        acc=torch.tensor([acc]).to(device),
                        test_acc=torch.tensor([test_acc]).to(device),
                        acc_avg=torch.tensor([acc_avg]).to(device),
                        test_acc_avg=torch.tensor([test_acc_avg]).to(device),
                        training_time=torch.tensor([training_time]).to(device),
                        nodes=torch.tensor(nodes).unsqueeze(0).to(device)
                    )
        elif NB101:
            # try: 
            training_time=graph.training_time.numpy().item()
            test_acc=graph.test_acc.numpy().item()
            data=Data(edge_index=edge_list.to(device),
                        num_nodes=num_nodes, 
                        node_atts=torch.LongTensor(node_atts).to(device),
                        acc=torch.tensor([acc]).to(device),
                        test_acc=torch.tensor([test_acc]).to(device),
                        nodes=torch.tensor(nodes).unsqueeze(0).to(device),
                        training_time=torch.tensor([training_time]).to(device),
                    )
            # except:
        else:
            data=Data(edge_index=edge_list.to(device),
                        num_nodes=num_nodes, 
                        node_atts=torch.LongTensor(node_atts).to(device),
                        acc=torch.tensor([acc]).to(device),
                        nodes=torch.tensor(nodes).unsqueeze(0).to(device)
                    )
        data_full=[data]
        for idx in range(max_num_nodes-1):
            num_nodes=idx+2
            if num_nodes>node_atts.size:
                data=Data(edge_index=subgraph(list(range(2)), edge_list)[0].to(device),
                        num_nodes=num_nodes,
                        node_atts=torch.LongTensor([node_atts[0]]).to(device), 
                        edges=torch.zeros(idx+1).unsqueeze(0).to(device)
                )   
            else:
                data=Data(edge_index=subgraph(list(range(num_nodes)), edge_list)[0].to(device),
                        num_nodes=num_nodes,
                        node_atts=torch.LongTensor([node_atts[idx+1]]).to(device), 
                        edges=to_dense_adj(edge_list)[0][:,idx+1][:idx+1].unsqueeze(0).to(device)
                )
            data_full.append(data)
        for idx in range(max_num_nodes-1):
            num_nodes=idx+2
            if num_nodes>node_atts_reverse.size:
                data=Data(edge_index=subgraph(list(range(2)), edge_list_reverse)[0].to(device),
                        num_nodes=num_nodes,
                        node_atts=torch.LongTensor([node_atts_reverse[0]]).to(device), 
                        edges=torch.zeros(idx+1).unsqueeze(0).to(device)
                ) 
            else:
                data=Data(edge_index=subgraph(list(range(num_nodes)), edge_list_reverse)[0].to(device),
                        num_nodes=num_nodes,
                        node_atts=torch.LongTensor([node_atts_reverse[idx+1]]).to(device), 
                        edges=to_dense_adj(edge_list_reverse)[0][:,idx+1][:idx+1].unsqueeze(0).to(device)
                )
            data_full.append(data)

        data_list.append(tuple(data_full))
    return data_list


def prep_data_8(data, max_num_nodes=8, aggr='sum', device='cpu'):
    device = torch.device(device)
    data_list=[]
    for graph in tqdm(data):
        node_atts=graph.node_atts.numpy()
        node_atts_reverse=np.flip(graph.node_atts.numpy(),0)
        num_nodes=node_atts.size
        L_list=list(range(num_nodes-1,-1,-1))
        L= { i : L_list[i] for i in range(0,len(L_list)) }
        edge_list= sort_edge_index(graph.edge_index,num_nodes)
        edge_index_reverse= torch.flip(edge_list,[0,1])
        edge_list_reverse=torch.LongTensor(np.stack(([L[x] for x in edge_index_reverse[0].numpy()],[L[x] for x in edge_index_reverse[1].numpy()])))
        edge_list_reverse= sort_edge_index(edge_list_reverse,num_nodes)
        nodes=np.zeros(max_num_nodes-1, dtype=int)
        nodes[:num_nodes-1]=1    
        data=Data(edge_index=edge_list.to(device),
                    num_nodes=num_nodes, 
                    node_atts=torch.LongTensor(node_atts).to(device),
                    nodes=torch.tensor(nodes).unsqueeze(0).to(device)
                )
        data_full=[data]
        for idx in range(max_num_nodes-1):
            num_nodes=idx+2
            if num_nodes>node_atts.size:
                data=Data(edge_index=subgraph(list(range(2)), edge_list)[0].to(device),
                        num_nodes=num_nodes,
                        node_atts=torch.LongTensor([node_atts[0]]).to(device), 
                        edges=torch.zeros(idx+1).unsqueeze(0).to(device)
                )   
            else:
                data=Data(edge_index=subgraph(list(range(num_nodes)), edge_list)[0].to(device),
                        num_nodes=num_nodes,
                        node_atts=torch.LongTensor([node_atts[idx+1]]).to(device), 
                        edges=to_dense_adj(edge_list)[0][:,idx+1][:idx+1].unsqueeze(0).to(device)
                )
            data_full.append(data)
        for idx in range(max_num_nodes-1):
            num_nodes=idx+2
            if num_nodes>node_atts_reverse.size:
                data=Data(edge_index=subgraph(list(range(2)), edge_list_reverse)[0].to(device),
                        num_nodes=num_nodes,
                        node_atts=torch.LongTensor([node_atts_reverse[0]]).to(device), 
                        edges=torch.zeros(idx+1).unsqueeze(0).to(device)
                ) 
            else:
                data=Data(edge_index=subgraph(list(range(num_nodes)), edge_list_reverse)[0].to(device),
                        num_nodes=num_nodes,
                        node_atts=torch.LongTensor([node_atts_reverse[idx+1]]).to(device), 
                        edges=to_dense_adj(edge_list_reverse)[0][:,idx+1][:idx+1].unsqueeze(0).to(device)
                )
            data_full.append(data)

        data_list.append(tuple(data_full))
    return data_list
