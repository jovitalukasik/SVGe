import random
import numpy as np
import torch
import itertools
from torch_geometric.data import Data


best_7_nodes=np.array([1, 3, 3, 3, 3, 2, 0])
best_7_edges=np.array([[0, 0, 0, 0, 1, 2, 3, 4, 5],
                       [1, 2, 4, 6, 5, 3, 4, 5, 6]])

n = 7
adjMatrix = [[0 for i in range(n)] for k in range(n)]

edge_u = best_7_edges[0]
edge_v = best_7_edges[1]

# scan the arrays edge_u and edge_v
for i in range(len(edge_u)):
    u = edge_u[i]
    v = edge_v[i]
    adjMatrix[u][v] = 1

node_atts=[]
node_atts.append(best_7_nodes[:-1])
last_row=np.zeros(7)
last_column=[]

cs=list((itertools.product(range(2), repeat=6)))
G=[]

for n in range(2,5):
    node_atts=np.append(best_7_nodes[:-1],[n,0])
    for i in cs:
        last_column=list(i)
        
        last_column.append(1)
        last_column.append(0) 
        last_column=np.array(last_column).reshape(8,1)

        adjacency=np.array(adjMatrix)
        adjacency=np.append(adjacency, [last_row], axis=0)
        adjacency=np.append(adjacency, last_column, axis=1)
        adjacency=np.array(adjacency)
        edge_list=np.transpose(np.nonzero(adjacency))
    
        G.append((node_atts, np.transpose(edge_list)))

first_row=[]
column=np.zeros(7)
for n in range(2,5):
    node_atts=np.append([1,n], best_7_nodes[1:])
    for i in cs:
        first_row=np.append(1, i)
    
        first_row=first_row.reshape(1,7)

        first_column=np.append(0,column).reshape(8,1)
        adjacency=np.array(adjMatrix)
        adjacency=np.append(first_row, adjacency, axis=0)
        adjacency=np.append(first_column, adjacency,axis=1)
        adjacency=np.array(adjacency)
        edge_list=np.transpose(np.nonzero(adjacency))
    
        G.append((node_atts, np.transpose(edge_list)))

node_atts=[2,3,4]
n=np.zeros(6)
for _ in range(1000):
    new=[1]
    for i in n:
        a=random.choice(node_atts)
        i=a
        new.append(i)
    new.append(0)
    N=8
    mat = [[0 for x in range(N)] for y in range(N)]
    for _ in range(N):
         for j in range(5):
            v1 = random.randint(0,N-1)
            v2 = random.randint(0,N-1)
            if(v1 > v2):
                 mat[v1][v2] = 1
            elif(v1 < v2):
                 mat[v2][v1] = 1
    adjacency=np.array(mat).transpose()
    edge_list=np.transpose(np.nonzero(adjacency))
    
    G.append((new, np.transpose(edge_list)))
    
data_list = list()
for graph in G:
    node_atts, edges=graph
    num_nodes = len(node_atts)
    
    data = Data(edge_index=torch.LongTensor((edges)),
            num_nodes=num_nodes,
            node_atts=torch.LongTensor(node_atts),
               )  
    data_list.append(data)  
torch.save(data_list, '../datasets/nasbench101/graphs_8.pth')