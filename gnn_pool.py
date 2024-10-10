import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn


class GNNpool(nn.Module):
    def __init__(self, input_dim, conv_hidden, mlp_hidden, num_clusters, device,activ):
        """
        implementation of mincutpool model from: https://arxiv.org/pdf/1907.00481v6.pdf
        @param input_dim: Size of input nodes features
        @param conv_hidden: Size Of conv hidden layers
        @param mlp_hidden: Size of mlp hidden layers
        @param num_clusters: Number of cluster to output
        @param device: Device to run the model on
        """
        super(GNNpool, self).__init__()
        self.device = device
        self.num_clusters = num_clusters
        self.mlp_hidden = mlp_hidden
        self.activ = activ
        # GNN conv
        # self.convs = pyg_nn.GCN(input_dim, conv_hidden, 1, act=activ)
        # # MLP
        # self.mlp = nn.Sequential(
        #     nn.Linear(conv_hidden, mlp_hidden), nn.SELU(), nn.Dropout(0.25),
        #     nn.Linear(mlp_hidden, self.num_clusters))
    
        
        # https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.GCN.html#torch_geometric.nn.models.GCN
        # # GNN conv
        # self.convs = pyg_nn.GCN(input_dim, conv_hidden, 1, act='selu')
        # # MLP
        # self.mlp = nn.Sequential(
        #     nn.Linear(conv_hidden, mlp_hidden), nn.SELU(), nn.Dropout(0.25),
        #     nn.Linear(mlp_hidden, self.num_clusters))
        
        # GNN conv
        if activ == "SELU":
            self.convs = pyg_nn.GCN(input_dim, conv_hidden, 2, act='selu',dropout= 0.25)
        # MLP
            self.mlp = nn.Sequential(
            nn.Linear(conv_hidden, mlp_hidden), nn.SELU(), nn.Dropout(0.25),
            nn.Linear(mlp_hidden, self.num_clusters))
        elif activ == "SiLU":
            self.convs = pyg_nn.GCN(input_dim, conv_hidden, 2, act='silu', dropout= 0.2)
            self.convs2 = pyg_nn.GCN(input_dim, conv_hidden//2, 2, act='silu', dropout= 0.2)
        # MLP
            self.mlp = nn.Sequential(
            nn.Linear(conv_hidden, mlp_hidden), nn.SiLU(), nn.Dropout(0.2),
            nn.Linear(mlp_hidden, self.num_clusters))

        elif activ == "SiLU_GAT":
            print('SiLU_GAT')
            # self.convs = pyg_nn.GCN(input_dim, conv_hidden, 2, act='silu', dropout= 0.2)
            self.convs = pyg_nn.GAT(input_dim, conv_hidden//2, 2, act='silu', dropout= 0.2, edge_dim=1)
            self.convs2 = pyg_nn.GAT(input_dim, conv_hidden, 2, act='silu', dropout= 0.2, edge_dim=1)
        # MLP
            self.mlp = nn.Sequential(
            nn.Linear(conv_hidden//2, mlp_hidden), nn.SiLU(), nn.Dropout(0.2),
            nn.Linear(mlp_hidden, self.num_clusters))
            self.mlp2 = nn.Sequential(
            nn.Linear(conv_hidden, conv_hidden//2 ), nn.SiLU(), nn.Dropout(0.2))

        elif activ == "SiLU_GAT1":
            print('SiLU_GAT1')
            # self.convs = pyg_nn.GCN(input_dim, conv_hidden, 2, act='silu', dropout= 0.2)
            self.convs = pyg_nn.GAT(input_dim, conv_hidden, 2, act='silu', dropout= 0.2, edge_dim=1)
        # MLP
            self.mlp = nn.Sequential(
            nn.Linear(conv_hidden, mlp_hidden), nn.SiLU(), nn.Dropout(0.2),
            nn.Linear(mlp_hidden, self.num_clusters))
        elif activ == "SiLU_GAT1_1":
            print('SiLU_GAT1_1')
            # self.convs = pyg_nn.GCN(input_dim, conv_hidden, 2, act='silu', dropout= 0.2)
            self.convs = pyg_nn.GAT(input_dim, conv_hidden, 1, act='silu', dropout= 0.2, edge_dim=1)
        # MLP
            self.mlp = nn.Sequential(
            nn.Linear(conv_hidden, mlp_hidden), nn.SiLU(), nn.Dropout(0.2),
            nn.Linear(mlp_hidden, self.num_clusters))

        elif activ == "SiLU_GATt2":
            print('SiLU_GATt2')
            # self.convs = pyg_nn.GCN(input_dim, conv_hidden, 2, act='silu', dropout= 0.2)
            self.convs = pyg_nn.GAT(input_dim, conv_hidden//3, 2, act='silu', dropout= 0.2, edge_dim=1)
            self.convs2 = pyg_nn.GAT(input_dim, conv_hidden//2, 2, act='silu', dropout= 0.2, edge_dim=1)
        # MLP
            self.mlp = nn.Sequential(
            nn.Linear(conv_hidden//3, mlp_hidden), nn.SiLU(), nn.Dropout(0.2),
            nn.Linear(mlp_hidden, self.num_clusters))
            self.mlp2 = nn.Sequential(
            nn.Linear(conv_hidden//2, conv_hidden//3 ), nn.SiLU(), nn.Dropout(0.2))

        elif activ == "SiLU_GATs2":
            print('SiLU_GATs2')
            # self.convs = pyg_nn.GCN(input_dim, conv_hidden, 2, act='silu', dropout= 0.2)
            self.convs = pyg_nn.GAT(input_dim, conv_hidden//2, 2, act='silu', dropout= 0.2, edge_dim=1)
            self.convs2 = pyg_nn.GAT(input_dim, conv_hidden, 2, act='silu', dropout= 0.2, edge_dim=1)
            self.convs3 = pyg_nn.GAT(conv_hidden//2, conv_hidden//3, 2, act='silu', dropout= 0.2, edge_dim=1)
            self.convs4 = pyg_nn.GAT(conv_hidden//2, conv_hidden//4, 2, act='silu', dropout= 0.2, edge_dim=1)
        # MLP
            self.mlp = nn.Sequential(
            nn.Linear(conv_hidden//4, mlp_hidden), nn.SiLU(), nn.Dropout(0.2),
            nn.Linear(mlp_hidden, self.num_clusters))
            self.mlp2 = nn.Sequential(
            nn.Linear(conv_hidden, conv_hidden//2 ), nn.SiLU(), nn.Dropout(0.2))
            self.mlp3 = nn.Sequential(
            nn.Linear(conv_hidden//3, conv_hidden//4 ), nn.SiLU(), nn.Dropout(0.2))
            

        elif activ == "SiLU_GAT2":
            # self.convs = pyg_nn.GCN(input_dim, conv_hidden, 2, act='silu', dropout= 0.2)
            self.convs = pyg_nn.GAT(input_dim, conv_hidden, 2, act='silu', dropout= 0.2, edge_dim=1)
            self.convs2 = pyg_nn.GAT(input_dim, conv_hidden//2, 2, act='silu', dropout= 0.2, edge_dim=1)
           # MLP
            self.mlp = nn.Sequential(
            nn.Linear(conv_hidden, conv_hidden//2), nn.SiLU(), nn.Dropout(0.2),  
            nn.Linear(conv_hidden//2, mlp_hidden), nn.SiLU(), nn.Dropout(0.2),
            nn.Linear(mlp_hidden, self.num_clusters))

        elif activ == "SiLU_GAT3":
            print('SiLU_GAT3')
            # self.convs = pyg_nn.GCN(input_dim, conv_hidden, 2, act='silu', dropout= 0.2)
            self.convs = pyg_nn.GAT(input_dim, conv_hidden//4, 1, act='silu', dropout= 0.2, edge_dim=1)
            self.convs2 = pyg_nn.GAT(input_dim, conv_hidden//2, 1, act='silu', dropout= 0.2, edge_dim=1)
            self.convs3 = pyg_nn.GAT(input_dim, conv_hidden, 1, act='silu', dropout= 0.2, edge_dim=1)
           # MLP
            self.mlp = nn.Sequential(
            nn.Linear(conv_hidden//4, mlp_hidden), nn.SiLU(), nn.Dropout(0.2),
            nn.Linear(mlp_hidden, self.num_clusters))
            self.mlp2 = nn.Sequential(
            nn.Linear(conv_hidden//2, conv_hidden//4 ), nn.SiLU(), nn.Dropout(0.2))
            self.mlp3 = nn.Sequential(
            nn.Linear(conv_hidden, conv_hidden//4 ), nn.SiLU(), nn.Dropout(0.2))                        

        elif activ == "GELU":
            self.convs = pyg_nn.GCN(input_dim, conv_hidden, 1, act='gelu',dropout= 0.25)
        # MLP
            self.mlp = nn.Sequential(
            nn.Linear(conv_hidden, mlp_hidden), nn.GELU(), nn.Dropout(0.25),
            nn.Linear(mlp_hidden, self.num_clusters))
        elif activ == "ELU":
            # GNN conv
            self.convs = pyg_nn.GCN(input_dim, conv_hidden, 1, act='elu',dropout= 0.25) ### mincutpool
          # MLP
            self.mlp = nn.Sequential(
            nn.Linear(conv_hidden, mlp_hidden), nn.ELU(), nn.Dropout(0.25),
            nn.Linear(mlp_hidden, self.num_clusters))
        else:
            
            # print("Hi3")
              # GNN conv            
            self.convs = pyg_nn.GCN(input_dim, conv_hidden, 1, act='relu',dropout= 0.25) ### mincutpool
              # MLP
            print("Hi4")

            self.mlp = nn.Sequential(
            nn.Linear(conv_hidden, mlp_hidden), nn.ELU(), nn.Dropout(0.25),
            nn.Linear(mlp_hidden, self.num_clusters))
            

    def forward(self, data, A):
        """
        forward pass of the model
        @param data: Graph in Pytorch geometric data format
        @param A: Adjacency matrix of the graph
        @return: Adjacency matrix of the graph and pooled graph (argmax of S)
        """
        x, edge_index, edge_atrr = data.x, data.edge_index, data.edge_attr

        if self.activ == "SiLU_GAT":
            x1 = self.convs(x, edge_index, edge_atrr)  # applying con5v
            x2 = self.convs2(x, edge_index, edge_atrr)  # applying con5v
            x = self.mlp2(x2) + x1
        elif self.activ == "SiLU_GATt2":
            #print('fwt2')
            x1 = self.convs(x, edge_index, edge_atrr)  # applying con5v
            x2 = self.convs2(x, edge_index, edge_atrr)  # applying con5v
            x = self.mlp2(x2) + x1
        elif self.activ == "SiLU_GAT1":
            x = self.convs(x, edge_index, edge_atrr)
        elif self.activ == "SiLU_GATs2":
            x1 = self.convs(x, edge_index, edge_atrr)  # applying con5v
            x2 = self.convs2(x, edge_index, edge_atrr)  # applying con5v
            x = self.mlp2(x2) + x1
            x3 = self.convs3(x, edge_index, edge_atrr)
            x4 = self.convs4(x, edge_index, edge_atrr)
            x = self.mlp3(x3) + x4
        elif self.activ == "SiLU_GAT3":
            x1 = self.convs(x, edge_index, edge_atrr)  # applying con5v
            x2 = self.convs2(x, edge_index, edge_atrr)  # applying con5v
            x3 = self.convs3(x, edge_index, edge_atrr)  # applying con5v
            x = self.mlp3(x3) + self.mlp2(x2) + x1  
        else:
            x = self.convs(x, edge_index, edge_atrr)

        # x = (x1 + x2) / 2
        x = F.silu(x)

        # if self.activ == "SELU":
        #     x = F.selu(x)
        # elif self.activ == "SiLU":
        #     x = F.silu(x)
        # elif self.activ == "GELU":
        #     x = F.gelu(x)
        # else:
        #     x = F.elu(x)

        # pass feats through mlp
        H = self.mlp(x)
        # cluster assignment for matrix S
        S = F.softmax(H, dim = 1)
        # S = torch.sigmoid(H)
        # So = 1 - torch.sigmoid(H)
        # S = torch.cat([S, So], dim=1)

        return A, S

    def loss(self, A, S):
        """
        loss calculation, relaxed form of Normalized-cut
        @param A: Adjacency matrix of the graph
        @param S: Polled graph (argmax of S)
        @return: loss value
        """
        C = S
        d = torch.sum(A, dim=1)
        m = torch.sum(A)
        B = A - torch.ger(d, d) / (2 * m)

        I_S = torch.eye(self.num_clusters, device=self.device)
        k = torch.norm(I_S)
        n = S.shape[0]

        modularity_term = (-1/(2*m)) * torch.trace(torch.mm(torch.mm(C.t(), B), C))

        collapse_reg_term = (torch.sqrt(k)/n) * (torch.norm(torch.sum(C, dim=0), p='fro')) - 1

        return modularity_term + collapse_reg_term

        # # cut loss
        # A_pool = torch.matmul(torch.matmul(A, S).t(), S)
        # num = torch.trace(A_pool)

        # D = torch.diag(torch.sum(A, dim=-1))
        # D_pooled = torch.matmul(torch.matmul(D, S).t(), S)
        # den = torch.trace(D_pooled)
        # mincut_loss = -(num / den)

        # # orthogonality loss
        # St_S = torch.matmul(S.t(), S)
        # I_S = torch.eye(self.num_clusters, device=self.device)
        # ortho_loss = torch.norm(St_S / torch.norm(St_S) - I_S / torch.norm(I_S))

        # return mincut_loss + ortho_loss
