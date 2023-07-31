import torch
from torch import nn
from torch_geometric.nn import GCNConv, GATConv
from torch.nn import functional as F


class MaskGenerator(nn.Module):
    def __init__(self, data, nhid, dropout, gnn='gcn'):
        super(MaskGenerator, self).__init__()
        if gnn == 'gcn':
            self.conv1 = GCNConv(data.num_features, nhid)
        if gnn == 'gat':
            self.conv1 = GATConv(data.num_features, nhid, 1)
        self.w = nn.Parameter(torch.randn(data.num_nodes, data.num_nodes))
        self.dropout = nn.Dropout(dropout)
        self.feature_loss = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(nhid, nhid),
            nn.ReLU(),
            nn.Linear(nhid, data.num_features),
            nn.Sigmoid()
        )
        self.adj_lin = nn.Linear(2 * nhid, 1)

    def forward(self, x, edge_index, k_hop_edge_index, neg_adj):
        out = self.conv1(x, edge_index)
        feature_out = self.mlp(out)
        out = self.dropout(out)
        concat_out = out[k_hop_edge_index.t()].reshape(out[k_hop_edge_index.t()].shape[0], -1)
        neg_out = out[neg_adj].reshape(neg_adj.shape[0], -1)
        adj = torch.cat((concat_out, neg_out), dim=0)
        adj_out = torch.sigmoid(self.adj_lin(adj)).squeeze(dim=1)
        return feature_out, adj_out

    def mask_loss(self, data, args, pgout_x, pgout_adj, model):
        x, edge_index = data.x, data.sparse_adj
        new_x = torch.mul(pgout_x, x)
        sub_loss = F.mse_loss(pgout_adj, data.adj_label)
        new_pgout_adj = torch.sparse_coo_tensor(data.k_hop_edge_index, pgout_adj[:data.k_hop_edge_index.shape[1]])
        new_adj = edge_index * new_pgout_adj
        new_edge_index = new_adj.coalesce().indices()
        new_edge_weight = new_adj.coalesce().values()
        new_out = model(new_x, new_edge_index, edge_weight=new_edge_weight)
        m_loss = F.cross_entropy(new_out[data.train_mask], data.y[data.train_mask])
        # print(f"adj_loss: {sub_loss}")
        if args.use_adj_loss ==True:
            return m_loss + sub_loss
        if args.use_adj_loss == False:
            return m_loss