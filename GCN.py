from torch import nn
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    def __init__(self, data, args):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(data.num_features, 16)
        self.conv2 = GCNConv(16, data.num_classes)
        self.dropout = nn.Dropout(args.dropout)
    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight)
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_weight)
        return x

class GCN_3layers(nn.Module):
    def __init__(self, data):
        super(GCN_3layers, self).__init__()
        self.conv1 = GCNConv(data.num_features, 20)
        self.conv2 = GCNConv(20, 20)
        self.conv3 = GCNConv(20, data.num_classes)
        self.dropout = nn.Dropout(0.5)
    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight)
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_weight)
        x = self.dropout(x)
        x = self.conv3(x, edge_index, edge_weight)
        return x
