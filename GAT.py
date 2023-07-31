from torch import nn
from torch_geometric.nn import GATConv


class GAT(nn.Module):
    def __init__(self, data, args):
        super(GAT, self).__init__()
        self.conv1 = GATConv(data.num_features, 16, heads=8)
        self.conv2 = GATConv(16 * 8, data.num_classes, heads=1)
        self.dropout = nn.Dropout(args.dropout)
    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight)
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_weight)
        return x
