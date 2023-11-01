import torch
import numpy as np
from torch_geometric.utils import k_hop_subgraph, remove_self_loops, to_dense_adj, degree, to_undirected, from_networkx
from GCN import GCN
import random
from torch.optim import Adam
from torch.nn import functional as F
from GenGraph import gen_syn2, gen_syn4, gen_syn5
from torch_geometric.datasets import Planetoid, PolBlogs, Coauthor
import torch_geometric.transforms as T
from torch_geometric.datasets.graph_generator import BAGraph
from torch_geometric.transforms import RemoveIsolatedNodes
import networkx as nx
from matplotlib import pyplot as plt
from torch_geometric.datasets import ExplainerDataset


def load_data(args):
    if args.dataset == 'Cora' or args.dataset == 'Citeseer':
        dataset = Planetoid(root='./datasets/Planetoid', name=args.dataset)
        data = dataset[0]
        if args.dataset == 'Citeseer':
            data = RemoveIsolatedNodes()(data)
        data.num_classes = dataset.num_classes
    if args.dataset == 'bashape':
        dataset = ExplainerDataset(
            graph_generator=BAGraph(num_nodes=300, num_edges=5),
            motif_generator='house',
            num_motifs=80,
            transform=T.Constant(),
        )
        data = dataset[0]
        data.num_classes = dataset.num_classes
    if args.dataset == 'bacom':
        G, label = gen_syn2()
        data = from_networkx(G)
        data.x = torch.tensor(data.feat, dtype=torch.float32)
        data.num_classes = max(label) + 1
        data.y = torch.tensor(label)
    if args.dataset == 'treecycle':
        G, label = gen_syn4()
        data = from_networkx(G)
        data.feat = np.array(data.feat).reshape(-1, 1)
        data.x = torch.tensor(data.feat, dtype=torch.float32)
        data.num_classes = max(label) + 1
        data.y = torch.tensor(label)
    if args.dataset == 'treegrid':
        G, label = gen_syn5()
        data = from_networkx(G)
        data.feat = np.array(data.feat).reshape(-1, 1)
        data.x = torch.tensor(data.feat, dtype=torch.float32)
        data.num_classes = max(label) + 1
        data.y = torch.tensor(label)
    if args.dataset == 'polblogs':
        dataset = PolBlogs(root='./datasets/PolBlogs')
        data = dataset[0]
        data = RemoveIsolatedNodes()(data)
        data.edge_index = to_undirected(data.edge_index)
        data.x = torch.eye(data.num_nodes)
        data.num_classes = dataset.num_classes
    if args.dataset == 'cs':
        dataset = Coauthor(root="./datasets", name="CS")
        data = dataset[0]
        data.num_classes = dataset.num_classes
    data = random_split(args, data)
    data.sparse_adj = torch.sparse_coo_tensor(data.edge_index, torch.ones((data.edge_index.shape[1],)))
    data.k_hop_edge_index, data.neg_adj, data.adj_label, data.node_index = get_k_hop_edge_index(args.khop, data)
    return data


def get_k_hop_edge_index(k, data):
    edge_index = data.sparse_adj
    import warnings
    warnings.filterwarnings('ignore', '.*Sparse CSR tensor support is in beta state.*')
    edge_index_2 = torch.sparse.mm(edge_index, edge_index)
    edge_index_3 = torch.sparse.mm(edge_index_2, edge_index)
    if k == 1:
        edge_index_k = edge_index
    if k == 2:
        edge_index_k = edge_index + edge_index_2
    if k == 3:
        edge_index_k = edge_index + edge_index_2 + edge_index_3

    edge_index_k = remove_self_loops(edge_index_k.coalesce().indices())[0]
    node_index = [0]
    zero_indices = torch.nonzero(to_dense_adj(edge_index_k)[0] == 0, as_tuple=False)
    num_indices = zero_indices.size(0)
    num_samples = edge_index_k.shape[1]
    neg_indices = zero_indices[torch.randperm(num_indices)[:num_samples]]
    adj_label = torch.zeros((2 * num_samples,))
    adj_label[:num_samples] = 1
    return edge_index_k, neg_indices, adj_label, node_index


def random_split(args, data):
    train_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
    val_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
    test_mask = torch.zeros(data.y.size(0), dtype=torch.bool)

    train_idx = np.random.choice(data.num_nodes, int(data.num_nodes * args.train_ratio), replace=False)
    residue = np.array(list(set(range(data.num_nodes)) - set(train_idx)))
    val_idx = np.random.choice(residue, int(data.num_nodes * args.val_ratio), replace=False)
    test_idx = np.array(list(set(residue) - set(val_idx)))

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    return data


def Generate_explanation(prototype, data, khop, node_idx):
    subset, subedges, inv, _ = k_hop_subgraph(node_idx, khop, data.edge_index)
    feat_mask = data.x * prototype[0]
    explain_edge_weight = list()
    prototype_adj = torch.sparse_coo_tensor(data.k_hop_edge_index, prototype[1][:data.k_hop_edge_index.shape[1]])
    for i, j in zip(subedges[0].tolist(), subedges[1].tolist()):
        explain_edge_weight.append(float(torch.round(prototype_adj[i][j], decimals=3)))
    G = nx.Graph()
    G.add_nodes_from(subset.tolist())
    G.add_edges_from([tuple(i) for i in subedges.t().tolist()])

    pos = nx.spring_layout(G)
    color_dict = {0: "r", 1: "g", 2: "y", 3: "c", 4: "m", 5: "b", 6: "darkorange", 7: "slategrey", 8: "r", 9: "g", 10: "y", 11: "c", 12: "m", 13: "b", 14: "darkorange"}
    color_nodes = [color_dict[idx] for idx in data.y[subset].detach().cpu().numpy()]
    plt.figure()
    nx.draw(G, pos, with_labels=True, node_size=300, font_size=8, node_color=np.array(color_nodes))
    # plt.show()
    return feat_mask


def Ground_Truth_Distance(prototype, data):
    '''compute the distance between the prototype and the ground truth explanation in BA_shape'''
    D = degree(data.edge_index[0])
    adj_pro = torch.sparse_coo_tensor(data.k_hop_edge_index, prototype[1][:data.k_hop_edge_index.shape[1]])
    adj_1 = adj_pro * data.sparse_adj
    values = adj_1.coalesce().values()
    distance = 0
    groundtruth = np.zeros((data.edge_index.shape[1],))
    for i in range(2800, data.edge_index.shape[1]):
        if data.edge_index[0][i] >= 300 and data.edge_index[1][i] >= 300:
            groundtruth[i] = 1
    distance = 0
    for i in range(300, 700, 5):
        subset, subedges, _, _ = k_hop_subgraph(i, 2, data.edge_index)
        edge_indices = list()
        edge_index = data.edge_index.detach().cpu().numpy()
        for j in subedges.cpu().numpy().T:
            edge_indices.append(np.where(np.logical_and(edge_index[0] == j[0], edge_index[1] == j[1]))[0][0])
        edge_val = values[edge_indices]
        edge_val = (edge_val / (D[subedges[0]] + D[subedges[1]])).tolist()
        dealed_val = np.array(edge_val) * len(edge_val) / sum(edge_val)
        sub_gt = groundtruth[edge_indices]
        distance += sum(sub_gt-dealed_val)**2 / sub_gt.shape[0]
        sorted_value = [value for value, _ in sorted(zip(edge_val, edge_indices), reverse=True)]
        sorted_neighs = [index for _, index in sorted(zip(edge_val, edge_indices), reverse=True)]

    distance = distance / 80
    return distance


def construct_pos_neg(data, prototype, args):
    feat_pro = prototype[0]
    D = degree(data.edge_index[0])
    new_feat = feat_pro * data.x
    adj_pro = torch.sparse_coo_tensor(data.k_hop_edge_index, prototype[1][:data.k_hop_edge_index.shape[1]])
    pos = list()
    neg = list()
    adj_pro_dense = adj_pro.to_dense()
    adj_pro_dense_clone = adj_pro_dense.clone()
    for i in range(data.num_nodes):
        index = torch.where(data.y == data.y[i])[0]
        idx = index[data.train_mask[index] == False]
        idx2 = torch.where(data.test_mask == True)[0]
        idx3 = torch.where(data.val_mask == True)[0]
        adj_pro_dense_clone[i][idx] = 1
        adj_pro_dense_clone[i][idx2] = 1
        adj_pro_dense_clone[i][idx3] = 1
    for i in range(data.num_nodes):
        neighs = torch.nonzero(adj_pro_dense[i]).squeeze().tolist()
        values = adj_pro_dense[i][neighs]
        values = (1 / D[neighs] * values).tolist()
        if type(neighs) == int:
            neighs = [neighs]
            values = [values]
        sorted_value = [value for value, _ in sorted(zip(values, neighs), reverse=True)]
        sorted_neighs = [index for _, index in sorted(zip(values, neighs), reverse=True)]
        set_num = int(len(sorted_neighs) * args.pos_ratio) + 1
        zero_indices = torch.where(adj_pro_dense_clone[i] == 0)[0]
        if set_num > zero_indices.shape[0]:
            set_num = zero_indices.shape[0]
        pos.append(sorted_neighs[:set_num])
        random_indices = random.sample(zero_indices.tolist(), set_num)
        neg.append(random_indices)
    return pos, neg, new_feat.clone().detach().requires_grad_(True)

def test_fidelity(feat_mask, data, args):
    device = torch.device(args.device)
    model = GCN(data).to(device)
    data = random_split(args, data).to(device)
    optim = Adam(model.parameters(), lr=args.gnn_lr, weight_decay=args.weight_decay)
    model.train()
    for i in range(200):
        optim.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optim.step()
    model.eval()
    pred = model(data.x * feat_mask, data.edge_index).argmax(dim=-1)
    correct = pred[data.test_mask] == data.y[data.test_mask]
    acc = int(correct.sum()) / int(data.test_mask.sum())
    print(f"previous acc:{acc}")
    top_k = 5
    for i in range(feat_mask.size(0)):
        row = feat_mask[i, :]
        _, indices = torch.topk(row, top_k)
        row[indices] = 0
    pred = model(data.x * feat_mask, data.edge_index).argmax(dim=-1)
    correct = pred[data.test_mask] == data.y[data.test_mask]
    acc = int(correct.sum()) / int(data.test_mask.sum())
    print(f"now acc:{acc}")
