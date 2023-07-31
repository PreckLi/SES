# -*- coding:utf-8 -*-
import torch
import numpy as np
from GCN import GCN
from GAT import GAT
from MaskGen import MaskGenerator
from utils import load_data
from train_eval import JointTrain, EnhanceTrain, Eval
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Cora', help='dataset')
parser.add_argument('--device', type=str, default='cuda:1', help='device')
parser.add_argument('--hid_size', type=int, default=128, help="hidden size")
parser.add_argument('--khop', type=int, default=2, help="k-hop subgraph")
parser.add_argument('--alpha', type=float, default=0.7, help="hyper parameter for protograph generator")
parser.add_argument('--beta', type=float, default=0.7, help="hyper parameter for supervise training")
parser.add_argument('--metric', type=str, default='acc', help="auc/acc")
parser.add_argument('--gnn', type=str, default='gcn', help="gcn/gat")
parser.add_argument('--gnn_lr', type=float, default=0.003, help="learning rate")
parser.add_argument('--p_lr', type=float, default=0.001, help="learning rate")
parser.add_argument('--train_ratio', type=float, default=0.6, help='train set ratio')
parser.add_argument('--val_ratio', type=float, default=0.2, help='validation set ratio')
parser.add_argument('--pos_ratio', type=float, default=0.8, help='positive ratio')
parser.add_argument('--split', type=str, default='random', help="split")
parser.add_argument('--gat_heads', type=int, default=8, help="gat heads")
parser.add_argument('--dropout', type=float, default=0.5, help="dropout")
parser.add_argument('--weight_decay', type=float, default=5e-4, help="weight decay")
parser.add_argument('--joint_epochs', type=int, default=300, help="joint training epochs")
parser.add_argument('--supv_epochs', type=int, default=100, help="supervision training epochs")
parser.add_argument('--use_mf', type=bool, default=True, help="use Mf impact on feature?")
parser.add_argument('--use_ms', type=bool, default=True, help="use Ms impact on structure?")
parser.add_argument('--use_adj_loss', type=bool, default=True, help="use adj loss?")
args = parser.parse_args()

acc = list()
for i in range(1):
    since = time.time()
    print("range{}----------------------------------------------------------".format(i + 1))
    device = torch.device(args.device)
    data = load_data(args)
    data = data.to(device)
    if args.gnn == 'gcn':
        model = GCN(data, args).to(device)
    if args.gnn == 'gat':
        model = GAT(data, args).to(device)
    if args.dataset == 'bashape' or args.dataset == 'bacom' or args.dataset == 'treegrid' or args.dataset == 'treecycle':
        args.joint_epochs = 2000
        from torch_geometric.nn import GCN

        model = GCN(data.num_node_features, hidden_channels=20, num_layers=3,
                    out_channels=data.num_classes).to(device)
    mgenerator = MaskGenerator(data, args.hid_size, args.dropout).to(device)

    optimizer_gnn = torch.optim.Adam(model.parameters(), lr=args.gnn_lr, weight_decay=args.weight_decay)
    optimizer_pg = torch.optim.Adam(mgenerator.parameters(), lr=args.p_lr, weight_decay=args.weight_decay)
    for i in range(1):
        model, max_test = JointTrain(model, args, data, optimizer_gnn, optimizer_pg, mgenerator, args.metric)
        masks = mgenerator(data.x, data.edge_index, data.k_hop_edge_index, data.neg_adj)
        time_elapsed = time.time() - since
        print('\nJoint training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        model, max_test = EnhanceTrain(model, args, data, masks, optimizer_gnn, max_test, args.metric)
    _, _, max_test, _, _ = Eval(model, data)
    acc.append(max_test)

acc_mean = np.mean(acc)
acc_std = np.std(acc)

print('---------args-----------')
for k in list(vars(args).keys()):
    print('%s: %s' % (k, vars(args)[k]))
print('--------end!----------\n')
