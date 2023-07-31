from torch.nn import functional as F
import torch
from utils import construct_pos_neg
from sklearn.metrics import roc_auc_score


def JointTrain(model, args, data, optimizer_gnn, optimizer_pg, pgenerator, type = 'acc'):
    max_test = 0
    train_loss_list = list()
    val_loss_list = list()
    for epoch in range(1, args.joint_epochs + 1):
        model.train()
        optimizer_gnn.zero_grad()
        optimizer_pg.zero_grad()
        out = model(data.x, data.edge_index)
        pgout_x, pgout_adj = pgenerator(data.x, data.edge_index, data.k_hop_edge_index, data.neg_adj)
        loss_gnn = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        loss_pg = pgenerator.mask_loss(data, args, pgout_x, pgout_adj, model)
        loss = (1 - args.alpha) * loss_gnn + args.alpha * loss_pg
        loss.backward()
        optimizer_gnn.step()
        optimizer_pg.step()
        tr, va, te, val_loss, _ = Eval(model, data, type)
        train_loss_list.append(loss.item())
        val_loss_list.append(val_loss.item())
        if te > max_test:
            max_test = te
        print('---------------------------------------------------------------------------------------------')
        print(f'Epoch:{epoch},Loss:{loss.item():.5f},Loss_gnn:{loss_gnn.item():.5f},Loss_pg:{loss_pg.item():.5f}')
        print(f'Train:{tr:.5f},Val:{va:.5f},Test:{te:.5f}')
    print(f'Max_test:{max_test:.5f}, metric:{type}')
    return model, max_test


def Eval(model, data, type='acc'):
    model.eval()
    out = model(data.x, data.edge_index)
    val_loss = F.cross_entropy(out[data.val_mask], data.y[data.val_mask])
    test_loss = F.cross_entropy(out[data.test_mask], data.y[data.test_mask])
    if type == 'acc':
        pred = out.argmax(dim=1)
        tr_correct = pred[data.train_mask] == data.y[data.train_mask]
        tr_acc = int(tr_correct.sum()) / int(data.train_mask.sum())
        va_correct = pred[data.val_mask] == data.y[data.val_mask]
        va_acc = int(va_correct.sum()) / int(data.val_mask.sum())
        te_correct = pred[data.test_mask] == data.y[data.test_mask]
        te_acc = int(te_correct.sum()) / int(data.test_mask.sum())
        return tr_acc, va_acc, te_acc, val_loss, test_loss
    if type == 'auc':
        if out.shape[1] == 2:
            pred = F.softmax(out, dim=1)[:, 1]
        else:
            pred = F.softmax(out, dim=1)
        tr_auc = roc_auc_score(data.y[data.train_mask].cpu().numpy(), pred[data.train_mask].detach().cpu().numpy(), multi_class='ovr')
        if data.val_mask.sum() != 0:
            va_auc = roc_auc_score(data.y[data.val_mask].cpu().numpy(), pred[data.val_mask].detach().cpu().numpy(), multi_class='ovr')
        else:
            va_auc = 0
        te_auc = roc_auc_score(data.y[data.test_mask].cpu().numpy(), pred[data.test_mask].detach().cpu().numpy(), multi_class='ovr')
        return tr_auc, va_auc, te_auc, val_loss, test_loss

def EnhanceTrain(model, args, data, masks, optimizer_gnn, max_test, type='acc'):
    pos, neg, new_feat = construct_pos_neg(data, masks, args)
    new_pgout_adj = torch.sparse_coo_tensor(data.k_hop_edge_index, masks[1][:data.k_hop_edge_index.shape[1]])
    new_adj = new_pgout_adj * data.sparse_adj
    edge_weight = new_adj.coalesce().values().clone().detach().requires_grad_(True)
    print('Prototype as Supervise Training!')
    for epoch in range(1, args.supv_epochs):
        model.train()
        optimizer_gnn.zero_grad()
        if args.use_mf and args.use_ms:
            out = model(new_feat, new_adj.coalesce().indices(), edge_weight)
        elif args.use_ms:
            out = model(data.x, new_adj.coalesce().indices(), edge_weight)
        elif args.use_mf:
            out = model(new_feat, data.edge_index)
        else:
            out = model(data.x, data.edge_index)
        loss_gnn = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        pos_feats = list()
        neg_feats = list()
        for i, j in zip(pos, neg):
            pos_feats.append(out[i])
            neg_feats.append(out[j])
        indices = torch.where(data.train_mask == True)[0]
        triplet_loss = 0
        for i in indices.tolist():
            pos_i = pos_feats[i]
            neg_i = neg_feats[i]
            anchor = out[i].unsqueeze(0).expand(pos_i.shape[0], data.num_classes)
            triplet_loss += F.triplet_margin_loss(anchor, pos_i, neg_i)
        triplet_loss = triplet_loss / len(indices)
        loss = (1-args.beta) * loss_gnn + args.beta * triplet_loss
        loss = loss_gnn
        loss.backward()
        optimizer_gnn.step()
        tr, va, te, val_loss, test_loss = Eval(model, data, type)
        if te > max_test:
            max_test = te
        print('---------------------------------------------------------------------------------------------')
        print(
            f'Epoch:{epoch + args.joint_epochs},Loss:{loss.item():.5f},Loss_gnn:{loss_gnn.item():.5f},Loss_triplet:{triplet_loss.item():.5f}, Val loss:{test_loss.item():.5f}, Test loss:{val_loss.item():.5f}')
        print(f'Train:{tr:.5f},Val:{va:.5f},Test:{te:.5f}')
    print(f'Max_test:{max_test:.5f}, metric:{type}')
    return model, max_test