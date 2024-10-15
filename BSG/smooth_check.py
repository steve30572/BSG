import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, pairwise_distances

import copy
from tqdm import tqdm
import torch
import torch.nn as nn
# from augmae.utils import create_optimizer, accuracy
import torch_geometric.transforms as T
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.utils import add_self_loops, negative_sampling, degree
import numpy as np
import torch.nn.functional as F
from torch_geometric.datasets import Amazon, Coauthor, Planetoid, Reddit, ExplainerDataset
from torch_geometric.utils import add_self_loops, negative_sampling, degree
from torch_geometric.utils.convert import from_dgl
from torch_geometric.nn import MessagePassing, DenseGCNConv, dense_diff_pool, GCNConv

class MeanAggregator(MessagePassing):
    def forward(self, x, edge_index):
        deg = degree(edge_index[0], x.size(0), dtype=x.dtype)
        aggr_out = self.propagate(edge_index, x=x)
        aggr_out = aggr_out / deg.view(-1, 1)
        return aggr_out
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return x_j

def MAD(z_new, data):
    temp = MeanAggregator().to(z_new.device)
    z_neigh = temp(z_new, data.edge_index.to(z_new.device))
    return torch.mean(torch.abs(z_new-z_neigh).mean(1))
    return F.mse_loss(z_new, z_neigh) * 1000
    
    
    z_new = z_new.cpu().detach().numpy()
    z_new = z_new / np.linalg.norm(z_new)
    dist_arr = pairwise_distances(z_new, z_new, metric='cosine')
    # adj_dense = to_dense_adj(edge_index)[0].cpu().numpy()
    # divide_arr = dist_arr * adj_dense
    divide_arr = (dist_arr != 0).sum(1) + 1e-9
    node_dist = dist_arr.sum(1) / divide_arr
    # return node_dist
    mad = np.mean(node_dist)
    return mad

# graphmae = torch.load('emb_graphmae.pt')
graphmae = torch.load('gmae_ori_new.pt')
ours = torch.load('84.75%_cora.pt')
maskgae = torch.load('original_cora.pt')
# maskgae = torch.load('original_cora_bn.pt')
# cca = torch.load('CCA-SSG_cora.pt')
cca = torch.load('Cora_new_CCA.pt')
sup_cora = torch.load('./sup_cora.pt')
ours_gmae = torch.load('./gmae_ours.pt')
sup_link = torch.load('./link_sup.pt')

temp = torch.ones(2708, 512)
from torch_geometric.datasets import Planetoid

dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

print(graphmae.shape, ours.shape, maskgae.shape, cca.shape, sup_cora.shape, ours_gmae.shape)
# print(MAD(graphmae, data)[:10], MAD(ours, data)[:10], MAD(maskgae, data)[:10], MAD(cca, data)[:10])
print(MAD(graphmae, data), MAD(ours, data), MAD(maskgae, data), MAD(cca, data), MAD(sup_cora, data), MAD(sup_link, data), MAD(temp, data))
exit(0)

from torch_geometric.datasets import Planetoid

dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]
# import torch
# import torch.nn.functional as F
# from torch_geometric.nn import GCNConv
# data.x = F.normalize(data.x, p=2, dim=1)

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 128)
        # self.conv2 = GCNConv(16, 16)
        self.conv2 = GCNConv(128, dataset.num_classes)
        # self.conv3 = GCNConv(256, 256)
        self.mlp = torch.nn.Linear(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        # return F.log_softmax(x, dim=1)
        return x # LP
        x = F.relu(x)
        # x = self.conv3(x, edge_index)
        # x = F.relu(x)
        x = self.mlp(x)
        return F.log_softmax(x, dim=1)

        return x
    def emb(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    out = F.normalize(out, p=2, dim=1)
    # loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss = F.nll_loss(out, data.y)
    loss.backward()
    optimizer.step()
model.eval()
out = model.emb(data)
out = F.normalize(out, p=2, dim=1)
# torch.save(out, 'sup_cora.pt')
pred = model(data).argmax(dim=1)
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
acc = int(correct) / int(data.test_mask.sum())
print(f'Accuracy: {acc:.4f}')

# exit(0)

# train_data, val_data, test_data = T.RandomLinkSplit(num_val=0.1, num_test=0.05,
#                                                     is_undirected=True,
#                                                     split_labels=True,
#                                                     add_negative_train_samples=False)(data)
train_data, val_data, test_data = T.RandomLinkSplit(num_val=0.5, num_test=0.1,
                                                    is_undirected=True,
                                                    split_labels=True,
                                                    add_negative_train_samples=False)(data)
model.eval()
feat = model(data)
feat = feat
result_auc = []
result_ap = []
best_val_auc = 0

for epoch in range(1,101):
        model.train()
        feat = model(train_data)
        train_embs = feat#model.embed(graph, feat)
        with torch.no_grad():
            pos_data = val_data.pos_edge_label_index
            train_embs_link = train_embs[pos_data[0]] * train_embs[pos_data[1]]
            pos_pred = torch.sum(train_embs_link, dim=1).sigmoid()#model(None, train_embs_link).sigmoid()
            ###negative
            neg_data = val_data.neg_edge_label_index
            train_embs_link_neg = train_embs[neg_data[0]] * train_embs[neg_data[1]]
            neg_pred = torch.sum(train_embs_link_neg, dim=1).sigmoid()#model(None, train_embs_link_neg).sigmoid()


            #
            pred = torch.cat([pos_pred, neg_pred], dim=0)
            pos_y = pos_pred.new_ones(pos_pred.size(0))
            neg_y = neg_pred.new_zeros(neg_pred.size(0))
            # print(pred.shape, pred[:5])
            # exit(0)

            y = torch.cat([pos_y, neg_y], dim=0)
            y, pred = y.cpu().numpy(), pred.cpu().numpy()

            val_auc = roc_auc_score(y, pred)
            val_ap = average_precision_score(y, pred)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                # correct = test_preds.eq(test_labels).double()
                # th.save(correct, "correct-CCA.pt")
                pos_data = test_data.pos_edge_label_index
                train_embs_link = train_embs[pos_data[0]] * train_embs[pos_data[1]]
                pos_pred = torch.sum(train_embs_link, dim=1).sigmoid()#model(None, train_embs_link).sigmoid()
                ###negative
                neg_data = test_data.neg_edge_label_index
                train_embs_link_neg = train_embs[neg_data[0]] * train_embs[neg_data[1]]
                neg_pred = torch.sum(train_embs_link_neg, dim=1).sigmoid()#model(None, train_embs_link_neg).sigmoid()


                #
                pred = torch.cat([pos_pred, neg_pred], dim=0)
                pos_y = pos_pred.new_ones(pos_pred.size(0))
                neg_y = neg_pred.new_zeros(neg_pred.size(0))

                y = torch.cat([pos_y, neg_y], dim=0)
                y, pred = y.cpu().numpy(), pred.cpu().numpy()

                eval_auc = roc_auc_score(y, pred)
                eval_ap = average_precision_score(y, pred)

                # eval_acc = val_auc
                # eval_ap = val_ap

            # print('Epoch:{}, train_acc:{:.4f}, val_acc:{:4f}, test_acc:{:4f}'.format(epoch, val_auc, val_ap, eval_acc), eval_ap)



        # aug_edge_index, _ = add_self_loops(train_data.edge_index)
        # neg_edges = negative_sampling(
        #     aug_edge_index,
        #     num_nodes=graph.num_nodes,
        #     num_neg_samples=train_data.pos_edge_label_index.view(2, -1).size(1),
        # ).view_as(train_data.pos_edge_label_index)

        #print('train_data',train_data.pos_edge_label_index.shape)  # ([2, 4488])
        #print('neg_edges',neg_edges.shape)  # ([2, 4488])

        copied_pos_edges = train_data.pos_edge_label_index.clone()

        model.train()
        optimizer.zero_grad()
        pos_data = train_data.edge_index
        train_embs_link = train_embs[pos_data[0]] * train_embs[pos_data[1]]
        pos_pred = torch.sum(train_embs_link, dim=1).sigmoid()#model(None, train_embs_link).sigmoid()
        ###negative
        aug_edge_index, _ = add_self_loops(train_data.edge_index)
        neg_edges = negative_sampling(aug_edge_index)
        neg_data = neg_edges
        train_embs_link = train_embs[neg_data[0]] * train_embs[neg_data[1]]
        neg_pred = torch.sum(train_embs_link, dim=1).sigmoid()#model(None, train_embs_link).sigmoid()
        #
        pred = torch.cat([pos_pred, neg_pred], dim=0)
        pos_y = pos_pred.new_ones(pos_pred.size(0))
        neg_y = neg_pred.new_zeros(neg_pred.size(0))

        pos_loss = F.binary_cross_entropy(pos_pred, torch.ones_like(pos_pred))
        neg_loss = F.binary_cross_entropy(neg_pred, torch.zeros_like(neg_pred))
        loss =  pos_loss + neg_loss
        loss.backward()
        optimizer.step()
        y = torch.cat([pos_y, neg_y], dim=0)
        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

        auc = roc_auc_score(y, pred)
        ap = average_precision_score(y, pred)
        # print(auc, ap)
print(f' test_auc: {eval_auc}, test_ap: {eval_ap}')
result_auc.append(eval_auc)
result_ap.append(eval_ap)
feat = model(data)
feat = F.normalize(feat, p=2, dim=1)
torch.save(feat, 'link_sup.pt')

