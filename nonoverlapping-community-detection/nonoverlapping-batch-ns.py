from __future__ import division
from __future__ import print_function

import argparse
import time
from tqdm import tqdm
import math
import numpy as np
from numpy.random import multinomial
from subprocess import check_output
import itertools
from collections import Counter
import random

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch import optim
import torch.autograd as autograd

import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

import collections
import re

from data_utils import load_cora_citeseer, load_webkb
from score_utils import calc_nonoverlap_nmi
import community
import torch
import numpy as np
from sklearn.cluster import KMeans



parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='s', help="models used")
parser.add_argument('--lamda', type=float, default=.1, help="")
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--n-neg', type=int, default=10, help='Negative sampling samples.')
parser.add_argument('--epochs', type=int, default=1001, help='Number of epochs to train.')
parser.add_argument('--embedding-dim', type=int, default=128, help='')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset-str', type=str, default='cora', help='type of dataset.')
# parser.add_argument('--task', type=str, default='community', help='type of dataset.')


# negative_samples = sample_negative(train_edges, 4)


def logging(args, epochs, nmi, modularity):
    with open('mynlog', 'a+') as f:
        f.write('{},{},{},{},{},{},{},{}\n'.format('gcn_vae', args.dataset_str, args.lr, args.embedding_dim, args.lamda, epochs, nmi, modularity))

def write_to_file(fpath, clist):
    with open(fpath, 'w') as f:
        for c in clist:
            f.write(' '.join(map(str, c)) + '\n')

def preprocess(fpath): 
    clist = []
    with open(fpath, 'rb') as f:
        for line in f:
            tmp = re.split(b' |\t', line.strip())[1:]
            clist.append([x.decode('utf-8') for x in tmp])
    
    write_to_file(fpath, clist)
            

def get_assignment(G, model, num_classes=5, tpe=0):
    model.eval()
    edges = [(u,v) for u,v in G.edges()]

    negative_samples = sample_negative(edges, sample_size=args.n_neg)

    context_tuple_list = []
    for e in edges:
        u, v = e
        context_tuple_list.append((u, v, next(negative_samples)))
    
    n_nodes = G.number_of_nodes()
    res = np.zeros((n_nodes, num_classes))

    context_tuple_batches = get_batches(context_tuple_list, batch_size=100)
    
    for j in range(len(context_tuple_batches)):
        target_tensor, context_tensor, negative_tensor = context_tuple_batches[j]
        bs = len(target_tensor)
        out, q, prior = model(target_tensor, context_tensor, negative_tensor, temp)
        num_classes = q.shape[1]
        q_argmax = q.argmax(dim=-1)

        for idx, e in enumerate(context_tuple_batches):
            if tpe == 0:
                res[e[0], :] += q[idx, :].cpu().data.numpy()
                res[e[1], :] += q[idx, :].cpu().data.numpy()
            else:
                res[e[0], q_argmax[idx]] += 1
                res[e[1], q_argmax[idx]] += 1


    # batch = torch.LongTensor(edges)
    # _, q, _ = model(batch[:, 0], batch[:, 1], 1.)

    # q_argmax = q.argmax(dim=-1)

    assignment = {}

    # for idx, e in enumerate(edges):
    #     if tpe == 0:
    #         res[e[0], :] += q[idx, :].cpu().data.numpy()
    #         res[e[1], :] += q[idx, :].cpu().data.numpy()
    #     else:
    #         res[e[0], q_argmax[idx]] += 1
    #         res[e[1], q_argmax[idx]] += 1

    res = res.argmax(axis=-1)
    assignment = {i : res[i] for i in range(res.shape[0])}
    # print('assignment', assignment)
    return res, assignment

def classical_modularity_calculator(graph, embedding, model='gcn_vae', cluster_number=5):
    """
    Function to calculate the DeepWalk cluster centers and assignments.
    """    
    if model == 'gcn_vae':
        assignments = embedding
    else:
        kmeans = KMeans(n_clusters=cluster_number, random_state=0, n_init = 1).fit(embedding)
        assignments = {i: int(kmeans.labels_[i]) for i in range(0, embedding.shape[0])}

    modularity = community.modularity(assignments, graph)
    return modularity


def loss_function(out, q_y, prior, norm=None, pos_weight=None):
    
    # print('recon_c', recon_c.shape, recon_c)
    # print('c', c.shape, c)
    # BCE = F.cross_entropy(recon_c, c, reduction='sum') / c.shape[0]
    # BCE = F.binary_cross_entropy_with_logits(recon_c, c, pos_weight=pos_weight)
    # return BCE
    BCE = out

    log_qy = torch.log(q_y  + 1e-20)
    KLD = torch.sum(q_y*(log_qy - torch.log(prior)),dim=-1).mean()

    ent = (- torch.log(q_y) * q_y).sum(dim=-1).mean()
    return BCE + KLD


def get_batches(context_tuple_list, batch_size=100):
    random.shuffle(context_tuple_list)
    batches = []
    batch_target, batch_context, batch_negative = [], [], []
    for i in range(len(context_tuple_list)):
        batch_target.append(context_tuple_list[i][0])
        batch_context.append(context_tuple_list[i][1])
        batch_negative.append([w for w in context_tuple_list[i][2]])
        # batch_negative.append(next(negative_samples))

        if (i+1) % batch_size == 0 or i == len(context_tuple_list)-1:
            tensor_target = autograd.Variable(torch.from_numpy(np.array(batch_target)).long())
            tensor_context = autograd.Variable(torch.from_numpy(np.array(batch_context)).long())
            tensor_negative = autograd.Variable(torch.from_numpy(np.array(batch_negative)).long())
            batches.append((tensor_target, tensor_context, tensor_negative))
            batch_target, batch_context, batch_negative = [], [], []
    return batches


def sample_negative(train_edges, sample_size):
    sample_probability = {}
    word_counts = dict(Counter(list(itertools.chain.from_iterable(train_edges))))
    normalizing_factor = sum([v**0.75 for v in word_counts.values()])
    for word in word_counts:
        sample_probability[word] = word_counts[word]**0.75 / normalizing_factor
    words = np.array(list(word_counts.keys()))
    while True:
        word_list = []
        sampled_index = np.array(multinomial(sample_size, list(sample_probability.values())))
        for index, count in enumerate(sampled_index):
            for _ in range(count):
                 word_list.append(words[index])
        yield word_list


class GCNModelGumbel_2(nn.Module):
    def __init__(self, size, embedding_dim, categorical_dim, dropout, device):
        super(GCNModelGumbel_2, self).__init__()
        self.embedding_dim = embedding_dim
        self.categorical_dim = categorical_dim
        self.device = device
        self.size = size

        self.community_embeddings = nn.Linear(embedding_dim, categorical_dim, bias=False).to(device)
        self.node_embeddings = nn.Embedding(size, embedding_dim)
        self.contextnode_embeddings = nn.Embedding(size, embedding_dim)

        self.decoder = nn.Sequential(
          nn.Linear( embedding_dim, size),
        ).to(device)

        self.init_emb()

    def init_emb(self):
        initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward_old1(self, w, c, neg, temp):

        bs = len(w)
        n_negs = neg.size()[1]

        w = self.node_embeddings(w).to(self.device)
        c_ = self.node_embeddings(c).to(self.device)
        c_context = self.contextnode_embeddings(c.unsqueeze(1)).to(self.device)

        emb_negative = self.contextnode_embeddings(neg).to(self.device).neg()

        q = w * c_
        q_unsq = q.unsqueeze(2)

        prior = w
        prior = F.softmax(prior, dim=-1)

        oloss = torch.bmm(c_context, q_unsq).squeeze().sigmoid().log().mean()
        nloss = torch.bmm(emb_negative, q_unsq).squeeze().sigmoid().log().view(-1, 1, n_negs).sum(2).mean(1)

        return -(oloss + nloss).mean(), F.softmax(q, dim=-1), prior

    def forward(self, w, c, neg, temp):

        w = self.node_embeddings(w).to(self.device)
        c_ = self.node_embeddings(c).to(self.device)

        c_context = self.contextnode_embeddings(c).to(self.device)
        c_context_community = self.community_embeddings(c_context)
        neg_context = self.contextnode_embeddings(neg).to(self.device)
        neg_context_community = self.community_embeddings(neg_context)  # neg  size: bs x n_neg x n_com

        q = self.community_embeddings(w * c_)

        z = F.gumbel_softmax(logits=q, tau=temp, hard=True)

        prior = self.community_embeddings(w)
        prior = F.softmax(prior, dim=-1)

        mulpositivebatch = torch.mul(z, c_context_community)
        positivebatch = F.logsigmoid(torch.sum(mulpositivebatch, dim=1))

        mulnegativebatch = torch.mul(z.view(len(q), 1, self.categorical_dim), neg_context_community)
        # negativebatch = torch.sum(
        negativebatch = torch.mean(
            F.logsigmoid(
                - torch.sum(mulnegativebatch, dim=2)
            ),
            dim=1)

        loss = positivebatch + negativebatch

        return -torch.mean(loss), F.softmax(q, dim=-1), prior


class GCNModelGumbel(nn.Module):
    def __init__(self, size, embedding_dim, categorical_dim, dropout, device):
        super(GCNModelGumbel, self).__init__()
        self.embedding_dim = embedding_dim
        self.categorical_dim = categorical_dim
        self.device = device
        self.size = size

        self.community_embeddings = nn.Linear(embedding_dim, categorical_dim, bias=False).to(device)
        self.node_embeddings = nn.Embedding(size, embedding_dim)
        self.contextnode_embeddings = nn.Embedding(size, embedding_dim)

        self.decoder = nn.Sequential(
          nn.Linear( embedding_dim, size),
        ).to(device)

        self.init_emb()

    def init_emb(self):
        initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, w, c, neg, temp):

        bs = len(w)

        w = self.node_embeddings(w).to(self.device)
        c_ = self.node_embeddings(c).to(self.device)
        c_context = self.contextnode_embeddings(c).to(self.device)
        c_context_comunity = self.community_embeddings(c_context)
        # c_context_comunity = F.gumbel_softmax(logits=c_context_comunity, tau=temp, hard=True)

        emb_negative = self.contextnode_embeddings(neg).to(self.device)
        emb_negative_comunity = self.community_embeddings(emb_negative)
        # emb_negative_comunity = F.gumbel_softmax(logits=emb_negative_comunity, tau=temp, hard=True)

        q = self.community_embeddings(w*c_)
        # q.shape: [batch_size, categorical_dim]
        # z = self._sample_discrete(q, temp)
        if self.training:
            z = F.gumbel_softmax(logits=q, tau=temp, hard=True)
        else:
            tmp = q.argmax(dim=-1).reshape(q.shape[0], 1)
            z = torch.zeros(q.shape).to(self.device).scatter_(1, tmp, 1.)

        prior = self.community_embeddings(w)
        prior = F.softmax(prior, dim=-1)
        # prior.shape [batch_num_nodes, 

        # z.shape [batch_size, categorical_dim]
        # new_z = torch.mm(z, self.community_embeddings.weight)
        # new_z shape [batch_size, embedding_dim]
        # recon = self.decoder(new_z)  # [bs, size]

        pos_prod = torch.mul(c_context_comunity, q)  # not sure use z or q
        # print('pos_prod before', pos_prod.shape)
        pos_prod = torch.sum(pos_prod, dim=1)
        # print('pos_prod after', pos_prod.shape)

        out = F.logsigmoid(pos_prod)
        # print('emb negative comunity', emb_negative_comunity.shape)
        neg_prod = torch.bmm(emb_negative_comunity, q.unsqueeze(2))
        # neg_prod = torch.mul(emb_negative_comunity, 
            # torch.repeat_interleave(q.unsqueeze(1), emb_negative_comunity.shape[1], dim=1))
        # print('neg_prod before', neg_prod.shape)

        # neg_prod = torch.sum(neg_prod, dim=1)
        # print('neg_prod after', neg_prod.shape)
        # neg_prod = neg_prod.flatten()
        # print(F.logsigmoid(-neg_prod).sum(1).squeeze())

        out += F.logsigmoid(-neg_prod).sum(1).squeeze()
        out = -out.sum() / bs
        # print('out', out)

        return out, F.softmax(q, dim=-1), prior


if __name__ == '__main__':
    args = parser.parse_args()
    embedding_dim = args.embedding_dim
    lr = args.lr
    epochs = args.epochs
    temp = 1.
    temp_min = 0.3
    ANNEAL_RATE = 0.00003

    # In[13]:
    if args.dataset_str in ['cora', 'citeseer']:
        G, adj, gt_membership = load_cora_citeseer(args.dataset_str)
    else:
        G, adj, gt_membership = load_webkb(args.dataset_str)

    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()
    categorical_dim = len(set(gt_membership))
    n_nodes = G.number_of_nodes()
    print(n_nodes, categorical_dim)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = GCNModelGumbel_2(adj.shape[0], embedding_dim, categorical_dim, args.dropout, device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    hidden_emb = None
    history_valap = []
    history_mod = []

    train_edges = [(u,v) for u,v in G.edges()]
    n_nodes = G.number_of_nodes()
    print('len(train_edges)', len(train_edges))

    model.train()
    losses = []

    negative_samples = sample_negative(train_edges, sample_size=args.n_neg)

    context_tuple_list = []
    for e in train_edges:
        u, v = e
        context_tuple_list.append((u, v, next(negative_samples)))

    for epoch in range(epochs):

        t = time.time()

        context_tuple_batches = get_batches(context_tuple_list, batch_size=100)

        for j in range(len(context_tuple_batches)):
            optimizer.zero_grad()
            target_tensor, context_tensor, negative_tensor = context_tuple_batches[j]
            # print('target', target_tensor)
            # print('context', context_tensor)
            # print('negative', negative_tensor)
            bs = len(target_tensor)
            out, q, prior = model(target_tensor, context_tensor, negative_tensor, temp)
            # todo 

            res = torch.zeros([n_nodes, categorical_dim], dtype=torch.float32).to(device)
            for idx, e in enumerate(context_tuple_batches[j]):
                res[e[0], :] += q[idx, :]
                res[e[1], :] += q[idx, :]
            smoothing_loss = args.lamda * ((res[target_tensor] - res[context_tensor])**2).mean()
            loss = loss_function(out, q, prior, None, None)
            loss += smoothing_loss

            loss.backward()
            optimizer.step()
            losses.append(loss.data)
        # print("Loss: ", np.mean(losses))
        
        if epoch % 10 == 0:
            temp = np.maximum(temp*np.exp(-ANNEAL_RATE*epoch),temp_min)
            
            model.eval()
            
            membership, assignment = get_assignment(G, model, categorical_dim)
            #print([(membership == i).sum() for i in range(categorical_dim)])
            #print([(np.array(gt_membership) == i).sum() for i in range(categorical_dim)])
            modularity = classical_modularity_calculator(G, assignment)
            nmi = calc_nonoverlap_nmi(membership.tolist(), gt_membership)
            
            print('Loss',  np.mean(losses), epoch, nmi, modularity)
            logging(args, epoch, nmi, modularity)
            
    print("Optimization Finished!")
