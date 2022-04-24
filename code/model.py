"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Define models here
"""
import world
import torch
from dataloader import Dataset
import dataloader
from torch import nn
import numpy as np
import math
import torch.nn.functional as F


class LightGCN(nn.Module):
    def __init__(self, 
                 config:dict, 
                 dataset:Dataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset : dataloader.BasicDataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        if self.config['pretrain'] == 0:
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            world.cprint('use NORMAL distribution initilizer')
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretarined data')
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        print(f"lgn is already to go(dropout:{self.config['dropout']})")


        # group layers
        if world.group_rep == 'geometric':
            self.wc = torch.nn.Linear(self.latent_dim,self.latent_dim,bias=False)
            self.wo = torch.nn.Linear(self.latent_dim,self.latent_dim,bias=False)
        elif world.group_rep == 'attentive':
            self.wc = torch.nn.Linear(self.latent_dim,self.latent_dim,bias=False)
            self.wo = torch.nn.Linear(self.latent_dim,self.latent_dim,bias=True)
            self.act_relu = torch.nn.ReLU(inplace=True)
            self.query = torch.nn.Linear(self.latent_dim,1,bias=False)
            self.key = torch.nn.Linear(self.latent_dim,self.latent_dim,bias=False)
            self.value = torch.nn.Linear(self.latent_dim,self.latent_dim,bias=False)
        self.centers = None
        self.offsets = None

        # parameters for interactions
        self.phi = nn.Sequential(
            nn.Linear(self.latent_dim,self.latent_dim,bias=True),
            nn.LeakyReLU(),
            nn.Linear(self.latent_dim,self.latent_dim,bias=True),
            nn.LeakyReLU(),
            nn.Linear(self.latent_dim,self.latent_dim,bias=True),
            nn.LeakyReLU()
        )
        # psi MLP
        self.psi = nn.Sequential(
            nn.Linear(self.latent_dim,self.latent_dim,bias=True),
            nn.LeakyReLU(),
            nn.Linear(self.latent_dim,self.latent_dim,bias=True),
            nn.LeakyReLU(),
            nn.Linear(self.latent_dim,self.latent_dim,bias=True),
            nn.LeakyReLU()
        )

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph
    
    def computer(self):
        """
        propagate methods for lightGCN
        """       
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        if self.config['dropout']:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph        
        else:
            g_droped = self.Graph    
        
        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items
    
    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating
    
    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))
        
        
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        pos_dises = torch.sqrt(torch.sum((users_emb - pos_emb)**2,dim=-1))
        neg_dises = torch.sqrt(torch.sum((users_emb - neg_emb)**2,dim=-1))
        dis_loss = torch.mean(torch.max(-pos_dises+neg_dises+1,torch.zeros(pos_dises.shape).to(world.device)))

        return dis_loss, reg_loss
       
    def forward(self, users, items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma     = torch.sum(inner_pro, dim=1)
        return gamma

    # group representations
    def group_rep(self,members,masks,all_users):
        centers = torch.empty(0).to(world.device)
        offsets = torch.empty(0).to(world.device)
        if world.group_rep == 'geometric':
            rep = self.geometric_rep
        elif world.group_rep == 'attentive':
            rep = self.attentive_rep
        for member,mask in zip(members,masks):
            embedding_member = torch.index_select(all_users,0,torch.masked_select(member,mask))
            center, offset = rep(embedding_member)
            center = torch.unsqueeze(center,dim=0)
            centers = torch.cat((centers,center),dim=0)
            offset = torch.unsqueeze(offset,dim=0)
            offsets = torch.cat((offsets,offset),dim=0)
        return (centers,offsets)
    def geometric_rep(self,embedding_member): # for single group
        u_max = torch.max(embedding_member,dim=0).values
        u_min = torch.min(embedding_member,dim=0).values
        center = self.wc((u_max+u_min)/2)
        offset = self.wo((u_max-u_min)/2)
        return (center,offset)
    def attentive_rep(self,embedding_member): # for single group
        key_user = self.key(embedding_member)
        key_user_query = F.softmax(self.query(key_user)/math.sqrt(self.latent_dim),dim=-1)
        value_user = self.value(embedding_member)
        attn = torch.squeeze(torch.matmul(value_user.T,key_user_query))
        center = self.wc(attn)
        offset = self.act_relu(self.wo(attn))
        return (center,offset)

    # group-item loss
    def gi_scores(self,centers,offsets,items,all_items):
        lower_left_s = centers - offsets
        upper_right_s = centers + offsets
        lower_left_s = torch.unsqueeze(lower_left_s,dim=1)
        upper_right_s = torch.unsqueeze(upper_right_s,dim=1)
        centers = torch.unsqueeze(centers,dim=1)
        embedding_items = all_items[items]
        dis_out = torch.max(embedding_items-upper_right_s,torch.zeros(embedding_items.shape).to(world.device)) + torch.max(lower_left_s-embedding_items,torch.zeros(embedding_items.shape).to(world.device))
        dis_out = torch.sqrt(torch.sum(dis_out**2,dim=-1))
        dis_in = centers-torch.min(upper_right_s,torch.max(lower_left_s,embedding_items))
        dis_in = torch.sqrt(torch.sum(dis_in**2,dim=-1))
        dises = dis_out + dis_in*0.5
        return dises
    def gi_loss(self,scores):
        pos = scores[:,0:1]
        neg = scores[:,1:]
        neg = torch.unsqueeze(torch.mean(neg,dim=-1),dim=-1)
        loss = torch.mean(torch.max(-pos + neg + 1,torch.zeros(pos.shape).to(world.device)))
        return loss
    
    # ssl loss
    def ssl_loss(self,temp_groupids,ssl_data,all_users):
        num_groupid = len(temp_groupids)
        ssl_loss = 0
        for i in temp_groupids:
            group1s = torch.IntTensor(ssl_data[i][0]).to(world.device)
            group1s_masks = torch.BoolTensor(ssl_data[i][1]).to(world.device)
            group2s = torch.IntTensor(ssl_data[i][2]).to(world.device)
            group2s_masks = torch.BoolTensor(ssl_data[i][3]).to(world.device)
            com_users = torch.LongTensor(ssl_data[i][4]).to(world.device)
            center1s,offset1s = self.group_rep(group1s,group1s_masks,all_users)
            center2s,offset2s = self.group_rep(group2s,group2s_masks,all_users)
            centers,offsets = self.interactions(center1s,offset1s,center2s,offset2s)
            scores = self.gi_scores(centers,offsets,com_users,all_users)
            ssl_loss += self.gi_loss(scores)
        return ssl_loss/num_groupid
    def interactions(self,center1s,offset1s,center2s,offset2s):
        centers = torch.empty(0).to(world.device)
        offsets = torch.empty(0).to(world.device)
        for center1,offset1,center2,offset2 in zip(center1s,offset1s,center2s,offset2s):
            a_den = torch.exp(self.phi(center1)) + torch.exp(self.phi(center2))
            a1 = torch.exp(self.phi(center1))/a_den
            a2 = torch.exp(self.phi(center2))/a_den
            center = a1*center1 + a2*center2
            offset = torch.min(offset1,offset2)*torch.sigmoid(self.psi(offset1+offset2))
            center = torch.unsqueeze(center,dim=0)
            centers = torch.cat((centers,center),dim=0)
            offset = torch.unsqueeze(offset,dim=0)
            offsets = torch.cat((offsets,offset),dim=0)
        return (centers, offsets)
