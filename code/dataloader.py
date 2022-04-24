"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Shuxian Bi (stanbi@mail.ustc.edu.cn),Jianbai Ye (gusye@mail.ustc.edu.cn)
Design Dataset here
Every dataset's index has to start at 0
"""
import os
from os.path import join
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import world
from world import cprint
from time import time
import pickle
import numpy_indexed as npi

class Loader(Dataset):

    def __init__(self,config = world.config,path="../data"):
        # train or test
        cprint(f'loading [{path}]')
        self.split = config['A_split']
        self.folds = config['A_n_fold']
        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train']
        self.n_user = 0
        self.m_item = 0
        self.path = path
        trainUniqueUsers, trainItem, trainUser = [], [], []
        testUniqueUsers, testItem, testUser = [], [], []
        self.traindataSize = 0
        self.testDataSize = 0

        # import data
        with open('../data/yelp_sum.pkl','rb') as f:
            gi_train,gi_test,gu,members,masks,ui_train,ui_test,social = pickle.load(f)
        self.n_group = max(gu[:,0]) + 1
        self.n_user = max(gu[:,1]) + 1
        self.m_item = max([max(gi_train[:,1]),max(gi_test[:,1]),max(ui_train[:,1]),max(ui_test[:,1])]) + 1
        print(self.n_user)
        print(self.m_item)


        # group visited items - gvi
        gvi = {}
        gi = np.vstack((gi_train,gi_test))
        gi_g = gi[:,0]
        gi_classfied = npi.group_by(gi_g).split(gi)
        for i in gi_classfied:
            group_id = i[0][0]
            gvi[group_id] = []
            for k in i:
                gvi[group_id].append(k[1])
                
        # group_train
        train_groupids = []
        train_members = []
        train_masks = []
        for i in gi_train[:,0]:
            train_groupids.append(i)
            train_members.append(members[i])
            train_masks.append(masks[i])
        self.train_groupids = train_groupids
        self.train_members = train_members
        self.train_masks = train_masks       

        # group_test - why not test-groupid?
        test_members = []
        test_masks = []
        for i in gi_test[:,0]:
            test_members.append(members[i])
            test_masks.append(masks[i])
        self.test_members = test_members
        self.test_masks = test_masks

        # generate negative items
        # train - the number of negative items is 5
        train_items = []
        for i in gi_train:
            pn = []
            pn.append(i[1])
            index = 0
            visited_items = gvi[i[0]]
            while index < 5:
                neg = np.random.randint(self.m_item)
                if neg not in visited_items:
                    pn.append(neg)
                    visited_items.append(neg)
                    index += 1
            train_items.append(pn)
            gvi[i[0]] = visited_items
        train_items = np.array(train_items)
        self.train_items = train_items
        #test
        test_items = []
        for i in gi_test:
            visited_items = gvi[i[0]]
            pn = [i for i in range(self.m_item) if i not in visited_items]
            pn.insert(0,i[1])
            test_items.append(pn)
        self.test_items = test_items


        # generate user data
        self.trainUniqueUser = np.unique(ui_train[:,0])
        self.trainUser = ui_train[:,0]
        self.trainItem = ui_train[:,1]

        self.testUniqueUser = np.unique(ui_test[:,0])
        self.testUser = ui_test[:,0]
        self.testItem = ui_test[:,1]

        self.traindataSize = len(ui_train)
        self.testDataSize = len(ui_test)

        # social
        self.social_u1 = social[:,0]
        self.social_u2 = social[:,1]
        self.socialNet = csr_matrix((np.ones(len(self.social_u1)), (self.social_u1, self.social_u2)),shape=(self.n_user, self.n_user))
        # group members
        self.members = torch.IntTensor(members)
        self.masks = torch.BoolTensor(masks)


        # generate SSL data

        # import SSL Data
        with open('../data/ssl.pkl','rb') as f:
            self.ssl_data = pickle.load(f)[0]

        self.Graph = None
        print(f"{self.trainDataSize} interactions for training")
        print(f"{self.testDataSize} interactions for testing")
        print(f"{world.dataset} Sparsity : {(self.trainDataSize + self.testDataSize) / self.n_users / self.m_items}")

        # (users,items), bipartite graph
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                      shape=(self.n_user, self.m_item))
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.
        # pre-calculate
        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        self.__testDict = self.__build_test()


        print(f"{world.dataset} is ready to go")

    @property
    def n_users(self):
        return self.n_user
    
    @property
    def m_items(self):
        return self.m_item
    
    @property
    def trainDataSize(self):
        return self.traindataSize
    
    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    def _split_A_hat(self,A):
        A_fold = []
        fold_len = (self.n_users + self.m_items) // self.folds
        for i_fold in range(self.folds):
            start = i_fold*fold_len
            if i_fold == self.folds - 1:
                end = self.n_users + self.m_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(world.device))
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
        
    def getSparseGraph(self):
        print("loading adjacency matrix")
        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat.npz')
                print("successfully loaded...")
                norm_adj = pre_adj_mat
            except :
                print("generating adjacency matrix")
                s = time()
                adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                
                # ui
                R = self.UserItemNet.tolil()
                adj_mat[:self.n_users, self.n_users:] = R
                adj_mat[self.n_users:, :self.n_users] = R.T

                # social
                # social = self.socialNet.tolil()
                # adj_mat[:self.n_users, :self.n_users] = social             


                adj_mat = adj_mat.todok()
                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)
                
                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time()
                print(f"costing {end-s}s, saved norm_mat...")
                sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)

            if self.split == True:
                self.Graph = self._split_A_hat(norm_adj)
                print("done split matrix")
            else:
                self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
                self.Graph = self.Graph.coalesce().to(world.device)
                print("don't split the matrix")
        return self.Graph

    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems