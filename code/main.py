import world
from world import cprint
from os.path import join
import dataloader
import model
import torch
from torch import optim
import numpy as np

dataset = dataloader.Loader()
Recmodel = model.LightGCN(world.config, dataset)
Recmodel = Recmodel.to(world.device)
config = world.config


# gi data
train_groupids = dataset.train_groupids
train_members = torch.IntTensor(dataset.train_members).to(world.device)
train_masks = torch.BoolTensor(dataset.train_masks).to(world.device)
train_items = torch.LongTensor(dataset.train_items).to(world.device)

test_members = dataset.test_members
test_masks = dataset.test_masks
test_items = dataset.test_items

# ssl data
ssl_data = dataset.ssl_data

# gi train and test
optimizer = optim.Adam(Recmodel.parameters(), lr=config['lr'])
for e in range(100):
    # train
    batch_size = config['bpr_batch_size']
    len_train = train_members.shape[0]
    num_batchs = np.int(np.ceil(len_train/batch_size))
    for i in range(num_batchs):
        start_index = i*(batch_size)
        end_index = (i+1)*(batch_size)
        if i == num_batchs - 1:
            end_index = len_train
        temp_groupids = train_groupids[start_index:end_index]
        temp_members = train_members[start_index:end_index]
        temp_masks = train_masks[start_index:end_index]
        temp_items = train_items[start_index:end_index]

        # gi loss
        all_users, all_items = Recmodel.computer()
        centers,offsets = Recmodel.group_rep(temp_members,temp_masks,all_users)
        scores = Recmodel.gi_scores(centers,offsets,temp_items,all_items)
        loss = Recmodel.gi_loss(scores)
        # ssl loss
        if world.mu != 0:
            ssl_loss = Recmodel.ssl_loss(temp_groupids,ssl_data,all_users)
            loss = loss + world.mu*ssl_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if torch.isnan(loss):
            exit()
    print(loss)

    # test
    hits = {}
    ndcgs = {}
    for k in [1,5,10,15,20,50,100]:
        hits[k] = []
        ndcgs[k] = []
    len_test = len(test_members)
    for i in range(len_test):
        start_index = i
        end_index = i+1
        temp_members = torch.IntTensor(test_members[start_index:end_index]).to(world.device)
        temp_masks = torch.BoolTensor(test_masks[start_index:end_index]).to(world.device)
        temp_items = torch.LongTensor(test_items[start_index:end_index]).to(world.device)
        all_users, all_items = Recmodel.computer()
        centers,offsets = Recmodel.group_rep(temp_members,temp_masks,all_users)
        scores = Recmodel.gi_scores(centers,offsets,temp_items,all_items)
        scores = torch.squeeze(scores)
        for k in [1,5,10,15,20,50,100]:
            _,indices = torch.topk(scores,k)
            if 0 in indices:
                hits[k].append(1)
                index = indices.tolist().index(0)
                ndcgs[k].append(np.reciprocal(np.log2(index+2)))
            else:
                hits[k].append(0)
                ndcgs[k].append(0)
    for k in [1,5,10,15,20,50,100]:
        print(np.mean(hits[k]))
        print(np.mean(ndcgs[k]))