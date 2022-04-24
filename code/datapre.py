import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import numpy_indexed as npi

data_name = 'yelp_la'
all_data = ['douban_sh','gowalla','meetup','yelp_la']
sorce_path = './data/'
cur_path = sorce_path + data_name + '/'
sum_path = 'sum.pkl'
ssl_path = 'ssl.pkl'

# the function to load data (gi,gu,ui,social)
def load_dat(path):
    res = []
    with open(path) as f:
        for line in f:
            entries = line[:-1].split('\t')
            parent = int(entries[0])
            children = entries[1].split(',')
            for child in children:
                res.append([parent,int(child)])
    res = np.array(res)
    return res


# define the function to get gi_train,gi_test,gu,members,masks,ui_train,ui_test,social
gi_path = 'gi.dat'
gu_path = 'gu.dat'
ui_path = 'ui.dat'
social_path = 'social.dat'
gi = load_dat(gi_path)
gu = load_dat(gu_path)
ui = load_dat(ui_path)
social = load_dat(social_path )
# split the ui and gi to train set and test set
gi_train, gi_test = train_test_split(gi,test_size=0.1,random_state=2021)
ui_train, ui_test = train_test_split(ui,test_size=0.1,random_state=2021)
# members and masks
group_gu = gu[:,0]
gu_classfied = npi.group_by(group_gu).split(gu)
max_length = 0
members = []
masks = []
for group in gu_classfied:
    max_length = max(max_length,len(group))
for group in gu_classfied:
    member = [-1 for i in range(max_length)]
    mask = [False for i in range(max_length)]
    for i in range(len(group)):
        member[i] = group[i][1]
        mask[i] = True
    members.append(member)
    masks.append(mask)
members = np.array(members)
masks = np.array(masks)

# store all variables to a pkl file
with open(sum_path,'wb') as f:
    pickle.dump([gi_train,gi_test,gu,members,masks,ui_train,ui_test,social],f)





# for each group, find other groups which has same users with the group
group_gu = gu[:,0]
gu_classfied = npi.group_by(group_gu).split(gu)
num_group = np.max(group_gu) + 1
group_inters = {}
members_nomask = []
print(num_group)
for i in gu_classfied:
    member = []
    for k in i:
        member.append(k[1])
    members_nomask.append(member)
for i in range(num_group):
    if i%100 == 0:
        print(i)
    group_inters[i] = []
    for k in range(num_group):
        if i != k:
            temp_interactions = [m for m in members_nomask[i] if m in members_nomask[k]]
            if len(temp_interactions) >= 1:
                temp_interactions.insert(0,k)
                group_inters[i].append(temp_interactions)


# handing isolated groups-generate new group that have overlapping users with isolated groups
# proportional swap & proportional imputation
isloated_type = 'ps'
# isloated_type = 'pi'


num_user = np.max(ui[:,0]) + 1
num_group = np.max(gu[:,0]) + 1

# group_inters_ps = group_inters.copy()

members_add = list(members.copy())
members_nomask_add = list(members_nomask.copy())
masks_add = list(masks.copy())

proportion = 0.5
max_length = max_length

print(num_group)
for i in range(num_group):
    if len(group_inters[i]) == 0:
        # generate new group
        cur_member = members_nomask_add[i]
        num_sample = max(1,np.int(np.floor(len(members_nomask_add[i])*proportion)))
        old_member = [x for x in cur_member]
        new_member = []
        k = 0
        while k < num_sample:
            cur_user = np.random.randint(num_user)
            if cur_user not in old_member+new_member:
                new_member.append(cur_user)
                k += 1
        if isloated_type == 'ps':
            np.random.shuffle(old_member)
            new_member = new_member + old_member[0:np.int(np.ceil(len(members_nomask_add[i])*proportion))]
        elif isloated_type == 'pi':
            new_member = new_member + old_member

        member = [-1 for o in range(max_length)]
        mask = [False for o in range(max_length)]

        for m in range(len(new_member)):
            member[m] = new_member[m]
            mask[m] = True

        members_add.append(member)
        masks_add.append(mask)
        members_nomask_add.append([l for l in new_member])
        new_index = len(members_add) - 1
        group_inters[new_index] = []
        num_group += 1


        # new interactions with new group - i and new_index
        for id in [i,new_index]:
            for q in range(num_group):
                if id != q:
                    temp_interactions = [n for n in members_nomask_add[id] if n in members_nomask_add[q]]
                    if len(temp_interactions) >= 1:
                        temp_interactions.insert(0,q)
                        group_inters[id].append(temp_interactions)

# prepare data for self supervised learning directly - each line is a sample - ps
# group1s = [] 
# group1s_masks = []
# group2s = []
# group2s_masks = []
# com_users = []
# num_user = np.max(ui[:,0]) + 1
# for i in range(len(group_inters.keys())):
#     cur_group_id = i
#     temp_inters = group_inters[i]
#     rows = np.arange(len(temp_inters))
#     selected_rows = np.random.choice(rows,10)
#     for j in selected_rows:
#         inter_group_id = temp_inters[j][0]
#         com_user = temp_inters[j][1:]
#         for u in com_user:
#             group1s.append(members_add[cur_group_id])
#             group1s_masks.append(masks_add[cur_group_id])
#             group2s.append(members_add[inter_group_id])
#             group2s_masks.append(masks_add[inter_group_id])
#             # generate negative users for each sample
#             users = [u]
#             index = 0
#             while index < 5:
#                 neg = np.random.randint(num_user)
#                 if (neg not in users) and (neg not in members_add[cur_group_id]) and (neg not in members_add[inter_group_id]):
#                     index += 1
#                     users.append(neg)
#             com_users.append(users)
# group1s = np.array(group1s)
# group1s_masks = np.array(group1s_masks)
# group2s = np.array(group2s)
# group2s_masks = np.array(group2s_masks)
# com_users = np.array(com_users)

# prepare ssl data for each group
num_user = np.max(ui[:,0]) + 1
ssl_data = {} # a dict to store ssl data where the key is the group id

for i in range(len(group_inters.keys())):
    group1s = [] 
    group1s_masks = []
    group2s = []
    group2s_masks = []
    com_users = []
    cur_group_id = i
    temp_inters = group_inters[i]
    rows = np.arange(len(temp_inters))
    selected_rows = np.random.choice(rows,10)
    for j in selected_rows:
        inter_group_id = temp_inters[j][0]
        com_user = temp_inters[j][1:]
        for u in com_user:
            group1s.append(members_add[cur_group_id])
            group1s_masks.append(masks_add[cur_group_id])
            group2s.append(members_add[inter_group_id])
            group2s_masks.append(masks_add[inter_group_id])
            # generate negative users for each sample
            users = [u]
            index = 0
            while index < 5:
                neg = np.random.randint(num_user)
                if (neg not in users) and (neg not in members_add[cur_group_id]) and (neg not in members_add[inter_group_id]):
                    index += 1
                    users.append(neg)
            com_users.append(users)
    cur_ssl = [group1s,group1s_masks,group2s,group2s_masks,com_users]
    ssl_data[cur_group_id] = cur_ssl
    

# store SSL data to pkl
with open(ssl_path,'wb') as f:
    pickle.dump([ssl_data],f)
