import random
import numpy as np
import collections
from collections import defaultdict,Counter
from model_paras_count.topk import top_frequent
from math import ceil,floor
import hashlib



def get_stat(nums:list) -> defaultdict(): 
    '''
    Return statistics of elements in a list
    e.g.
    List: ['A','A','B','C','C','C']
    stat: {'A': [0, 1], 'B': [2], 'C': [3, 4, 5]}
    '''
    stat = defaultdict(list)
    for i,num in enumerate(nums):
        stat[num].append(i)
    #stat_keys = list(stat_dict.keys())
    #stat = [(key,len(stat_dict[key])) for key in stat_keys]
    return stat

def top_process(stat:defaultdict(),k:int):
    '''
    Return k elements with highest frequency
    e.g.
    stat: {'A': [0, 1], 'B': [2], 'C': [3, 4, 5]}
    k: 2
    frequents:  [('C',3),('A',2)]
    elements: [('C',3,[3,4,5]),('A',2,[0,1])]
    Notice: 
    if k < |stat_keys|, set k_in = min(k,stat_keys)
    topk <=> sort at that time
    '''
    stat_keys = list(stat.keys())
    stat_count = [(key,len(stat[key])) for key in stat_keys]
    k_in = min(k,len(stat_keys))
    frequents = top_frequent(stat_count,k_in,"heap")
    elements = [(freq[0],freq[1],stat[freq[0]]) for freq in frequents]
    return frequents,elements

def top_tiny(nums,k):
    '''
    Only stat frequents, no need for elements
    For anchor stat in each local client
    '''
    stat_freq = Counter(nums)
    stat_count = list(stat_freq.items())
    frequents = top_frequent(stat_count,k,"heap")
    return frequents

def one_hot(x:list,n:int) -> np.array:
    '''
    Return one-hot vector of a set
    e.g.
    x: [2,5,9]
    n: 10
    one_hot: [0,0,1,0,0,1,0,0,0,1]
    '''
    one_hot = np.zeros(n)
    for i in x:
        one_hot[i] = 1
    return one_hot

def count_process(stats:list,n:int,k:int,batch_enc=10):
    '''
    README.md
    '''
    m,n = len(stats),n
    Sp = []
    Lp = []
    for stat in stats:
        frequents,elements = top_process(stat,k)
        select = []
        l = 0
        for i in range(len(elements)):
            select += elements[i][2]
            l += elements[i][1]
        select = one_hot(select, n)
        Sp.append(select)
        Lp.append(l)
    Sp = np.mat(Sp)
    # sort as most Lp
    Lpindex = sorted(range(len(Lp)), key=lambda k: Lp[k], reverse=True)
    Sp = Sp[Lpindex]
    # computer intersection
    S_inter = np.ones((1,n))
    B_inter = []
    B_sparse = []
    for i,s in enumerate(Sp):
        s_mat = np.diag(s.tolist()[0])
        flag = np.matmul(S_inter,s.T)
        if flag > 0:
            S_inter = np.matmul(S_inter,s_mat)
            B_inter.append(i)
        else:
            B_sparse.append(i)
    inter_clients = np.argwhere(S_inter[0] == np.max(S_inter[0])).reshape(-1).tolist()

    '''batch cut'''
    residue = len(B_inter) % batch_enc
    if residue != 0:
        B_sparse += B_inter[-residue:]
        B_inter = B_inter[:-residue] 

    '''evaluate_cost'''
    '''
    inter_batch_enc = batch_enc * 2
    inter_cost = ceil(len(B_inter)/inter_batch_enc)
    # rough in sparse
    sparse_cost = floor(len(B_sparse)/((n-1)*batch_enc)) * (n-1)
    margin = len(B_sparse)%((n-1)*batch_enc)
    #print("Mar",sparse_cost,margin)
    if margin <= (n-1):
        sparse_cost += margin
    else:
        sparse_cost += (n-1)
    # exact in sparse
    ###
    #
    cost = inter_cost + sparse_cost
    print("Costs:",cost,"Inter costs:",inter_cost,"Sparse costs:",sparse_cost)
    '''
    return B_inter,B_sparse,inter_clients


def pick_represents(hash_list:list,k:int,sparse:float= 0.05,batch_enc=10):

    (m,n) = (len(hash_list),len(hash_list[0])) # m:paras n:clients
    stats = [get_stat(h) for h in hash_list]
    alloc = dict()
    alloc_num = 0
    while 1 - alloc_num / m >= sparse and (m - alloc_num)>batch_enc:
        B_inter,B_sparse,inter_clients = count_process(stats,n,k,batch_enc)
        main_client = random.choice(inter_clients)
        alloc[main_client] = B_inter
        alloc_num += len(B_inter)
        stats = [stats[i] for i in B_sparse]
        for i in range(len(stats)):
            for key in stats[i].keys():
                stats[i][key] = [s for s in stats[i][key] if s != main_client]
    empty_keys = [k for k in alloc.keys() if len(alloc[k]) == 0]
    for key in empty_keys:
        del alloc[key]
        
    # equally distribute sparse paras into selected clients
    '''
    n_sc = len(alloc.keys())
    if n_sc > len(B_sparse):
        alloc_B_sparse = [[p] for p in B_sparse] + [[] for p in range(n_sc-len(B_sparse))]
    else:
        alloc_B_sparse = [list(range(i,len(B_sparse),n_sc)) for i in range(n)]
    for i,key in enumerate(list(alloc.keys())):
        alloc[key] += alloc_B_sparse[i]
    '''
    
    # output to monitor distribution of selected clients
    #print('selected:',list(alloc.keys()),'indexes:',[len(v) for v in alloc.values()])
    #selected = one_hot(list(alloc.keys()),n)
    selected = list(alloc.keys())
    return alloc,selected,B_sparse