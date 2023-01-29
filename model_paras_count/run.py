import numpy as np
import os
from pre_process import *
from count import count_process
from analysis import *
import sys


if __name__ == "__main__":

    K = 10
    sparse_bound = 0.01
    batch_enc = 36
    paras = get_paras("dnn_weights.npy")
    print(paras.shape)
    hash_list = generate_hashlist(paras)
    all_costs = []
    batches = np.arange(10,20,10)
    for b in batches:
        costs = np.array([])
        for k in range(1,K):
            alloc,B_inter,B_sparse,inter_clients,sparse,cost = count_process(hash_list,k,b)
            costs = np.append(costs,cost)
            if sparse < sparse_bound:
                #print("Optimized!")
                break
            paras_count_result = {"inter_paras":B_inter,"sparse_paras:":B_sparse,"inter_clients":inter_clients}
            np.save("./result/paras_count_result.npy",paras_count_result)
        all_costs.append(costs)
    all_costs = np.array(all_costs)
    print(all_costs)



