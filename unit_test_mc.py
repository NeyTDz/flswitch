from timeit import default_timer as timer
import numpy as np
from model_paras_count.count import *
from model_paras_count.pre_process import *
from model_paras_count.topk import *
from train_params import *

K = 1
SPARSE = 0.01
N = 20
PARAS_SIZE = 4096
np.random.seed(0)
paras = np.random.uniform(-1, 1, PARAS_SIZE)
stand_power = POWER_CHOICE[1]
stand_products = [10**stand_power,10**-stand_power]
hash_scale = 10**(2+stand_power)
perfect_hash = UniversalHash(hash_scale)
print(paras[0:10])

def unit_test1(num):
    sep_time,hash_time = [],[]
    for i in range(num):
        sep1 = timer()
        anchors =  (paras * stand_products[0]).astype(int)
        residues = paras - anchors * stand_products[1]
        sep2 = timer()
        sep_time.append(sep2-sep1)
        hash1 = timer()
        hash_list = generate_hashlist(anchors,perfect_hash)
        hash2 = timer()
        hash_time.append(hash2-hash1)
        if i % 10 == 0:
            print(i)
    sep_time = np.array(sep_time)
    hash_time = np.array(hash_time)
    return sep_time,hash_time

def unit_test_count(num,k,sp):
    count_time = []
    for i in range(num):
        paras = np.random.uniform(-1, 1, size=(N,PARAS_SIZE))
        anchors =  (paras * stand_products[0]).astype(int)
        residues = paras - anchors * stand_products[1]
        hash_list = generate_hashlist(anchors,perfect_hash)
        hash_list = hash_list.T
        count1 = timer()
        pick_represents(hash_list,k,sp,batch_enc=ENCRYPT_BATCHSIZE)
        count2 = timer()
        count_time.append(count2-count1)
        #if i % 10 == 0:
        #    print(i)
    return count_time

'''
sep_time,hash_time = unit_test1(100)
mean_sep_time = np.mean(sep_time)
mean_hash_time = np.mean(hash_time)
print("Mean sep time:",mean_sep_time)
print("Mean hash time:",mean_hash_time)
'''
for k in [1,2,3,4,5]:
    for sp in [0.01,0.05,0.1]:
        count_time = unit_test_count(100,k,sp)
        mean_count_time = np.mean(count_time)
        print("k={} sp={} Mean count time:{}".format(k,sp,mean_count_time*1000))