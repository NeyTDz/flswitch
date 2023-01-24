import numpy as np
import hashlib
import os
from collections import defaultdict,Counter
from model_paras_count.PerfectHash import *

def get_paras(paras_file="dnn_weights.npy") -> np.array:
    '''
    load paras from .npy file
    '''
    paras_path = "./paras"
    path = os.path.join(paras_path,paras_file)
    paras = np.load(path)
    print("Para",paras.shape)
    return paras

def sep_anch_resd(paras,choice=[0,0]):
    '''
    choice = [0,0] common power first, less anchors' value
    choice = [1,n] min power first, more anchors' value
                   n means power = min power *10^(-n), further more anchors' value
    '''
    power_dict = dict()
    for p in paras:
        pe = '{:.5e}'.format(p) #scientific notation format: AeB
        power = pe[-1:] # get B as power
        power_dict[power] = power_dict[power]+1 if power_dict.get(power) else 1
    min_power = int(max(list(power_dict.keys()))) # B max <=> -B min
    common_power = int(max(power_dict.items(),key=lambda x:x[1])[0])
    if choice[0] == 0:
        stand_power = common_power
    elif choice[0] == 1:
        stand_power = min_power + choice[1]
    else:
        stand_power = choice[1]
    stand_products = [10**stand_power,10**-stand_power] 
    anchors =  (paras * stand_products[0]).astype(int)
    residues = paras - anchors * stand_products[1]
    return anchors,residues,stand_power


def count_anchors(anchors):
    anchors_count = Counter(anchors)

def generate_hashlist(anchors:np.array,perfect_hash):
    '''
    hash the anchor part of paras
    '''
    # compute the hash value of head and its value space
    #hash_scale = 10**(stand_power+2)
    #perfect_hash = UniversalHash(hash_scale)
    hash_list = perfect_hash.get_hashed_value(anchors)
    return hash_list