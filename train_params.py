import logging
import os
from pickle import TRUE
import torch
import numpy as np

""" MSG CONSTANT """
READY_MSG = "ready"
COMP_MSG = "complete"
MASK = True # Cipher or Plaintext, equal to assign BACKEND to 'plain' 
PRED_CON = True
""" NETWORK PARAM """
SERVER_IP = "127.0.0.1"
SERVER_PORT = 12337
LEADER_IP = "127.0.0.1"
LEADER_PORT = 12338
CLIENT_IPs = ["127.0.0.1", "127.0.0.1"]
CLIENT_PORTs = [10002, 10003]
BUFF_SIZE = 1024*1024

CLIENT_NUM = 10
CLIENT_WEIGHTS = [1 for _ in range(CLIENT_NUM)]

""" TRAIN PARAM"""
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATASET = "MNIST" # ["MNIST","FASHION-MNIST","CIFAR10","CIFAR100"]
NUM_CLASSES = 100 if DATASET == "CIFAR100" else 10
TRAINDATA_SIZE = 50000 if "CIFAR" in DATASET else 60000
EPOCH = 100 if DATASET != "CIFAR100" else 150
BATCH_SIZE_DICT = {10:256, 20:128, 50:100, 100:100}
BATCH_SIZE = BATCH_SIZE_DICT[CLIENT_NUM]
LR = 1E-3
SEED = 999

""" CRYPTO PARAM """
BACKEND = "batchcrypt"  # ["plain", "batchcrypt"]
PRECISION = 8
ENCRYPT_BATCHSIZE = 136

""" PROTOCOL 2 """
SWITCH_MODE = "thre"
#SWITCH_MODE = "pred"
PARASCOUNT = True
LOAD_MODEL = False
SAVE_MODEL = False
K = 3
SPARSE = 0.05
POWER_CHOICE = [2,3]
ADDSPARSE = True
CON_ACC_DICT = {'MNIST':0.9,'FASHION-MNIST':0.85,'CIFAR10':0.64,'CIFAR100':0.4} if not MASK \
                else {'MNIST':0.85,'FASHION-MNIST':0.82,'CIFAR10':0.6,'CIFAR100':0.4}
CON_ACC = CON_ACC_DICT[DATASET]
LOCAL_ANCHORS_PATH = 'choices/local_anchors/client{}-{}-top{}-pw{}-sparse{}/'.format(CLIENT_NUM,DATASET,K,POWER_CHOICE[1],SPARSE)
LOAD_MODEL_PATH = 'models/model_{}ResNet_{}.pt'.format(DATASET,0.7)
SAVE_MODEL_PATH = 'models/model_{}ResNet_{}.pt'.format(DATASET,CON_ACC)
SWITCH_MODEL_PATH = 'switch/switch_model_{}.pth'.format('CIFAR' if 'CIFAR' in DATASET else 'MNIST')

SWITCH_DIR = 'pred1' if PRED_CON and SWITCH_MODE == 'pred' else SWITCH_MODE
ACC_RECORDS_PATH = 'records/{}/acc_records/{}/records-clients{}-top{}-sparse{}-{}-{}.npy'.\
                    format(SWITCH_DIR,DATASET,CLIENT_NUM,K,SPARSE,DATASET,SWITCH_DIR)
CHOICE_PATH = 'choices/{}/{}/choices-client{}-top{}-stan{}-sparse{}-{}-{}.npy'.\
                    format(SWITCH_DIR,DATASET,CLIENT_NUM,K,POWER_CHOICE[1],SPARSE,DATASET,SWITCH_DIR)
PRE_RECORDS_PATH = 'records/{}/pre_records/pre_records-clients{}-top{}-sparse{}-{}.npy'.format(SWITCH_DIR,CLIENT_NUM,K,SPARSE,DATASET)
PRE_PARAMS_PATH = 'records/{}/pre_params/pre_params-clients{}-top{}-sparse{}-{}.npy'.format(SWITCH_DIR,CLIENT_NUM,K,SPARSE,DATASET)

""" LOG Config """
LOG_DIR = "logs/{}-{}bit-{}clients/".format(BACKEND, PRECISION, CLIENT_NUM)
if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
LOG_LEVEL = logging.DEBUG

