from tkinter import E
import numpy as np
import torch
import pickle
import torch.nn as nn
from model_paras_count.pre_process import *
from model_paras_count.count import *
from network.rbphe_network import ObRBPHENetwork
from network.plain_network import PlainNetwork
from socket import socket, AF_INET, SOCK_STREAM
from train_params import *
import argparse
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import math
import time
import logging


def server_func():
    logging.basicConfig(level=LOG_LEVEL,
                        format='%(asctime)s %(filename)s %(levelname)s %(message)s',
                        datefmt='%a %d %b %Y %H:%M:%S',
                        filename="{}/server.log".format(LOG_DIR),
                        filemode="w")
    logging.debug("Debug mode")

    """ Init public key """
    lg_max_add = max(6, math.ceil(math.log2(CLIENT_NUM)))
    if BACKEND == "obrbphe":
        model = ObRBPHENetwork(PRECISION, key_size=KEY_SIZE, lg_max_add=lg_max_add,
                               sec_param=SECURITY)
    else:
        model = PlainNetwork()
        
    encryptor = model.encryptor

    """ Init socket """

    ServerSocket = socket()
    ServerSocket.bind((SERVER_IP, SERVER_PORT))
    ServerSocket.listen(5)

    leader_conn, address = ServerSocket.accept()

    """ Send public key """
    sendData = pickle.dumps(encryptor)
    connections = {}
    for _ in range(CLIENT_NUM):
        conn, address = ServerSocket.accept()
        conn_id = pickle.loads(conn.recv(BUFF_SIZE))
        connections[conn_id] = conn
        #print(conn_id,connections[conn_id])
        conn.send(sendData)
    
    """** Local Anchors **"""
    #if not os.path.exists(LOCAL_ANCHORS_PATH):
    #    os.makedirs(LOCAL_ANCHORS_PATH)

    logging.debug("init success")
    logging.info("Params: MODE:{} CLIENTS:{} K:{} SPARSE:{} DATASET:{}".\
                 format(SWITCH_MODE,CLIENT_NUM,K,SPARSE,DATASET))
    protocol_switch = '1'
    switch_back = False
    batch_num = math.ceil(TRAINDATA_SIZE // CLIENT_NUM / BATCH_SIZE) if CLIENT_NUM < 50 else math.ceil(TRAINDATA_SIZE // 25 / BATCH_SIZE)
    model_chose = np.zeros((EPOCH,batch_num,CLIENT_NUM)) # save distribution of selected clients
    for epoch in range(EPOCH):
        for b in range(batch_num):
            logging.debug("epoch: {} batch seq: {}".format(epoch,b))
            print("epoch:",epoch,"step:",b)

            '''recv switch from leader and broadcast'''
            # recv switch info from leader
            protocol_switch = pickle.loads(leader_conn.recv(BUFF_SIZE))
            #leader_conn.send(pickle.dumps(COMP_MSG))
            logging.debug("receive switch from leader")

            time.sleep(0.5)
            # broadcast switch info to clients
            for i in range(1,CLIENT_NUM):
                conn = connections[i]
                #print(i,conn)
                conn.send(pickle.dumps(protocol_switch))
                #resp = conn.recv(BUFF_SIZE)
                #if pickle.loads(resp) != COMP_MSG:
                #    raise ValueError("receive unrecognized message!") 
            logging.debug("broadcast switch")
            #time.sleep(0.8)

            '''trans process'''                     
            if PARASCOUNT and protocol_switch == '2':
                # recieve hash of anchors
                hash_list = None
                for i in range(CLIENT_NUM):
                    conn = connections[i]
                    remain = pickle.loads(conn.recv(BUFF_SIZE))
                    conn.send(pickle.dumps(READY_MSG))
                    recvData = b""
                    while remain > 0:
                        each_data = conn.recv(BUFF_SIZE)
                        remain -= len(each_data)
                        recvData += each_data
                    hashs = pickle.loads(recvData)
                    if i == 0:
                        hash_list = hashs
                    else:
                        hash_list = np.vstack((hash_list,hashs))
                    #conn.send(pickle.dumps(COMP_MSG))
                    
                logging.debug("receive clients' hash list")

                # Model Paras Count
                # add restore mask 2022.4.26
                hash_list = hash_list.T
                alloc,selected,sparse_index = pick_represents(hash_list,K,SPARSE,ENCRYPT_BATCHSIZE)
                alloc_num = len(alloc.keys())
                selected_index = []
                selected_info = dict()
                selected_freq = np.zeros(CLIENT_NUM).astype(int)
                begin,end = 0,0
                for c in selected:
                    begin = end
                    end = begin + len(alloc[c])
                    selected_info[c] = {'index':alloc[c],'begin':begin,'end':end}
                    selected_index += alloc[c]
                    selected_freq[c] = len(alloc[c])
                print([(key,len(selected_info[key]['index'])) for key in selected_info.keys()])
                #logging.debug([(key,len(selected_info[key]['index'])) for key in selected_info.keys()])
                selected_index += sparse_index
                restore_mask = np.argsort(np.array(selected_index))
                selected_vec = one_hot(selected,CLIENT_NUM)
                model_chose[epoch,b,:] = selected_freq
                # If only one client in selection, switch back to protocol 1
                if SWITCH_MODE == 'thre' and len(selected) <= 1:
                    switch_back = True
                    print("switch back to protocol 1")
                else:
                    switch_back = False

                np.save(CHOICE_PATH,model_chose)

                # send indexes to selected clients
                # send selected info to clients 2022.4.26
                for i in range(CLIENT_NUM):
                    if selected_vec[i]:
                        sendData = pickle.dumps({'switch_back':switch_back,'selected':True,'remask':restore_mask,'sparse':sparse_index,'selected_info':selected_info[i]})
                    else:
                        sendData = pickle.dumps({'switch_back':switch_back,'selected':False,'remask':restore_mask,'sparse':sparse_index})
                    remain = len(sendData)
                    conn = connections[i]
                    conn.send(pickle.dumps(remain))
                    resp = conn.recv(BUFF_SIZE)
                    if pickle.loads(resp) != READY_MSG:
                        raise ValueError("receive unrecognized message!")
                    conn.send(sendData)
                        
                logging.debug("send alloc info to clients")

                if not switch_back:
                    # receive anchors and residues
                    grad_anchors,grad_residues = None, None
                    first_conn = min(list(alloc.keys()))
                    for i in range(CLIENT_NUM):
                        conn = connections[i]
                        if selected_vec[i]:
                            # recv anchors
                            remain = pickle.loads(conn.recv(BUFF_SIZE))
                            conn.send(pickle.dumps(READY_MSG))
                            recvData = b""
                            while remain > 0:
                                each_data = conn.recv(BUFF_SIZE)
                                remain -= len(each_data)
                                recvData += each_data
                            
                            anch_res = pickle.loads(recvData)
                            if i == first_conn:
                                grad_anchors = np.array(anch_res['anchors'])
                                grad_residues = np.array(anch_res['residues'])
                            else:
                                grad_anchors += np.array(anch_res['anchors'])
                                grad_residues += np.array(anch_res['residues'])
                            #conn.send(pickle.dumps(COMP_MSG))

                    logging.debug("receive local anchors & residues from clients")

                    # broadcast global gradients
                    grad_residues /= alloc_num
                    global_grads = [grad_anchors,grad_residues]                
                    sendData = pickle.dumps(global_grads)
                    remain = len(sendData)
                    for i in range(CLIENT_NUM):
                        conn = connections[i]
                        conn.send(pickle.dumps(remain))
                        resp = conn.recv(BUFF_SIZE)
                        if pickle.loads(resp) != READY_MSG:
                            raise ValueError("receive unrecognized message!")
                        conn.send(sendData)
                        #resp = conn.recv(BUFF_SIZE)
                        #if pickle.loads(resp) != COMP_MSG:
                        #    raise ValueError("receive unrecognized message!")
                    logging.debug("send global anchors & residues to clients")   
                else:
                    logging.debug("switch back to protocol 1 in epoch {} step {}".format(epoch,b))
            elif protocol_switch == '1':
                # receive local encrypted gradients
                agg_grad = None
                for i in range(CLIENT_NUM):
                    conn = connections[i]  
                    remain = pickle.loads(conn.recv(BUFF_SIZE))
                    conn.send(pickle.dumps(READY_MSG))
                    recvData = b""
                    while remain > 0:
                        each_data = conn.recv(BUFF_SIZE)
                        remain -= len(each_data)
                        recvData += each_data
                    grads = pickle.loads(recvData)
                    if i == 0:
                        agg_grad = np.array(grads)
                    else:
                        agg_grad += np.array(grads)
                    #conn.send(pickle.dumps(COMP_MSG))
                logging.debug("receive local gradients from clients")

                # send global encrypted gradients
                #time.sleep(0.1)
                sendData = pickle.dumps(agg_grad)
                remain = len(sendData)
                for i in range(CLIENT_NUM):
                    conn = connections[i]
                    conn.send(pickle.dumps(remain))
                    resp = conn.recv(BUFF_SIZE)
                    if pickle.loads(resp) != READY_MSG:
                        raise ValueError("receive unrecognized message!")
                    conn.send(sendData)
                    #resp = conn.recv(BUFF_SIZE)
                    #if pickle.loads(resp) != COMP_MSG:
                    #    raise ValueError("receive unrecognized message!") 
                logging.debug("send global gradients to clients")
            else:
                print('unknown protocol!')
                assert 0

if __name__ == "__main__":
    server_func()


