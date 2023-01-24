from base64 import encode
import logging

import os
import torch
import pickle
import torch.nn as nn
from model_paras_count.pre_process import *
from model_paras_count.count import *
from model_paras_count.PerfectHash import *
from network.rbphe_network import ObRBPHENetwork
from network.plain_network import PlainNetwork
from socket import socket, AF_INET, SOCK_STREAM
from train_params import *
from switch.switch_network import *
from switch.switch_utils import *
#from sendmail.sendqq import *
import argparse
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from load_data import load_dataset,sep_dataset
import math
import time
import numpy as np
import random
from rbphe.obfusacted_residue_cryptosystem import ObfuscatedRBPHE

device = torch.device(DEVICE)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def client_func(client_idx):
    set_seed(SEED)
    logging.basicConfig(level=LOG_LEVEL,
                        format='%(asctime)s %(filename)s %(levelname)s %(message)s',
                        datefmt='%a %d %b %Y %H:%M:%S',
                        filename="{}/client_{}.log".format(LOG_DIR, client_idx),
                        filemode="w")
    logging.debug("client {} start".format(client_idx))

    train_file,test_file = load_dataset(DATASET)
    assert train_file

    train_file = sep_dataset(client_idx,train_file)

    train_loader = DataLoader(
        dataset=train_file,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader = DataLoader(
        dataset=test_file,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    lg_max_add = max(6, math.ceil(math.log2(CLIENT_NUM)))

    """ Init socket """
    if client_idx == 0: # set client0 as leader in switch
        while True:
            try:
                tcp_leader_socket_server = socket(AF_INET, SOCK_STREAM)
                tcp_leader_socket_server.connect((SERVER_IP, SERVER_PORT))
                break
            except:
                print("leader soocket waiting for leader...")
                time.sleep(3) 
    time.sleep(1) # wait leader connect set up
    while True:
        try:
            tcp_client_socket_server = socket(AF_INET, SOCK_STREAM)
            tcp_client_socket_server.connect((SERVER_IP, SERVER_PORT))
            tcp_client_socket_server.send(pickle.dumps(client_idx))
            break
        except:
            print("waiting for server...")
            time.sleep(3)
    """ Request public key """
    recvData = tcp_client_socket_server.recv(BUFF_SIZE)
    encryptor = pickle.loads(recvData)
    if BACKEND == "obrbphe":
        model = ObRBPHENetwork(PRECISION, key_size=KEY_SIZE, lg_max_add=lg_max_add,\
                               sec_param=SECURITY, encryptor=encryptor).to(device)
    else:
        model = PlainNetwork(encryptor=encryptor).to(device)

    """ Switch model """
    if SWITCH_MODE == 'pred' and client_idx == 0:
        switch_model = PredictionNet().to(device)
        switch_model.load_state_dict(torch.load(SWITCH_MODEL_PATH,map_location=device))
        switch_model.to(device)
    #switch_model = torch.load(SWITCH_MODEL_PATH,map_location=device)
    
    """ Hash Func """
    hash_scale = 10**(2+POWER_CHOICE[1])
    perfect_hash = UniversalHash(hash_scale)
    optim = torch.optim.Adam(model.parameters(), LR)
    lossf = nn.CrossEntropyLoss()
    logging.debug("init success")


    control_acc = 0
    protocol_switch = '1'
    switch_back = False
    switch_threshold = False
    acc_records = []
    protocol_fix = [False,'2']
    pre_con = 0
    #local_anch_records = []
    global_params = []
    if LOAD_MODEL:
        model.load_state_dict(torch.load(LOAD_MODEL_PATH,map_location=device))  # load model

    batch_num = math.ceil(TRAINDATA_SIZE // CLIENT_NUM / BATCH_SIZE) if CLIENT_NUM < 50 else math.ceil(TRAINDATA_SIZE // 25 / BATCH_SIZE)
    for epoch in range(EPOCH):
        epoch_waste = []
        for step, (data, targets) in enumerate(train_loader):
            logging.debug("epoch: {} batch seq: {},{}".format(epoch,step,batch_num))
            optim.zero_grad()
            data = data.to(device)
            targets = targets.to(device)
            output = model(data)
            loss = lossf(output, targets)
            loss.backward()
            if client_idx == 0: #leader
                if SWITCH_MODE == 'thre':
                    if protocol_switch == '1':
                        if control_acc > CON_ACC:
                            protocol_switch = '2'
                            if not switch_threshold and SAVE_MODEL:
                                model.save_model(SAVE_MODEL_PATH)
                            switch_threshold = True # do not save later
                    if protocol_switch == '2':
                        if switch_back or control_acc < CON_ACC:
                            switch_back = False
                            protocol_switch = '1'
                            # when switch back, stay protocol 1 until next step
                            control_acc = CON_ACC if control_acc >= CON_ACC else control_acc
                    logging.debug("batch seq: {} protocol: {}".format(step,protocol_switch))
                elif SWITCH_MODE == 'pred':
                    if PRED_CON and protocol_fix[0]:
                        protocol_switch = protocol_fix[1]
                        logging.debug("batch seq: {} protocol: {} stable".format(step,protocol_switch))
                        print("predict: protocol: {} stable".format(protocol_switch))
                    else:
                        if epoch > 1:
                            curr_index = epoch*batch_num+step
                            acc_loss_vec = pre_records[curr_index-batch_num:curr_index,6:].T
                            acc_loss_vec = torch.tensor(acc_loss_vec).float().to(device)
                            protocol_switch,raw_pre = switch_model.predict(acc_loss_vec)
                            logging.debug("batch seq: {} protocol: {} raw predict: {}".format(step,protocol_switch,raw_pre))
                            print("predict: {} in {}, so protocol: {}".format(raw_pre,[0] if protocol_switch=='1' else [1,2],protocol_switch))
                            if PRED_CON:
                                if protocol_switch == '1' and pre_con > 0: 
                                    pre_con = 0
                                if protocol_switch  == '2':
                                    pre_con += 1
                                if pre_con >= batch_num*3:
                                    protocol_fix = [True,'2']
                    
                else:
                    print("unknown switch mode!")
                    assert 0

                sendSignal = pickle.dumps(protocol_switch)
                tcp_leader_socket_server.send(sendSignal)
                #resp = tcp_leader_socket_server.recv(BUFF_SIZE)
                #if pickle.loads(resp) != COMP_MSG:
                #    raise ValueError("receive unrecognized message!") 
                
                
            else: # not leader
                #time.sleep(0.1)
                resp = tcp_client_socket_server.recv(BUFF_SIZE)
                #tcp_client_socket_server.send(pickle.dumps(COMP_MSG))
                protocol_switch = pickle.loads(resp)
                logging.debug("batch seq: {} protocol: {}".format(step,protocol_switch))
            '''trans process''' 
            if PARASCOUNT and protocol_switch == '2':
                # send hash of anchors
                raw_grads = model.get_raw_grads()
                anchors, residues, stand_power = sep_anch_resd(raw_grads,POWER_CHOICE)
                hashs = generate_hashlist(anchors,perfect_hash)
                '''
                encode_hashs = []
                obf = ObfuscatedRBPHE()
                for i in range(math.ceil(len(hashs)/ENCRYPT_BATCHSIZE)):
                    ent = obf.encoder.encode(hashs[i*ENCRYPT_BATCHSIZE:(i+1)*ENCRYPT_BATCHSIZE])
                    encode_hashs.append(ent)
                '''
                sendData = pickle.dumps(hashs)
                remain = len(sendData)
                logging.debug("encode hashs complete, endoce size:{}".format(remain))
                tcp_client_socket_server.send(pickle.dumps(remain))
                resp = tcp_client_socket_server.recv(BUFF_SIZE)
                if pickle.loads(resp) != READY_MSG:
                    raise ValueError("receive unrecognized message!")
                tcp_client_socket_server.send(sendData)
                #resp = tcp_client_socket_server.recv(BUFF_SIZE)
                #if pickle.loads(resp) != COMP_MSG:
                #    raise ValueError("receive unrecognized message!")                
                logging.debug("send hash of anchors to server. stand power:{}".format(stand_power))

                # receive indexs and send anchors&residues
                index_len = pickle.loads(tcp_client_socket_server.recv(BUFF_SIZE))
                tcp_client_socket_server.send(pickle.dumps(READY_MSG))
                recvData = b"" 
                remain = index_len
                while remain > 0:
                    each_data = tcp_client_socket_server.recv(BUFF_SIZE)
                    remain -= len(each_data)
                    recvData += each_data
                alloc_info = pickle.loads(recvData)
                #tcp_client_socket_server.send(pickle.dumps(COMP_MSG))

                switch_back = alloc_info['switch_back']
                remask = alloc_info['remask']
                sparse_index = alloc_info['sparse']
                if not switch_back:
                    if alloc_info['selected']:
                        #st = time.perf_counter()
                        # alloc
                        selected_info = alloc_info['selected_info']
                        anchors = anchors.reshape(len(anchors))
                        residues = residues.reshape(len(residues))
                        #local_anchors = {'epoch':epoch,'step':step,'client':client_idx,'top_anch':top_tiny(anchors,10)}
                        #local_anch_records.append(local_anchors)
                        #np.save(LOCAL_ANCHORS_PATH+'client'+str(client_idx)+'.npy',local_anch_records,allow_pickle=True)
                        if not MASK:
                            plain_anchors = np.zeros(len(anchors))
                            plain_anchors[selected_info['index']] = anchors[selected_info['index']]
                            sendData = pickle.dumps({'anchors':plain_anchors,'residues':residues})
                        else:
                            part_anchors = anchors[selected_info['index']]
                            border = [selected_info['begin'],selected_info['end']]
                            logging.debug("receive index from server")
                            pack_anchors = model.pack_grad_with_anchors(part_grad = part_anchors,border = border)
                            sendData = pickle.dumps({'anchors':pack_anchors,'residues':residues})
                        # send 
                        remain = len(sendData)
                        tcp_client_socket_server.send(pickle.dumps(remain))
                        resp = tcp_client_socket_server.recv(BUFF_SIZE)
                        if pickle.loads(resp) != READY_MSG:
                            raise ValueError("receive unrecognized message!")
                        tcp_client_socket_server.send(sendData)
                        #resp = tcp_client_socket_server.recv(BUFF_SIZE)
                        #if pickle.loads(resp) != COMP_MSG:
                        #    raise ValueError("receive unrecognized message!")                    
                        logging.debug("send local anchors&residues to server")
                        #et = time.perf_counter()
                        #print('alloc time',et-st)
                    else:
                        logging.debug("not selected this turn")
                                            
                    # receive global anchors & residues
                    grad_len = pickle.loads(tcp_client_socket_server.recv(BUFF_SIZE))
                    tcp_client_socket_server.send(pickle.dumps(READY_MSG))
                    recvData = b""
                    while grad_len > 0:
                        each_data = tcp_client_socket_server.recv(BUFF_SIZE)
                        grad_len -= len(each_data)
                        recvData += each_data
                    #tcp_client_socket_server.send(pickle.dumps(COMP_MSG))
                    logging.debug("receive global anchors & residues from server")
                    anchors = anchors.reshape(len(anchors))
                    if not MASK:
                        [plain_anchors,plain_residues] = pickle.loads(recvData)

                        grad_dict = model.plain_anchor_residues_process(plain_anchors,plain_residues,stand_power,anchors,sparse_index)
                    else:
                        [enc_anchors,enc_residues] = pickle.loads(recvData)
                        grad_dict = model.unpack_grad_with_residues(anchors,enc_anchors,enc_residues,sparse_index,stand_power,remask)
                else:
                    logging.debug("switch back to protocol 1 in epoch {} step {}".format(epoch,step))
                
            # after IF                          
            elif protocol_switch == '1':
                # send local encrypted gradients
                if not MASK:
                    flat_grad = model.get_raw_grads()
                    sendData = pickle.dumps(flat_grad)
                else:
                    pack_grads, pack_size = model.pack_grad(return_size=True,return_waste=WASTE)
                    logging.debug("encryption complete, pack_size:{}".format(pack_size))
                    #print('pk',pack_size)
                    sendData = pickle.dumps(pack_grads)
                remain = len(sendData)
                tcp_client_socket_server.send(pickle.dumps(remain))
                resp = tcp_client_socket_server.recv(BUFF_SIZE)
                if pickle.loads(resp) != READY_MSG:
                    raise ValueError("receive unrecognized message!")
                tcp_client_socket_server.send(sendData)
                #resp = tcp_client_socket_server.recv(BUFF_SIZE)
                #if pickle.loads(resp) != COMP_MSG:
                #    raise ValueError("receive unrecognized message!")                
                logging.debug("send local gradients to server")
                # receive global encrypted gradients
                grad_len = pickle.loads(tcp_client_socket_server.recv(BUFF_SIZE))
                tcp_client_socket_server.send(pickle.dumps(READY_MSG))
                recvData = b""
                while grad_len > 0:
                    each_data = tcp_client_socket_server.recv(BUFF_SIZE)
                    grad_len -= len(each_data)
                    recvData += each_data
                #tcp_client_socket_server.send(pickle.dumps(COMP_MSG))
                logging.debug("receive global gradients from server")
                if not MASK:
                    plain_gradients = pickle.loads(recvData)
                    grad_dict = model.plain_grad_process(plain_gradients)
                else:
                    enc_gradients = pickle.loads(recvData)
                    grad_dict = model.unpack_grad(enc_gradients)
                    logging.debug("decryption complete")
                
                
            else:
                print('unknown protocol!')
                assert 0
            
            '''train process''' 
            for k, v in model.named_parameters():
                v.grad = grad_dict[k].to(device).type(dtype=v.grad.dtype)    
            optim.step()
            if SWITCH_MODE == "pred":
            #if True:
                if client_idx == 0:
                    #st = time.perf_counter()
                    batch_loss,batch_acc = model.test(test_loader,lossf)
                    if epoch == 0:
                        if step == 0:
                            pre_records = np.array([epoch,step,batch_acc,batch_loss,10.,10.,10.,10.])
                        else:
                            pre_records = np.vstack((pre_records,np.array([epoch,step,batch_acc,batch_loss,10.,10.,10.,10.])))
                    else:
                        curr_index = epoch*batch_num+step
                        mean_acc,mean_loss = np.mean(pre_records[curr_index-batch_num+1:curr_index+1,2:4],axis=0)
                        last_mean_acc,last_mean_loss = pre_records[curr_index-1][4],pre_records[curr_index-1][5]
                        delta_acc,delta_loss = mean_acc-last_mean_acc,mean_loss-last_mean_loss
                        pre_records = np.vstack((pre_records,np.array([epoch,step,batch_acc,batch_loss,mean_acc,mean_loss,delta_acc,delta_loss])))
                    np.save(PRE_RECORDS_PATH,pre_records)
                    #global_params.append(grad_dict)
                    #np.save(PRE_PARAMS_PATH,global_params)
                    #et = time.perf_counter()
                    #print('one shot',et-st)   
                    if step % 10 == 0 and step < 20: #show acc in testset
                        acc_records.append([epoch,step,batch_loss,batch_acc,int(protocol_switch)])
                        np.save(ACC_RECORDS_PATH,acc_records)
                        print("client {}, epoch: {}, step: {}, loss: {}, acc: {}".format(client_idx, epoch, step, batch_loss, batch_acc))
                        logging.info("client {}, epoch: {}, step: {}, loss: {}, acc: {}".format(client_idx,epoch, step, batch_loss, batch_acc))
                        control_acc = batch_acc

            elif SWITCH_MODE == "thre":
                if client_idx == 0:
                    if step % 10 == 0 and step < 20: #show acc in testset
                        batch_loss,batch_acc = model.test(test_loader,lossf)
                        acc_records.append([epoch,step,batch_loss,batch_acc,int(protocol_switch)])
                        np.save(ACC_RECORDS_PATH,acc_records)
                        print("client {}, epoch: {}, step: {}, loss: {}, acc: {}".format(client_idx, epoch, step, batch_loss, batch_acc))
                        logging.info("client {}, epoch: {}, step: {}, loss: {}, acc: {}".format(client_idx,epoch, step, batch_loss, batch_acc))
                        control_acc = batch_acc

        #show avarage acc in this epoch
        if client_idx == 0:
            epoch_loss,epoch_acc = model.test(test_loader,lossf)
            print("client {}, epoch: {}, loss: {}, acc: {}".format(client_idx, epoch, epoch_loss, epoch_acc))
            logging.info("---------- epoch: {}, loss: {}, acc: {} ----------".format(epoch, epoch_loss, epoch_acc))
        # waste
        #print("client {}, Mean waste:{}".format(client_idx, sum(epoch_waste)/len(epoch_waste)))
        #all_waste.append(epoch_waste)
        #logging.info("Mean waste:{}",format(epoch_waste))
        #for name, parameters in model.named_parameters():
        #    logging.debug("name: {}, size: {}, weights: {}".format(name, parameters.size(), parameters.flatten()[:5]))
        #logging.info("--------------------------------------------------")

    tcp_client_socket_server.close()

    '''
    if SAVE_MODEL and client_idx == 0 and not switch_1_2:
        print('end accuracy:',acc)
        model.save_model(SAVE_MODEL_PATH)
    '''


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--index", type=int, help="client index, from 0 to CLIENT_NUM-1")
    args = parser.parse_args()
    if args.index >= CLIENT_NUM or args.index < 0:
        raise ValueError("Invalid client index")

    client_idx = args.index
    client_func(client_idx)

