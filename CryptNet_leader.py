import torch
import pickle
import torch.nn as nn
from network.rbphe_network import ObRBPHENetwork
from network.plain_network import PlainNetwork
from socket import socket, AF_INET, SOCK_STREAM
from train_params import *
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import math
import time
import logging


def leader_func():
    logging.basicConfig(level=LOG_LEVEL,
                        format='%(asctime)s %(filename)s %(levelname)s %(message)s',
                        datefmt='%a %d %b %Y %H:%M:%S',
                        filename="{}/leader.log".format(LOG_DIR),
                        filemode="w")
    logging.debug("leader start")
    """ Init public key """
    lg_max_add = max(6, math.ceil(math.log2(CLIENT_NUM)))
    if BACKEND == "obrbphe":
        model = ObRBPHENetwork(PRECISION, key_size=KEY_SIZE, lg_max_add=lg_max_add,
                               sec_param=SECURITY)
    else:
        model = PlainNetwork()
        
    encryptor = model.encryptor

    """ Init socket """
    while True:
        try:
            tcp_client_socket_server = socket(AF_INET, SOCK_STREAM)
            tcp_client_socket_server.connect((SERVER_IP, SERVER_PORT))
            break
        except:
            logging.info("waiting for server...")
            time.sleep(3)

    LeaderSocket = socket()
    LeaderSocket.bind((LEADER_IP, LEADER_PORT))
    LeaderSocket.listen(5)

    """ Send public key """
    sendData = pickle.dumps(encryptor)
    connections = []
    for _ in range(CLIENT_NUM):
        conn, address = LeaderSocket.accept()
        connections.append(conn)
        conn.send(sendData)
    logging.debug("init success")
    batch_num = math.ceil(60000 // CLIENT_NUM / BATCH_SIZE)
    for epoch in range(EPOCH):
        for _ in range(batch_num):
            recvData = tcp_client_socket_server.recv(BUFF_SIZE)
            enc_grad_len = pickle.loads(recvData)
            tcp_client_socket_server.send(pickle.dumps(READY_MSG))
            recvData = b""
            while enc_grad_len > 0:
                each_data = tcp_client_socket_server.recv(BUFF_SIZE)
                recvData += each_data
                enc_grad_len -= len(each_data)
            enc_grad = pickle.loads(recvData)
            logging.debug("receive server encrypted gradients")
            grad_dict = model.unpack_grad(enc_grad)
            sendData = pickle.dumps(grad_dict)
            for conn in connections:
                conn.send(pickle.dumps(len(sendData)))
                resp = conn.recv(BUFF_SIZE)
                if pickle.loads(resp) != READY_MSG:
                    raise ValueError("receive unrecognized message!")
                conn.send(sendData)
            logging.debug("send decrypted gradients to clients")
    tcp_client_socket_server.close()


if __name__ == "__main__":
    leader_func()

