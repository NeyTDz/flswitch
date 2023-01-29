import CryptNet_client
import CryptNet_server
from train_params import *
from load_data import mk_dirs
from multiprocessing import Process, set_start_method
import time
import os

if __name__ == "__main__":
    mk_dirs()
    start_time = time.time()
    set_start_method('spawn')
    process_list = []
    p_server = Process(target=CryptNet_server.server_func, args=())
    p_server.start()
    process_list.append(p_server)
    time.sleep(2)

    for i in range(CLIENT_NUM):
        p = Process(target=CryptNet_client.client_func, args=(i, ))
        p.start()
        process_list.append(p)
        time.sleep(1)

    for p in process_list:
        p.join()
    end_time = time.time()
    print('Start:',time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(start_time)))
    print('End:  ',time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(end_time)))
    print('Cost: ',time.strftime("%H:%M:%S", time.gmtime(end_time-start_time)))
    print('end')

