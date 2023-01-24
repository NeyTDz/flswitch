import math
from rbphe.obfusacted_residue_cryptosystem import ObfuscatedRBPHE
from rbphe.residue_cryptosystem import RBPHE
from comparison.batchcrypt.batch_crypt import BatchCrypt
from timeit import default_timer as timer
import numpy as np
import sys

CLIENT_NUM = 10


def BatchCrypt_perf(input_data, precision=10, add_max=10, key_size=1024):
    encryptor = BatchCrypt(precision, clipping=1.0, add_max=add_max, padding=2, key_size=key_size)
    encode_batch_size = encryptor.batch_size
    diff_max = 0.0
    enc_total = 0
    dec_total = 0
    add_total = 0
    data_size = len(input_data)
    print("\nBatchCrypt encode_batch_size:",encode_batch_size)
    print("BatchCrypt batch num: ", math.ceil(len(input_data)/encode_batch_size))
    for i in range(math.ceil(len(input_data)/encode_batch_size)):
        # encrypt
        begin = timer()
        ct = encryptor.encrypt(input_data[i*encode_batch_size:(i+1)*encode_batch_size])
        end = timer()
        enc_total += (end - begin)
        # add
        begin = timer()
        for _ in range(CLIENT_NUM-1):
            ct_add = ct + ct
        end = timer()
        add_total += (end - begin)
        # decrypt
        begin = timer()
        vals = encryptor.decrypt(ct)
        end = timer()
        dec_total += (end-begin)
        l = len(input_data[i*encode_batch_size:(i+1)*encode_batch_size])
        diff = np.abs(np.array(vals[:l]) - np.array(input_data[i*encode_batch_size:(i+1)*encode_batch_size]))
        local_max = np.max(diff)
        diff_max = diff_max if diff_max > local_max else local_max
    batch_num = math.ceil(len(input_data)/encode_batch_size)
    ct_size = batch_num * sys.getsizeof(ct.message["ciphertext"])
    print("BatchCrypt: ", enc_total, dec_total, add_total, diff_max, batch_num, ct_size)
    return enc_total, dec_total, add_total, diff_max, math.ceil(len(input_data)/encode_batch_size)


def ObRBPHE_perf(input_data, precision=10, lg_max_add=4, key_size=1024):
    sec_param = 80
    encryptor = ObfuscatedRBPHE(sec_param=sec_param, batch_size="auto", precision=precision,
                                lg_max_add=lg_max_add, key_size=key_size)
    encode_batch_size = encryptor.batch_size
    enc_total = 0
    dec_total = 0
    diff_max = 0.0
    add_total = 0
    mul_total = 0
    data_size = len(input_data)
    print("\nObRBPHEencode_batch_size:",encode_batch_size)
    print("ObRBPHE batch num: ", math.ceil(len(input_data)/encode_batch_size))
    for i in range(math.ceil(len(input_data)/encode_batch_size)):
        # encrypt
        begin = timer()
        ct = encryptor.encrypt(input_data[i*encode_batch_size:(i+1)*encode_batch_size])
        end = timer()
        enc_total += (end - begin)
        # add
        begin = timer()
        for _ in range(CLIENT_NUM-1):
            ct_add = ct + ct
        end = timer()
        add_total += (end - begin)
        # multiply
        w = encryptor.convert_weight(2)
        begin = timer()
        for _ in range(CLIENT_NUM):
            ct_mul = ct * w
        end = timer()
        mul_total += (end - begin)
        # decrypt
        begin = timer()
        vals = encryptor.decrypt(ct)
        end = timer()
        dec_total += (end-begin)
        l = len(input_data[i*encode_batch_size:(i+1)*encode_batch_size])
        diff = np.abs(np.array(vals[:l]) - np.array(input_data[i*encode_batch_size:(i+1)*encode_batch_size]))
        local_max = np.max(diff)
        diff_max = diff_max if diff_max > local_max else local_max

    batch_num = math.ceil(len(input_data)/encode_batch_size)
    ct_size = batch_num * (sys.getsizeof(ct.enc_p_values.message["ciphertext"]) + sys.getsizeof(ct.q_values))
    print("ObRBPHE: ", enc_total, dec_total, add_total, mul_total, diff_max, batch_num, ct_size)
    return enc_total, dec_total, add_total, mul_total, diff_max, math.ceil(len(input_data)/encode_batch_size)


def RBPHE_perf(input_data, precision=10, lg_max_add=4, key_size=1024):
    encryptor = RBPHE(batch_size="auto", precision=precision, lg_max_add=lg_max_add, key_size=key_size)
    encode_batch_size = encryptor.batch_size
    enc_total = 0
    dec_total = 0
    diff_max = 0.0
    add_total = 0
    mul_total = 0
    data_size = len(input_data)
    print("\nRBPHE encode_batch_size:",encode_batch_size)
    print("RBPHE batch num: ", math.ceil(len(input_data)/encode_batch_size))
    for i in range(math.ceil(len(input_data)/encode_batch_size)):
        # encrypt
        begin = timer()
        ct = encryptor.encrypt(input_data[i*encode_batch_size:(i+1)*encode_batch_size])
        end = timer()
        enc_total += (end - begin)
        # add
        begin = timer()
        for _ in range(CLIENT_NUM-1):
            ct_add = ct + ct
        end = timer()
        add_total += (end - begin)
        # multiply
        begin = timer()
        for _ in range(CLIENT_NUM-1):
            ct_add = ct * 2
        end = timer()
        mul_total += (end - begin)
        # decrypt
        begin = timer()
        vals = encryptor.decrypt(ct)
        end = timer()
        dec_total += (end-begin)
        l = len(input_data[i*encode_batch_size:(i+1)*encode_batch_size])
        diff = np.abs(np.array(vals[:l]) - np.array(input_data[i*encode_batch_size:(i+1)*encode_batch_size]))
        local_max = np.max(diff)
        diff_max = diff_max if diff_max > local_max else local_max

    batch_num = math.ceil(len(input_data)/encode_batch_size)
    ct_size = batch_num * sys.getsizeof(ct.message["ciphertext"])
    print("RBPHE: ", enc_total, dec_total, add_total, mul_total, diff_max, batch_num, ct_size)
    return enc_total, dec_total, add_total, mul_total, diff_max, math.ceil(len(input_data)/encode_batch_size)

if __name__ == "__main__":
    data = np.random.uniform(-1, 1, 65536)
    print(data)
    result1 = BatchCrypt_perf(data, precision=8, add_max=10, key_size=2048)

    result2 = ObRBPHE_perf(data, precision=8, lg_max_add=4, key_size=2048)

    result3 = RBPHE_perf(data, precision=8, lg_max_add=4, key_size=2048)

    # compute ciphertext size
    # since each ciphertext contain public key in python-paillier, we only consider the message...
    message_size_bytes = 572 # 572 for 2048 key size (300 for 1024)

