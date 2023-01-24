from timeit import default_timer as timer
import numpy as np
from paillier.paillier import CPaillier
from rbphe.obfusacted_residue_cryptosystem import ObfuscatedRBPHE
from rbphe.residue_cryptosystem import RBPHE

batch_size = "auto"
sec_param = 80
key_size = 1024
lg_max_add = 8
precision = 8

cipher = CPaillier.generate(1024)



# def correct_test():
#     encoder = BatchEncoder(batch_size=batch_size, bit_length=10)
#     ob_encoder = ObfuscatedBatchEncoder(batch_size=batch_size, bit_length=10, log_max_add=10, security=80)
#     x = list(np.random.uniform(-1, 1, batch_size))
#     y = list(np.random.uniform(-1, 1, batch_size))
#
#     c1 = random.randint(0, 127)
#     c2 = random.randint(0, 127)
#
#     encode_number1 = encoder.encode(x)
#     encrypt_number1 = cipher.encrypt(encode_number1.value)
#     encrypt_number1 = encrypt_number1 * c1
#
#     encode_number2 = encoder.encode(y)
#     encrypt_number2 = cipher.encrypt(encode_number2.value)
#     encrypt_number2 = encrypt_number2 * c2
#
#     z = encrypt_number1 + encrypt_number2
#     z = cipher.decrypt(z)
#     decode_values = encoder.decode(z)
#
#     correct = list(np.array(x) * c1 + np.array(y) * c2)
#     if (np.abs(np.array(correct) - np.array(decode_values)) > 1).any():
#         return False
#     else:
#         return True

def unit_test():
    np.random.seed(123)
    x = list(np.random.uniform(-1, 1, batch_size))
    y = list(np.random.uniform(-1, 1, batch_size))
    w1 = 1
    w2 = 11
    t0 = timer()
    ob_encryptor = RBPHE(batch_size, precision, lg_max_add, key_size)
    t1 = timer()
    ct1 = ob_encryptor.encrypt(x)
    t2 = timer()
    print("setup time: ", t1 - t0)
    print("encrypt time: ", t2 - t1)

    ct2 = ob_encryptor.encrypt(y)

    w1_pt = ob_encryptor.convert_weight(w1)
    w2_pt = ob_encryptor.convert_weight(w2)
    ct1 = ct1 * w1_pt
    ct2 = ct2 * w2_pt

    ct = ct1 + ct2
    t0 = timer()
    pt = ob_encryptor.decrypt(ct)
    t1 = timer()
    print("decrypt time: ", t1 - t0)

    correct = list(np.array(x)*w1+ np.array(y)*w2)
    print("correct: \t", correct)
    print("decrypt: \t", pt)
    print("diff: \t", list(np.array(correct) - np.array(pt)))


def unit_test_obfuscated():
    # x = list(np.random.uniform(-1, 1, batch_size))
    # y = list(np.random.uniform(-1, 1, batch_size))

    w1 = 5
    w2 = 20
    t0 = timer()
    ob_encryptor = ObfuscatedRBPHE(sec_param, batch_size, precision, lg_max_add, key_size)
    # print(ob_encryptor.max_add)
    x = np.random.uniform(-8, 8, ob_encryptor.batch_size)
    y = np.random.uniform(-8, 8, ob_encryptor.batch_size)
    t1 = timer()
    ct1 = ob_encryptor.encrypt(x)
    t2 = timer()
    # print("setup time: ", t1 - t0)
    # print("encrypt time: ", t2 - t1)

    ct2 = ob_encryptor.encrypt(y)

    w1_pt = ob_encryptor.convert_weight(w1)
    w2_pt = ob_encryptor.convert_weight(w2)
    ct1 = ct1 * w1_pt
    ct2 = ct2 * w2_pt

    ct = ct1 + ct2
    t0 = timer()
    pt = ob_encryptor.decrypt(ct)
    t1 = timer()
    # print("decrypt time: ", t1 - t0)

    correct = list(np.array(x)*w1+ np.array(y)*w2)
    diff = list(np.array(correct) - np.array(pt))
    if np.max(diff) > 1.0:
        print("error!")
    # else:
    #     print("correct: \t", correct)
    #     print("decrypt: \t", pt)
    #     print("diff: \t", diff)



if __name__ == "__main__":
    for _ in range(1000):
        unit_test_obfuscated()
    # for i in range(1000):
    #     if correct_test():
    #         print(i)
    #     else:
    #         print("NO!")
    #         break