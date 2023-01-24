import sys
from paillier.paillier import CPaillier
from rbphe.obfuscated_batch_encoder import ObfuscatedBatchEncoder
from rbphe.crt import CRT
import math
from timeit import default_timer as timer

sys.path.append('..')
from train_params import *

class ObfuscatedRBPHEPlaintext(object):
    def __init__(self, p_values, q_values):
        self.p_values = p_values
        self.q_values = q_values

    def __add__(self, other):
        return ObfuscatedRBPHEPlaintext(self.p_values + other.p_values, self.q_values + other.q_values)

    def __mul__(self, other):
        return ObfuscatedRBPHEPlaintext(self.p_values * other.p_values, self.q_values * other.q_values)


class ObfuscatedRBPHECiphertext(object):
    def __init__(self, enc_p_values, q_values):
        self.enc_p_values = enc_p_values
        self.q_values = q_values

    def __add__(self, other):
        return ObfuscatedRBPHECiphertext(self.enc_p_values + other.enc_p_values, self.q_values + other.q_values)

    def __mul__(self, other: ObfuscatedRBPHEPlaintext):
        return ObfuscatedRBPHECiphertext(self.enc_p_values * other.p_values, self.q_values * other.q_values)


class ObfuscatedRBPHE(object):
    def __init__(self, sec_param=80, batch_size="auto", precision=10, lg_max_add=10, key_size=1024,
                 encryptor=None, clipping=8):
        """
        RBPHE Encryptor
        :param sec_param: int, security length for random value
        :param batch_size: if "auto", generate a proper (as large as possible) batch_size under given precision,
        key_size and lg_max_add.
        :param lg_max_add: int, bit length for generate approximate allowed maximum add number,
                the maximum add number is usually smaller than pow(2, lg_max_add),
                use ObfuscatedRBPHE.max_add to get the accurate maximum add number

        """
        self.sec_param = sec_param
        if batch_size == "auto":
            self.batch_size = self.gen_batch_size(lg_max_add, precision, sec_param, key_size)
        else:
            self.batch_size = batch_size
        
        self.precision = precision
        self.lg_max_add = lg_max_add
        self.key_size = key_size
        if not encryptor:
            self.encoder = ObfuscatedBatchEncoder(self.batch_size, precision, sec_param, lg_max_add, clipping)
            self.cipher = CPaillier.generate(key_size)
        else:
            self.encoder = encryptor[0]
            self.cipher = encryptor[1]

    @property
    def encryptor(self):
        return self.encoder, self.cipher

    @property
    def max_add(self):
        return self.encoder.max_add

    @staticmethod
    def gen_batch_size(log_max_add, bit_length, security, key_size):
        return ObfuscatedBatchEncoder.gen_batch_size(log_max_add, bit_length, security, key_size)

    @property
    def public_key(self):
        return self.cipher.public_key

    def encrypt(self, messages):
        #print('m:',len(messages))
        p_value, q_value = self.encoder.encode(messages)
        #print('pq:',p_value, q_value)
        enc_p_value = self.cipher.encrypt(p_value)
        return ObfuscatedRBPHECiphertext(enc_p_value, q_value)

    def decrypt(self, ciphertext: ObfuscatedRBPHECiphertext):
        p_value = self.cipher.decrypt(ciphertext.enc_p_values)
        q_value = ciphertext.q_values
        messages = self.encoder.decode(p_value, q_value)
        return messages

    def convert_weight(self, weight):
        weight_vector = [weight] * (self.batch_size+1)
        encode_weight_p = CRT(weight_vector, self.encoder.p_primes)
        encode_weight_q = CRT(weight_vector, self.encoder.q_primes)
        return ObfuscatedRBPHEPlaintext(encode_weight_p, encode_weight_q)
