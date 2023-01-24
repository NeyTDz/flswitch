from paillier.paillier import CPaillier
from paillier.integer_arithmetic import IntegerArithmetic
from rbphe.batch_encoder import BatchEncoder
from rbphe.crt import CRT
import math


class RBPHE(object):
    def __init__(self, batch_size="auto", precision=10, lg_max_add=10, key_size=1024, encryptor=None):
        if batch_size == "auto":
            self.batch_size = self.gen_batch_size(lg_max_add, precision, key_size)
        else:
            self.batch_size = batch_size
        self.precision = precision
        self.lg_max_add = lg_max_add
        self.key_size = key_size
        if not encryptor:
            self.encoder = BatchEncoder(self.batch_size, precision, lg_max_add)
            self.cipher = CPaillier.generate(key_size)
        else:
            self.encoder = encryptor[0]
            self.cipher = encryptor[1]

    @property
    def encryptor(self):
        return self.encoder, self.cipher

    @staticmethod
    def gen_batch_size(log_max_add, bit_length, key_size):
        return BatchEncoder.gen_batch_size(log_max_add, bit_length, key_size)

    def encrypt(self, messages):
        encode_value = self.encoder.encode(messages)
        return self.cipher.encrypt(encode_value)

    def decrypt(self, ciphertext):
        encode_value = self.cipher.decrypt(ciphertext)
        messages = self.encoder.decode(encode_value)
        return messages

    def convert_weight(self, weight):
        weight_vector = [weight] * (2*self.batch_size+1)
        encode_weight = CRT(weight_vector, self.encoder.primes)
        return encode_weight
