import math
import random
from timeit import default_timer as timer
import numpy as np
from rbphe.crt import CRT
from paillier.integer_arithmetic import IntegerArithmetic
from rbphe.primes_table import *
import logging


class ObfuscatedBatchEncoder(object):
    def __init__(self, batch_size=8, bit_length=10, security=80, log_max_add=8, clipping_coeff=8):
        """
        RBPHE encoder, encoding range [-R, R] = [-1, 1]
        :param clipping_coeff: extend encoding range for some large values within 1*clipping_coeff. For example, when
        encoding 1.25, the encoder will generate two values (1.0, 0.25) and add them within the encoding process.
        Simultaneously, the residue of modulo q_0 will be 2 instead of 1 to mark the addition times. Note that encoding
        a number out of [-clipping_coeff, clipping_coeff] is also supported, but the addition time will not be correctly
        marked with the q_0 residue. Therefore, an input value larger than clipping_coeff is not recommended!
        :param batch_size:
        :param security: bit length for random value
        :param lg_max_add: int, bit length for generate approximate allowed maximum add number,
                the maximum add number is usually smaller than pow(2, lg_max_add),
                use ObfuscatedRBPHE.max_add to get the accurate maximum add number
        """
        if bit_length not in [8, 10, 16]:
            raise ValueError("bit_length should be 8, 10, 16")
        if log_max_add > bit_length:
            logging.warning("log_max_add should not be larger than bit_length, "
                            "RBPHE only guarantees correct addition of O(bit_length).")
        self.bit_length = bit_length
        self.security = security
        self.clipping_coeff = clipping_coeff
        if bit_length == 8:
            self._p_primes = obphe_primes_table_p_8[0:batch_size]
            self._q_primes = obphe_primes_table_q_8[0:batch_size]
            self._k_table = obphe_k_table_8[0:batch_size]
        elif bit_length == 10:
            self._p_primes = obphe_primes_table_p_10[0:batch_size]
            self._q_primes = obphe_primes_table_q_10[0:batch_size]
            self._k_table = obphe_k_table_10[0:batch_size]
        elif bit_length == 16:
            self._p_primes = obphe_primes_table_p_16[0:batch_size]
            self._q_primes = obphe_primes_table_q_16[0:batch_size]
            self._k_table = obphe_k_table_16[0:batch_size]
        self._p0 = IntegerArithmetic.getprimeover(security + log_max_add + 1)
        self._q0 = IntegerArithmetic.getprimeover(log_max_add + 1)
        while self._p0 in list(self._p_primes) + list(self._q_primes):
            self._p0 = IntegerArithmetic.getprimeover(security + log_max_add + 1)
        while self._q0 in [self._p0] + list(self._p_primes) + list(self._q_primes):
            self._q0 = IntegerArithmetic.getprimeover(log_max_add + 1)  # use log_max_add+1 as upper bound
        self._modulus_p = 1
        for v in list(self._p_primes) + [self._p0]:
            self._modulus_p *= int(v)
        self._modulus_q = 1
        for v in list(self._q_primes) + [self._q0]:
            self._modulus_q *= int(v)
        self.batch_size = batch_size

    @staticmethod
    def gen_batch_size(log_max_add, bit_length, security, key_size):
        rem = key_size - 1 - 2*(log_max_add+1) - security
        max_rem = pow(2, rem)
        batch_size = 0
        prime_prod = 1
        if bit_length == 8:
            _p_primes = obphe_primes_table_p_8
        elif bit_length == 10:
            _p_primes = obphe_primes_table_p_10
        elif bit_length == 16:
            _p_primes = obphe_primes_table_p_16
        while True:
            prime_prod *= _p_primes[batch_size]
            if prime_prod >= max_rem:
                break
            else:
                batch_size += 1
        return batch_size

    @property
    def max_add(self):
        return min(self._q0, np.min(np.array(self._q_primes) / np.array(self._k_table) - 1))

    @property
    def modulus_p(self):
        return self._modulus_p

    @property
    def modulus_q(self):
        return self._modulus_q

    @property
    def p_primes(self):
        return [self._p0] + list(self._p_primes)

    @property
    def q_primes(self):
        return [self._q0] + list(self._q_primes)

    @staticmethod
    def min_max_scaler(inputs: np.array):
        min_v = np.min(inputs)
        max_v = np.max(inputs)
        return (inputs - min_v) / (max_v - min_v)

    def encode(self, norm_inputs):
        if len(norm_inputs) != self.batch_size:
            logging.warning("length of norm_inputs is not equal to batch size!")
            if len(norm_inputs) < self.batch_size:
                norm_inputs = np.concatenate([norm_inputs, [0]*(self.batch_size - len(norm_inputs))])
            else:
                norm_inputs = norm_inputs[:self.batch_size]
        if np.max(np.abs(norm_inputs)) > self.clipping_coeff:
            logging.warning("An input is larger/smaller than +/-clipping_coeff, check your inputs!")
            norm_inputs = np.clip(norm_inputs, -self.clipping_coeff, self.clipping_coeff)
        coeff = math.ceil(np.max(np.abs(norm_inputs)) / 1.0)
        coeff = min(coeff, self.clipping_coeff)

        r = random.randint(0, self._p0//self._q0 - 1)
        encode_p_values = [r * coeff]
        encode_q_values = [coeff]
        for i, x in enumerate(norm_inputs):
            one_coeff = int(abs(x))
            zero_coeff = coeff - one_coeff - 1
            zero_p = (self._p_primes[i] - 1) // 2
            zero_q = (self._q_primes[i] - 1) // 2
            if x >= 0:
                fraction = x - one_coeff
                one_p = self._p_primes[i] - 1
                one_q = self._q_primes[i] - 1
            else:
                fraction = x + one_coeff
                one_p = 0
                one_q = 0
            encode_p = int(round((fraction + 1) * (self._p_primes[i] - 1) / 2))
            encode_q = int(encode_p * self._k_table[i])
            encode_p = (encode_p + one_p*one_coeff + zero_coeff*zero_p + r*coeff*zero_p) % self._p_primes[i]
            encode_q = (encode_q + one_q*one_coeff + zero_coeff*zero_q + r*coeff*zero_q) % self._q_primes[i]
            encode_p_values.append(encode_p)
            encode_q_values.append(encode_q)
        encode_p_value = CRT(encode_p_values, [self._p0] + list(self._p_primes))
        encode_q_value = CRT(encode_q_values, [self._q0] + list(self._q_primes))
        return encode_p_value, encode_q_value

    def decode(self, encode_p_value, encode_q_value):
        norm_outputs = []
        r = encode_p_value % self._p0
        M = encode_q_value % self._q0
        for i in range(len(self._p_primes)):
            p = self._p_primes[i]
            q = self._q_primes[i]
            k = self._k_table[i]
            zero_p = (p - 1) // 2
            zero_q = (q - 1) // 2
            unit_p = (2 / (p-1))

            residue_p = (encode_p_value - r * zero_p) % p
            residue_q = (encode_q_value - r * zero_q) % q
            if residue_q < residue_p * k:
                residue_q += q
            overflow_time = (residue_q - k * residue_p) / (k-1)
            if int(overflow_time) - overflow_time != 0:
                logging.warning("Detecting overflow, check inputs and addition times!")

            item = (residue_p + overflow_time * p - M * zero_p) * unit_p
            norm_outputs.append(item)
        return np.array(norm_outputs)


if __name__ == "__main__":
    from timeit import default_timer as timer
    x = np.array([9.3, -0.34, -0.11, 0.23])
    y = np.array([0.83, 0.26, -0.97, 0.11])
    w1 = 1
    w2 = 1
    encoder = ObfuscatedBatchEncoder(batch_size=4, bit_length=8, security=128, log_max_add=8)
    t1 = timer()
    encode_p_1, encode_q_1 = encoder.encode(x)
    t2 = timer()
    encode_p_2, encode_q_2 = encoder.encode(y)
    encode_p_3 = encode_p_1*w1 + encode_p_2*w2
    encode_q_3 = encode_q_1*w1 + encode_q_2*w2
    t3 = timer()
    decode_values = encoder.decode(encode_p_3, encode_q_3)
    t4 = timer()
    correct = x*w1 + y*w2
    print("correct: ", correct)
    print("decode: ", decode_values)
