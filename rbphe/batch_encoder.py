import math
import numpy as np
from rbphe.crt import CRT
from paillier.integer_arithmetic import IntegerArithmetic
from rbphe.primes_table import *
import warnings


class BatchEncoder(object):
    def __init__(self, batch_size=8, bit_length=10, log_max_add=10, clipping_coeff=8):
        if bit_length not in [8, 10, 16]:
            raise ValueError("bit_length should be 8, 10, 16")
        if log_max_add > bit_length:
            warnings.warn("log_max_add should not be larger than bit_length")
        self.batch_size = batch_size
        self.bit_length = bit_length
        self.clipping_coeff = clipping_coeff
        if bit_length == 8:
            self._p_primes = primes_table_p_8[0:batch_size]
            self._q_primes = primes_table_q_8[0:batch_size]
            self._k_table = k_table_8[0:batch_size]
        elif bit_length == 10:
            self._p_primes = primes_table_p_10[0:batch_size]
            self._q_primes = primes_table_q_10[0:batch_size]
            self._k_table = k_table_10[0:batch_size]
        elif bit_length == 16:
            self._p_primes = primes_table_p_16[0:batch_size]
            self._q_primes = primes_table_q_16[0:batch_size]
            self._k_table = k_table_16[0:batch_size]
        self._p0 = IntegerArithmetic.getprimeover(log_max_add)
        while self._p0 in list(self._p_primes) + list(self._q_primes):
            self._p0 = IntegerArithmetic.getprimeover(log_max_add)
        self.Q = 1
        for v in list(self._p_primes) + list(self._q_primes) + [self._p0]:
            self.Q *= int(v)

    @staticmethod
    def gen_batch_size(log_max_add, bit_length, key_size):
        rem = key_size - 1 - 2*log_max_add
        max_rem = pow(2, rem)
        batch_size = 0
        prime_prod = 1
        if bit_length == 8:
            _p_primes = primes_table_p_8
            _q_primes = primes_table_q_8
        elif bit_length == 10:
            _p_primes = primes_table_p_10
            _q_primes = primes_table_q_10
        elif bit_length == 16:
            _p_primes = primes_table_p_16
            _q_primes = primes_table_q_16
        while True:
            prime_prod *= _p_primes[batch_size] * _q_primes[batch_size]
            if prime_prod >= max_rem:
                break
            else:
                batch_size += 1
        return batch_size

    @property
    def modulo(self):
        return self.Q

    @property
    def primes(self):
        return list(self._p_primes) + list(self._q_primes) + [self._p0]

    @staticmethod
    def min_max_scaler(inputs: np.array):
        min_v = np.min(inputs)
        max_v = np.max(inputs)
        return (inputs - min_v) / (max_v - min_v)

    def encode(self, norm_inputs):
        if len(norm_inputs) != self.batch_size:
            warnings.warn("length of norm_inputs is not equal to batch size!")
            if len(norm_inputs) < self.batch_size:
                #print("waste:",self.batch_size-len(norm_inputs))
                norm_inputs = np.concatenate([norm_inputs, [0]*(self.batch_size - len(norm_inputs))])
            else:
                norm_inputs = norm_inputs[:self.batch_size]
        encode_p_values = []
        encode_q_values = []
        coeff = math.ceil(np.max(np.abs(norm_inputs)) / 1.0)
        coeff = min(coeff, self.clipping_coeff)

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
            encode_p = (encode_p + one_p*one_coeff + zero_coeff*zero_p) % self._p_primes[i]
            encode_q = (encode_q + one_q*one_coeff + zero_coeff*zero_q) % self._q_primes[i]
            encode_p_values.append(encode_p)
            encode_q_values.append(encode_q)
        total_primes = [self._p0] + list(self._p_primes) + list(self._q_primes)
        encode_values = [coeff] + encode_p_values + encode_q_values
        return CRT(encode_values, total_primes)

    def decode(self, encode_value):
        norm_outputs = []
        M = encode_value % self._p0
        bound = math.ceil(M/2)
        p_low_bound = q_low_bound = -bound
        for i in range(len(self._p_primes)):
            p = self._p_primes[i]
            q = self._q_primes[i]
            k = self._k_table[i]
            residue_p = ((encode_value % p) - M * ((p - 1) / 2)) % p
            residue_q = ((encode_value % q) - M * ((q - 1) / 2)) % q
            if p_low_bound == 0:
                norm_outputs.append((residue_p + p_low_bound * p) / ((p - 1) / 2))
                continue
            # sync index
            while True:
                result_p_0 = (residue_p + p_low_bound * p) / ((p - 1) / 2)
                result_q_0 = (residue_q + q_low_bound * q) / ((q - 1) / 2)
                if result_p_0 - result_q_0 < -1:
                    p_low_bound += 1
                elif result_p_0 - result_q_0 > 1:
                    q_low_bound += 1
                else:
                    break
            uni_diff = abs(2 * (k*p - q)/(q-1))
            index = round(abs(result_p_0 - result_q_0) / uni_diff)
            final_result = (residue_p + (p_low_bound+index) * p) / ((p - 1) / 2)
            norm_outputs.append(final_result)
        return norm_outputs


if __name__ == "__main__":
    x = np.array([0.89, -0.34, -0.11, 0.23])
    y = np.array([0.13, 0.26, -0.97, 0.11])
    w1 = 1
    w2 = 128

    encoder = BatchEncoder(batch_size=4, bit_length=10, log_max_add=10)
    encode_number1 = encoder.encode(x)
    encode_number2 = encoder.encode(y)
    z = w1*encode_number1 + w2*encode_number2
    decode_values = encoder.decode(z)
    correct = x*w1 + y*w2
    print("correct: ", correct)
    print("decode: ", decode_values)