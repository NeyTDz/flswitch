import warnings

from paillier.paillier import *
import numpy as np
import math


class BatchCryptEncoder(object):
    def __init__(self, qt_bit_width=8, clipping=0.5, batch_size=16, padding=3, add_max=2):
        lg_add_max = math.ceil(math.log2(add_max))
        self.bit_width = qt_bit_width + lg_add_max
        self.r_max = clipping * pow(2, lg_add_max)
        self.batch_size = batch_size
        self.padding = padding
        self.add_max = add_max

    def encode(self, values):
        values = np.array(values)
        if len(values) != self.batch_size:
            warnings.warn("length of values is not equal to batch size!")
            if len(values) < self.batch_size:
                #print("waste:",self.batch_size-len(values))
                values = np.concatenate((values, [0]*(self.batch_size - len(values))))
            else:
                values = values[:self.batch_size]
        quantize_values = values * (pow(2, self.bit_width - 1) - 1.0) / self.r_max
        quantize_values = np.rint(quantize_values)
        ecd_values = []
        number = 0
        for val in quantize_values:
            if val < 0:
                ecd_val = int(2 ** (self.bit_width + 1) + val)
            else:
                ecd_val = int(val)
            ecd_values.append(ecd_val)
            number = number * int(pow(2, self.padding + self.bit_width))
            number += ecd_val
        return number

    @staticmethod
    def _two_comp_to_true_(two_comp, bit_width=8, pad_zero=3):
        def two_comp_lit_to_ori(lit, _bit_width):  # convert 2's complement coding of neg value to its original form
            return - 1 * (2 ** (_bit_width - 1) - lit)

        if two_comp < 0:
            raise Exception("Error: not expecting negtive value")
        # two_com_string = bin(two_comp)[2:].zfill(bit_width+pad_zero)

        sign = two_comp >> (bit_width - 1)
        literal = two_comp & (2 ** (bit_width - 1) - 1)

        if sign == 0:  # positive value
            return literal
        elif sign == 4:  # positive value, 0100
            return literal
        elif sign == 1:  # positive overflow, 0001
            return pow(2, bit_width - 1) - 1
        elif sign == 3:  # negtive value, 0011
            return two_comp_lit_to_ori(literal, bit_width)
        elif sign == 7:  # negtive value, 0111
            return two_comp_lit_to_ori(literal, bit_width)
        elif sign == 6:  # negtive overflow, 0110
            print('neg overflow: ' + str(two_comp))
            return - (pow(2, bit_width - 1) - 1)
        else:  # unrecognized overflow
            print('unrecognized overflow: ' + str(two_comp))
            warnings.warn('Overflow detected, consider using longer r_max')
            return - (pow(2, bit_width - 1) - 1)

    def decode(self, encode_value):
        un_batched_nums = np.zeros(self.batch_size, dtype=int)
        for i in range(self.batch_size):
            filter_ = (pow(2, self.bit_width + self.padding) - 1) << ((self.bit_width + self.padding) * i)
            comp = (filter_ & encode_value) >> ((self.bit_width + self.padding) * i)
            un_batched_nums[self.batch_size - 1 - i] = self._two_comp_to_true_(comp, self.bit_width, self.padding)

        un_batched_nums = un_batched_nums.astype(int)
        og_sign = np.sign(un_batched_nums)
        uns_matrix = un_batched_nums * og_sign
        uns_result = uns_matrix * self.r_max / (pow(2, self.bit_width - 1) - 1.0)
        result = og_sign * uns_result
        return result


class BatchCrypt(object):
    def __init__(self, precision=8, clipping=0.5, batch_size="auto", padding=3, add_max=2, key_size=2048,
                 encryptor=None):
        if batch_size == "auto":
            self.batch_size = self.gen_batch_size(add_max, precision, padding, key_size)
        else:
            self.batch_size = batch_size
        self.encoder = BatchCryptEncoder(precision, clipping, self.batch_size, padding, add_max)
        if encryptor:
            self.cipher = encryptor
        else:
            self.cipher = CPaillier.generate(key_size)

    @staticmethod
    def gen_batch_size(add_max, precision, padding, key_size):
        lg_add_max = math.ceil(math.log2(add_max))
        each_bits = lg_add_max + precision + padding
        return (key_size - 1) // each_bits

    def encrypt(self, values):
        encode_val = self.encoder.encode(values)
        ciphertext = self.cipher.encrypt(encode_val)
        return ciphertext

    def decrypt(self, ciphertext):
        plain = self.cipher.decrypt(ciphertext)
        values = self.encoder.decode(plain)
        return values


if __name__ == "__main__":
    from timeit import default_timer as timer
    N = 65536
    encryptor = CPaillier.generate(2048)
    cipher = BatchCrypt(precision=8, add_max=10, clipping=1.0)
    print(cipher.batch_size)
    enc_total = 0
    dec_total = 0
    for _ in range(math.ceil(N/cipher.batch_size)):
        A = np.random.uniform(-0.5, 0.5, cipher.batch_size)
        enc_begin = timer()
        ct = cipher.encrypt(A)
        enc_end = timer()
        dec_begin = timer()
        values = cipher.decrypt(ct)
        dec_end = timer()
        enc_total += (enc_end - enc_begin)
        dec_total += (dec_end - dec_begin)
    print(enc_total)
    print(dec_total)
