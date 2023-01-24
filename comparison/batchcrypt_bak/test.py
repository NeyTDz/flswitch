from timeit import default_timer as timer
from comparison.batchcrypt_bak.paillier import PaillierKeypair

if __name__ == "__main__":
    pk, sk = PaillierKeypair.generate_keypair(1024)
    x = 1.2
    precision = 8
    t1 = timer()
    ct = pk.encrypt(x, precision)
    t2 = timer()
    pt = sk.decrypt(ct)
    print(t2-t1)
    print(pt)