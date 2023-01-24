from paillier.integer_arithmetic import IntegerArithmetic
def get_prime(start, size):
    num = []
    i = start
    count = 0
    while count < size:
        for j in range(2, i):
            if (i % j == 0):
                i+=1
                break
        else:
            num.append(i)
            i+=1
            count += 1
    return num


def is_prime(num):
    for j in range(2, num):
        if (num % j == 0):
            return False
    return True


def genObRBPHE(n=300):
    p_list = []
    q_list = []
    k_list = []
    primes = get_prime(65536, 500)
    mult = list(range(2, 8))
    for p in primes:
        for coeff in mult:
            if is_prime((p-1) * coeff + 1):
                q = (p-1)*coeff + 1
                if (p not in p_list) and (q not in q_list):
                    p_list.append(p)
                    k_list.append(coeff)
                    q_list.append(q)
                    break
        if len(p_list) == 300:
            break
    print(len(p_list))
    print(p_list)
    print(q_list)
    print(k_list)


def genRBPHE(n=300, prec=8):
    p_list = []
    q_list = []
    k_list = []
    primes = get_prime(pow(2, prec), 500)
    mult = list(range(2, 8))
    for p in primes:
        for coeff in mult:
            if is_prime((p-1) * coeff + 1):
                q = (p-1)*coeff + 1
                if (p not in p_list+q_list) and (q not in p_list+q_list):
                    p_list.append(p)
                    k_list.append(coeff)
                    q_list.append(q)
                    break
        if len(p_list) == 300:
            break
    print(len(p_list))
    print(p_list)
    print(q_list)
    print(k_list)

if __name__ == "__main__":
    genRBPHE(prec=16)