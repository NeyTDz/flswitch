import math
import random
import numpy as np

class PerfectHash:
    def __init__(self, n=20):
        self.n = n
        self.upper = [False for i in range(self.n)]
        self.lower = [[] for i in range(self.n)]
        self.top_level_hash = UniversalHash(self.n)
        self.lower_level_hashes = [UniversalHash(4) for i in range(self.n)]
        self.sum_of_sqs = 0

    def reset_hashes(self):
        self.top_level_hash.reset_hash(self.n)
        self.lower_level_hashes = [UniversalHash() for i in range(self.n)]

    def rehash_top_level(self, new_n = 0):
        self.n = new_n
        new_lower = [[] for i in range(self.n)]
        self.reset_hashes(self.n)
        for i in range(len(self.lower)):
           for j in range(len(self.lower[i])):
               item = self.lower[i]
               top_bucket = self.top_level_hash.get_hashed_value(item.key)
               self.upper[top_bucket] = True
               new_lower[top_bucket].append(item)

        self.lower = [[] for i in range(self.n)]
        self.sum_of_sqs = 0
        for i in range(len(new_lower)):
            self.reconstruct_bottom_level(i, new_lower[i])
            self.sum_of_sqs = len(new_lower[i])**2

        if self.sum_of_sqs > 5*self.n:
            self.reshash_top_level(self.n)

    def reconstruct_bottom_level(self, i, old_items=False):
        if not old_items:
            old_items = [item for item in self.lower[i] if item]
        secondary_hash = self.lower_level_hashes[i]
        previous_size = secondary_hash.n
        secondary_hash.reset_hash(4*previous_size**2)
        new_bucket = [False for i in range(4*previous_size**2)]
        for item in old_items:
            if new_bucket[secondary_hash.get_hashed_value(item.key)]:
                self.reconstruct_bottom_level(i, old_items)
                return
            new_bucket[secondary_hash.get_hashed_value(item.key)] = item
        self.lower[i] = new_bucket

    def find(self, item):
        top_bucket = self.top_level_hash.get_hashed_value(item.key)
        return self.lower[top_bucket][self.lower_level_hashes.get_hashed_value(item.key)]

    def insert(self, item):
        if self.find(item):
            raise "Value is already in the data structure."
        
        top_bucket = self.top_level_hash.get_hashed_value(item.key)
        secondary_bucket = self.lower_level_hashes[top_bucket].get_hashed_value(item.key)
        old_item = self.lower[top_bucket][secondary_bucket]
        self.n += 1
        if old_item:
            self.lower[top_bucket].append(item)
            self.reconstruct_bottom_level(top_bucket)
            new_size = len(self.lower[top_bucket])
            self.sum_of_sqs += 2*new_size-1

            # rehash if we break the constraint that \sum_{i=1}^n s_i^2 = O(n)
            if self.sum_of_sqs > 5*self.n:
                self.rehash_top_level(self.n)
        else:
            self.lower[top_bucket][secondary_bucket] = item
        
    def delete(self, item):
        top_bucket = self.top_level_hash.get_hashed_value(item.key)
        previous_value = self.lower[top_bucket][self.lower_level_hashes.get_hashed_value(item.key)]  
        if previous_value:
            self.lower[top_bucket][self.lower_level_hashes.get_hashed_value(item.key)] = False
            self.n -= 1
        return previous_value

class UniversalHash:
    def __init__(self, n=20):
        self.n = n
        #self.random_seed = random_seed
        #self.init_hash(random_seed)
        self.reset_hash(self.n)
    '''
    def save_hash(self,path):
        hash_dict = {'n':self.n,'p':self.p,'a':self.a,'b':self.b}
        np.save(path+'uni_hash.npy')
    '''

    def get_hashed_value(self, key):
        return (self.a * key + self.b % self.p) % self.n

    def reset_hash(self, size):
        self.n = size
        self.p = self.pick_prime(self.n)
        self.a = random.randrange(1, self.p)
        self.b = random.randrange(1, self.p)

    def pick_prime(self, n):
        primes_list = [True for i in range(2*n+1)]
        primes_list[0] = primes_list[1] = False
    
        for (i, isprime) in enumerate(primes_list):
            if isprime:
                if i > n:
                    return i
                for j in range(i**2, 2*n+1, i):
                    primes_list[j] = False
        return None


if __name__ == '__main__':
    h = UniversalHash(50)
    print(h.get_hashed_value(223420))