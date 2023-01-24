import numpy as np
import collections
from collections import defaultdict,Counter

def top_frequent(stat_count:list,k:int,algo="heap") -> list:
    if algo == "sort":
        return BaseSort(stat_count,k)
    elif algo == "heap":
        return HeapFrequent(stat_count,k)
    elif algo == "bucket":
        assert 0 #Ban bucket
        return BucketFrequent(stat_count,k)
    else:
        print("Algorithm not exists!")
        assert 0

################ Base Sort ################
def BaseSort(stat_count:list,k:int) -> list:
    '''
    Get top k frequent elements in stat_count via base sorting algorithm
    Args:
      stat_count(list): the frequent of elements in a list
      k(int): top k frequent
    Return:
      the top k frequent of stat(sorted)
    e.g.
      stat_count: [('A',2),('B',1),('C',3)]
      k: 2
      return [('C',3),('A',2)]
    '''
    stat_sorted = sorted(stat_count,key = lambda x:x[1],reverse=True)
    result = stat_sorted[:k]
    return result

############## Heap Frequent ##############
# basic operation
def lchild(node):
    return node << 1
def rchild(node):
    return node << 1 | 1
def father(node):
    return node >> 1
# rise up a node to insert a new node
def heap_up(heap,node):
    val = heap[node]
    while father(node)>0 and val[1]<heap[father(node)][1]:
        heap[node] = heap[father(node)]
        node = father(node)
    heap[node] = val
# sink down a node to adjust the heap
def heap_down(heap,node,k):
    root = node
    val = heap[node]
    while lchild(root) <= k: #check lchild
        child = lchild(root)
        if child|1 <= k and heap[child|1][1] < heap[child][1]: #check rchild
            child = child | 1
        if heap[child][1] < val[1]:
            heap[root] = heap[child]
            root = child
        else:
            break
    heap[root] = val

def HeapFrequent(stat_count:list,k:int) -> list:
    '''
    Get top k frequent elements in stat_count via min-heap algorithm
    Args,Return same in BaseSort()
    '''
    heap = [(0,0)] #up-bound as place holder
    # build the heap
    for i in range(k):
        heap.append(stat_count[i])
        heap_up(heap, len(heap)-1)
    # adjust the heap
    for i in range(k,len(stat_count)):
        if stat_count[i][1] > heap[1][1]:
            heap[1] = stat_count[i]
            heap_down(heap,1,k) 
    # heap sort
    result = sorted(heap, key=lambda x: x[1],reverse=True)[:-1]
    return result

############# Bucket Frequent #############
def BucketFrequent(stat_count:list,k:int,spacesize=1000) -> list:
    '''
    Get top k frequent elements in stat_count via bucket algorithm
    Args,Return same in BaseSort()
    '''    
    Bucket = [[] for i in range(spacesize)]
    for s in stat_count:
        Bucket[s[1]].append(s)
    r = 0
    flag = False
    result = []
    for i in range(len(Bucket)-1,-1,-1):
        if not flag and Bucket[i] != []:
            for s in Bucket[i]:
                result.append(s)
                r += 1
                if r == k:
                    flag = True
                    break
    return result


