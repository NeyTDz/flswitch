
#include<iostream>
#include<vector>
#include<algorithm>
#include<map>
#include<string>
#include<cassert>
#include"utils.h"
using namespace std;

//############## Heap Frequent ##############
// basic operation
int lchild(int node){
    return node << 1;
}
int rchild(int node){
    return node << 1 | 1;
}
int father(int node){
    return node >> 1;
}
// rise up a node to insert a new node
void heap_up(vector<Freq>& heap, int node){
    Freq val = heap[node];
    while(father(node)>0 && val.second < heap[father(node)].second){
        heap[node] = heap[father(node)];
        node = father(node);
    heap[node] = val;
    }
}
// sink down a node to adjust the heap
void heap_down(vector<Freq>& heap,int node,int k){
    int root = node;
    Freq val = heap[node];
    while(lchild(root) <= k){
        int child = lchild(root);
        if(child|1 <= k && heap[child|1].second < heap[child].second){
            child = child | 1;
        }
        if(heap[child].second < val.second){
            heap[root] = heap[child];
            root = child;
        }
        else
            break; 
    }
    heap[root] = val;
}
// heap sort
bool cmp(Freq a,Freq b){
    return a.second > b.second;
}

vector<Freq> HeapFrequent(vector<Freq> stat_count,int k){
    /*
    Get top k frequent elements in stat_count via min-heap algorithm
    Args,Return same in BaseSort()
    */
    Freq init(100,0);
    vector<Freq> heap;
    heap.push_back(init); // up-bound as place holder
    
    // build the heap
    for(int i=0;i<k;i++){
        heap.push_back(stat_count[i]);
        heap_up(heap, heap.size()-1);
    }
    for(int i=k;i<stat_count.size();i++){
        if(stat_count[i].second > heap[1].second){
            heap[1] = stat_count[i];
            heap_down(heap,1,k);
        }
    }
    // heap sort
    sort(heap.begin(),heap.end(),cmp);
    heap.pop_back();

    return heap;    
}

//All Top Frequent Funcs
vector<Freq> top_frequent(vector<Freq> stat_count,int k,string algo="heap"){
    if(algo == "sort"){
        assert(0);
        //return BaseSort(stat_count,k)
    }
    else if(algo == "heap"){
        return HeapFrequent(stat_count,k);
    }
    else if(algo == "bucket"){
        assert(0); //Ban bucket
        //return BucketFrequent(stat_count,k);
    }
    else{
        cout << "Algorithm not exists!";
        assert(0);
    }
}

/*
int main(){

    int k = 3;
    vector<Freq> heap{{102,2},{103,1},{105,5},{106,1},{107,4},{109,5}};
    print_freqs(heap); 
    cout << "Top " << k << ", sort result:" << endl;
    vector<Freq> sort_result = HeapFrequent(heap,k);
    print_freqs(sort_result);

    return 0;
}
*/

