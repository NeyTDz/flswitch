#include<iostream>
#include<vector>
#include<map>
#include<algorithm>
#include"utils.h"
#include"topk.cpp"
using namespace std;

//typedef  map<string,vector<int>> Stat;
//typedef pair<string,int> Freq;


Stat get_stat(vector<int> nums){
    /*
    Return statistics of elements in a list
    e.g.
    List: ['A','A','B','C','C','C']
    stat: {'A': [0, 1], 'B': [2], 'C': [3, 4, 5]}
    */   
    Stat stat;
    for(int i=0;i<nums.size();i++){
        stat[nums[i]].push_back(i);
    }
    return stat;
}

pair<vector<Freq>,vector<Element> > top_process(Stat stat,int k){
    /*
    Return k elements with highest frequency
    e.g.
    stat: {'A': [0, 1], 'B': [2], 'C': [3, 4, 5]}
    k: 2
    frequents:  [('C',3),('A',2)]
    elements: [('C',3,[3,4,5]),('A',2,[0,1])]
    Notice: 
    if k < |stat_keys|, set k_in = min(k,stat_keys)
    topk <=> sort at that time
    */

    
    vector<int> stat_keys = get_keys(stat);
    int len_keys = stat_keys.size();
    vector<Freq> stat_count = get_freq(stat,stat_keys);
    int k_in = min(k,len_keys);
    vector<Freq> frequents = top_frequent(stat_count,k_in,"heap");
    vector<Element> elements;
    for(int i=0;i<frequents.size();i++){
        Element element;
        int key = frequents[i].first;
        int freq = frequents[i].second;
        element.key = key;
        element.freq = freq;
        element.eles = stat[key];
        elements.push_back(element);
    }
    //vector<Freq> frequents;
    return make_pair(frequents,elements);
}




int main(){
    int k = 5;
    Stat stat;
    //vector<int>nums = {{100},{100},{90},{75},{75},{75}};
    vector<int>nums = generate_rand_vector(3044170,0,10);
    //print_vector(nums);
    cout << endl;

    stat = get_stat(nums);
    //print_stat(stat);
    cout << endl;

    pair<vector<Freq>,vector<Element> > Fre_Ele;
    Fre_Ele = top_process(stat,k);
    print_freqs(Fre_Ele.first);
    return 0;
}

