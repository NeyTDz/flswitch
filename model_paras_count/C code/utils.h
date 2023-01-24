#ifndef _TEST_H_
#define _TEST_H_

#include<iostream>
#include<cstdlib>
#include<vector>
#include<map>
#include<string>
using namespace std;

#define random_int(a,b)    (rand()%(b-a)+a)

typedef  map<int,vector<int> > Stat; //key should be str, but int for efficiency
typedef pair<int,int> Freq;
typedef struct Element
{
    int key = 100;
    int freq = 0;
    vector<int> eles;
}Ele;

vector<int> generate_rand_vector(int len,int down=0,int upper=100){
    vector<int> rand_vec;
    for(int i=0;i<len;i++){
        rand_vec.push_back(random_int(down,upper));
    }
    return rand_vec;
}

template <typename T> void print_vector(vector<T> vec){
    cout << "[";
    for(int i=0;i<vec.size();i++){
        cout << vec[i] << (i==vec.size()-1 ? "]" : ", ");
    }
}

void print_freqs(vector<Freq> freq_list){
    cout << "[";
    for(int i=0;i<freq_list.size();i++){
        cout << '(' << freq_list[i].first << ", "; 
        cout << freq_list[i].second << (i==freq_list.size()-1 ? ")]" : "), "); 
    }
    cout << endl;    
}

//Get the keys of stat dict
vector<int> get_keys(Stat stat){
    vector<int> keys;
    for(Stat::iterator it = stat.begin(); it != stat.end(); it++)
        keys.push_back(it->first);
    return keys;    
}

//Get the freqs of stat dict 
vector<Freq> get_freq(Stat stat, vector<int> keys){
    vector<Freq> freqs;
    for(int i=0;i<keys.size();i++){
        Freq freq;
        freq.first = keys[i];
        freq.second = stat[keys[i]].size();
        freqs.push_back(freq);
    }
    return freqs;
}

void print_stat(Stat stat){
    vector<int> keys = get_keys(stat);
    cout << "{";
    for(int i=0;i<keys.size();i++){
        int key = keys[i];
        cout << "\'" << key << "\': ";
        print_vector(stat[key]);
        cout << (i==keys.size()-1 ? "" : ",");
    }
    cout << "}";
}

#endif