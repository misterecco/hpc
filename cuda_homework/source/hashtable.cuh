#pragma once

#include <cstdio>
#include <limits>
#include <vector>


template<typename State, typename Seed>
__device__ void deduplicate(State* states, int* hashtable, 
                            const std::vector<Seed>& seeds, int& stateNumber) {
  State& st = states[stateNumber];
  int z = 0;
  int d = seeds.size();

  for (int j = 0; j < d; j++) {
    unsigned int hash = st.hash(seeds[j]);
    int val = hashtable[hash];
    if (val == -1 || states[val].equals(st)) {
      z = j;
      break;
    }
  }

  State t = st;
  unsigned int hash = st.hash(seeds[z]);
  int tInd = atomicExch(hashtable + hash, stateNumber);

  if (tInd != -1 && states[tInd].equals(st)) {
    st.clear();
    // stateNumber = -1;
    return;
  }

  for (int j = 0; j < d; j++) {
    unsigned int hash = st.hash(seeds[j]);
    int val = hashtable[hash];
    if (j != z && val != -1 && states[val].equals(st)) {
      st.clear();
      //stateNumber = -1;
      return;
    }
  }
}

