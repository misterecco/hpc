#pragma once

#include <cstdio>
#include <limits>
#include <vector>


template<typename State, typename Seed>
__device__ void deduplicate(State* states, int* hashtable, 
                            Seed* sds, int stateNumber) {
  State& st = states[stateNumber];
  if (st.isNull()) {
    return;
  }
  const Seed seeds[2] = {100000007u, 350002487u};
  int z = 0;
  // TODO: pass seeds somehow - preferably use constant memory
  int d = 2;

  for (int j = 0; j < d; j++) {
    unsigned int hash = st.hash(seeds[j]);
    int val = hashtable[hash];
    if (val == -1 || states[val].equals(st)) {
      z = j;
      break;
    }
  }

  unsigned int hash = st.hash(seeds[z]);
  int tInd = atomicExch(hashtable + hash, stateNumber);

  if (tInd != -1 && states[tInd].equals(st) && states[tInd].g <= st.g) {
    st.clear();
    return;
  }

  for (int j = 0; j < d; j++) {
    unsigned int hash = st.hash(seeds[j]);
    int val = hashtable[hash];
    if (j != z && val != -1 && states[val].equals(st) && states[val].g <= st.g) {
      st.clear();
      return;
    }
  }
}

