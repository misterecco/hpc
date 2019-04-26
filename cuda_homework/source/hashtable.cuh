#pragma once

#include <cstdio>
#include <limits>
#include <vector>


template<typename State>
__device__ void deduplicate(State* states, int* hashtable, int stateNumber) {
  State& st = states[stateNumber];
  if (st.isNull()) {
    return;
  }
  // TODO: store seeds in constant memory
  const unsigned int seeds[4] = {100000007u, 350002487u, 700003991u, 12345743u};
  int z = 0;
  int d = 4;

  for (int j = 0; j < d; j++) {
    unsigned int hash = st.hash(seeds[j]);
    int val = hashtable[hash];
    if (val != -1 && states[val].equals(st) && states[val].g <= st.g) {
      st.clear();
      return;
    }
    if (val == -1 || (states[val].equals(st) && st.g < states[val].g)) {
      z = j;
      break;
    }
  }

  unsigned int hash = st.hash(seeds[z]);
  int tInd = atomicExch(hashtable + hash, stateNumber);

  if (tInd != -1 && states[tInd].equals(st)) {
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

