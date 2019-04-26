#pragma once

#include <cstdio>
#include <limits>
#include <vector>


template<typename State>
__device__ void deduplicate(State* states, int* hashtable, int stateNumber, int tableSize) {
  State& st = states[stateNumber];
  if (st.isNull()) {
    return;
  }
  // TODO: store seeds in constant memory
  const unsigned long long seeds[4] = {100000007u, 350002487u, 700003991u, 12345743u};
  int z = 0;
  int d = 4;

  for (int j = 0; j < d; j++) {
    unsigned int hash = st.hash(seeds[j], tableSize);
    int val = hashtable[hash];
    if (val != -1 && states[val].equals(st) && states[val].g <= st.g) {
      st.clear();
      return;
    }
    if (val == -1 || (states[val].equals(st) && st.g < states[val].g)) { z = j;
      z = j;
      break;
    }
  }

  unsigned int hash = st.hash(seeds[z], tableSize);
  int tInd = atomicExch(hashtable + hash, stateNumber);

  if (tInd != -1 && states[tInd].equals(st) && states[tInd].g <= st.g) {
    st.clear();
    return;
  }

  for (int j = 0; j < d; j++) {
    unsigned int hash = st.hash(seeds[j], tableSize);
    int val = hashtable[hash];
    if (j != z && val != -1 && val != stateNumber && states[val].equals(st) && states[val].g <= st.g) {
      st.clear();
      return;
    }
  }
}

