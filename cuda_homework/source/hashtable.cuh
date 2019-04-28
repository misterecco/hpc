#pragma once

#include <cstdio>
#include <limits>
#include <vector>

#include "errors.h"
#include "memory.h"


template<int tableSize, typename State>
class Hashtable {
 public:
  void init() {
    HANDLE_ERROR(cudaMalloc(&hashtableCuda, sizeof(int) * tableSize));

    int* hashtableHost = (int*) HANDLE_NULLPTR(malloc(sizeof(int) * tableSize));
    for (int i = 0; i < tableSize; i++) {
      hashtableHost[i] = -1;
    }

    HANDLE_ERROR(cudaMemcpy(hashtableCuda, hashtableHost, sizeof(int) * tableSize,
          cudaMemcpyHostToDevice));

    free(hashtableHost);
  }

  ~Hashtable() {
    maybeCudaFree(hashtableCuda);
  }

  __device__ void deduplicate(State* states, int stateNumber) {
    State& st = states[stateNumber];
    if (st.isNull()) {
      return;
    }
    const unsigned long long seeds[4] = {100000007u, 350002487u, 700003991u, 12345743u};
    int z = 0;
    int d = 4;

    for (int j = 0; j < d; j++) {
      unsigned int hash = st.hash(seeds[j], tableSize);
      int val = hashtableCuda[hash];
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
    int tInd = atomicExch(hashtableCuda + hash, stateNumber);

    if (tInd != -1 && states[tInd].equals(st) && states[tInd].g <= st.g) {
      st.clear();
      return;
    }

    for (int j = 0; j < d; j++) {
      unsigned int hash = st.hash(seeds[j], tableSize);
      int val = hashtableCuda[hash];
      if (j != z && val != -1 && val != stateNumber && states[val].equals(st) && states[val].g <= st.g) {
        st.clear();
        return;
      }
    }
  }

 private:
  int* hashtableCuda = nullptr;
};


