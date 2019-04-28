#pragma once

#include "errors.h"
#include "memory.h"

template<typename T>
class Queues {
 public:
  Queues(int heapSize, int queuesCount)
    : heapSize(heapSize), queuesCount(queuesCount) { }

  void init(T initQState) {
    HANDLE_ERROR(cudaMalloc(&queuesCuda, sizeof(T) * queuesCount * heapSize));
    HANDLE_ERROR(cudaMalloc(&queueSizesCuda, sizeof(int) * queuesCount));

    HANDLE_ERROR(cudaMemcpy(queuesCuda, &initQState, sizeof(T),
                cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemset(queueSizesCuda, 0, sizeof(int) * queuesCount));

    int one = 1;
    HANDLE_ERROR(cudaMemcpy(queueSizesCuda, &one, sizeof(int),
          cudaMemcpyHostToDevice));
  }

  ~Queues() {
    maybeCudaFree(queuesCuda);
    maybeCudaFree(queueSizesCuda);
  }


  __device__ void push(int queueNumber, T st) {
    assert(queueNumber < queuesCount);
    int& queueSize = queueSizesCuda[queueNumber];
    assert(queueSize < heapSize);

    T* heap = queuesCuda + queueNumber * heapSize;
    heap[queueSize] = st;
    int child = queueSize++;

    while (child > 0 && heap[child].f < heap[(child - 1) / 2].f) {
      swap(heap, child, (child - 1) / 2);
      child = (child - 1) / 2;
    }
  }

  __device__ T top(int queueNumber) {
    assert(queueNumber < queuesCount);
    int& queueSize = queueSizesCuda[queueNumber];
    assert(queueSize > 0);

    T* heap = queuesCuda + queueNumber * heapSize;

    return heap[0];
  }

  __device__ T pop(int queueNumber) {
    assert(queueNumber < queuesCount);
    int& queueSize = queueSizesCuda[queueNumber];
    assert(queueSize > 0);

    T* heap = queuesCuda + queueNumber * heapSize;
    T st = heap[0];

    T last = heap[--queueSize];
    heap[0] = last;

    int k = 0;

    while(2 * k + 1 < queueSize) {
      int leftChild = 2 * k + 1;
      int rightChild = 2 * k + 2;

      if ((rightChild < queueSize && heap[leftChild].f < heap[rightChild].f)
          || rightChild == queueSize) {
        if (leftChild < queueSize && heap[leftChild].f < last.f) {
          swap(heap, k, leftChild);
          k = leftChild;
        } else {
          break;
        }
      } else {
        if (rightChild < queueSize && heap[rightChild].f < last.f) {
          swap(heap, k, rightChild);
          k = rightChild;
        } else {
          break;
        }
      }
    }

    return st;
  }

  __device__ bool empty(int queueNumber) {
    assert(queueNumber < queuesCount);
    return size(queueNumber) == 0;
  }

  __device__ int size(int queueNumber) {
    return queueSizesCuda[queueNumber];
  }

 private:
  T* queuesCuda = nullptr;
  int* queueSizesCuda = nullptr;
  int heapSize;
  int queuesCount;

  __device__ void swap(T* heap, int a, int b) {
    T tmp = heap[a];
    heap[a] = heap[b];
    heap[b] = tmp;
  }
};


