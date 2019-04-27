#pragma once

#define HEAP_SIZE 8 * 1024

template<typename T>
__device__ void swap(T* heap, int a, int b) {
  T tmp = heap[a];
  heap[a] = heap[b];
  heap[b] = tmp;
}

template<typename T>
__device__ void push(T* heap, int& heapSize, T st) {
  assert(heapSize < HEAP_SIZE);

  heap[heapSize] = st;
  int child = heapSize++;

  while (child > 0 && heap[child].f < heap[(child - 1) / 2].f) {
    swap(heap, child, (child - 1) / 2);
    child = (child - 1) / 2;
  }
}

template<typename T>
__device__ T top(T* heap, int& heapSize) {
  assert(heapSize > 0);

  return heap[0];
}

template<typename T>
__device__ T pop(T* heap, int& heapSize) {
  assert(heapSize > 0);

  T st = heap[0];

  T last = heap[--heapSize];
  heap[0] = last;

  int k = 0;

  while(2 * k + 1 < heapSize) {
    int leftChild = 2 * k + 1;
    int rightChild = 2 * k + 2;

    if ((rightChild < heapSize && heap[leftChild].f < heap[rightChild].f)
        || rightChild == heapSize) {
      if (leftChild < heapSize && heap[leftChild].f < last.f) {
        swap(heap, k, leftChild);
        k = leftChild;
      } else {
        break;
      }
    } else {
      if (rightChild < heapSize && heap[rightChild].f < last.f) {
        swap(heap, k, rightChild);
        k = rightChild;
      } else {
        break;
      }
    }
  }

  return st;
}

__device__ bool empty(int heapSize) {
  return heapSize == 0;
}

