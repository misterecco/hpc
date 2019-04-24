#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <algorithm>
#include <vector>

#define HEAP_SIZE 8192

using namespace std;

class Queue {
 public:
  Queue() {
    heap = (int*) malloc(sizeof(int) * HEAP_SIZE);
  }

  ~Queue() {
    free(heap);
  }

  void push(int st) {
    assert(heapSize < HEAP_SIZE);
    
    heap[heapSize] = st;
    int child = heapSize++;

    while (child > 0 && heap[child] < heap[(child - 1) / 2]) {
      swap(child, (child - 1) / 2);
      child = (child - 1) / 2;
    }
  }

  void print() {
    if (heapSize == 0) {
      printf("empty");
    }
    for (int i = 0; i < heapSize; i++) {
      printf("%d ", heap[i]);
    }
    printf("\n");
  }

  int top() {
    assert(heapSize > 0);

    int st = heap[0];

    int last = heap[--heapSize];
    heap[0] = last;

    int k = 0;

    while(2 * k + 1 < heapSize) {
      int leftChild = 2 * k + 1;
      int rightChild = 2 * k + 2;

      if ((rightChild < heapSize && heap[leftChild] < heap[rightChild]) || rightChild == heapSize) {
        if (leftChild < heapSize && heap[leftChild] < last) {
          swap(k, leftChild);
          k = leftChild;
        } else {
          break;
        }
      } else {
        if (rightChild < heapSize && heap[rightChild] < last) {
          swap(k, rightChild);
          k = rightChild;
        } else {
          break;
        }
      }
    }

    return st;
  }

  bool empty() {
    return heapSize == 0;
  }


 private:
  int* heap = nullptr;
  int heapSize = 0;

  void swap(int a, int b) {
    int tmp = heap[a];
    heap[a] = heap[b];
    heap[b] = tmp;
  }
};

void simpleTest() {
  Queue q;

  int ar[10] = {4, 2, 9, 3, 1, 8, 7, 6, 5, 0};

  for (int i : ar) {
    q.push(i);
  }

 
  while (!q.empty()) {
    int i = q.top();
    printf("%d ", i);
  }
  printf("\n");
}

void longerTest() {
  Queue q;

  vector<int> v;

  for (int i = 0; i < 5000; i++) {
    int r = rand();
    v.push_back(r);
    q.push(r);
  }

  sort(v.begin(), v.end());

  for (int i = 0; i < v.size(); i++) {
    assert(v[i] == q.top());
  }
  printf("Long test OK!\n");
}

int main() {
  simpleTest();
  longerTest();
}
