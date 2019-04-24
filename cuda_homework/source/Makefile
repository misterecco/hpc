CUDA_SOURCES := $(shell find . -name '*.cu')
HEADERS := $(shell find . -name '*.h')
CC := nvcc

all: astar_gpu heap_cpu

pathfinding.o: pathfinding.cu pathfinding.cuh heap.cuh hashtable.cuh
	$(CC) -c $@ $<

astar_gpu: astar_gpu.cu pathfinding.o
	$(CC) -o $@ $^

heap_cpu: heap_cpu.cpp
	clang++ -o $@ $^ --std=c++11

clean:
	rm astar_gpu heap_cpu pathfinding.o
