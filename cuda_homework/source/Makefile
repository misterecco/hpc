CC := nvcc
CCFLAGS := --gpu-architecture=compute_60 --gpu-code=sm_60 -std=c++14 -rdc=true --compiler-options="-Wall -O3" -O3

all: astar_gpu

pathfinding.o: pathfinding.cu pathfinding.cuh errors.h memory.h
	$(CC) $(CCFLAGS) -c $@ $<

slidingpuzzle.o: slidingpuzzle.cu slidingpuzzle.cuh errors.h
	$(CC) $(CCFLAGS) -c $@ $<

astar_gpu: astar_gpu.cu pathfinding.o slidingpuzzle.o solver.cuh hashtable.cuh queues.cuh memory.h
	$(CC) $(CCFLAGS) -o $@ $< pathfinding.o slidingpuzzle.o

heap_cpu: heap_cpu.cpp
	clang++ -o $@ $^ --std=c++11

clean:
	rm astar_gpu pathfinding.o slidingpuzzle.o
