CUDA_SOURCES := $(shell find . -name '*.cu')
HEADERS := $(shell find . -name '*.h')
CC := nvcc

all: astar_gpu

pathfinding.o: pathfinding.cu pathfinding.h
	$(CC) -c $@ $<

astar_gpu: astar_gpu.cu pathfinding.o
	$(CC) -o $@ $^
