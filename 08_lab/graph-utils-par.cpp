/*
 * A template for the 2019 MPI lab at the University of Warsaw.
 * Copyright (C) 2016, Konrad Iwanicki.
 * Refactoring 2019, Łukasz Rączkowski
 */

#include <algorithm>
#include <cassert>
#include <mpi.h>
#include "graph-base.h"
#include "graph-utils.h"

int getFirstGraphRowOfProcess(int numVertices, int numProcesses, int myRank) {
    return myRank * (numVertices / numProcesses) 
          + std::min(myRank, numVertices % numProcesses);
}

Graph* createAndDistributeGraph(int numVertices, int numProcesses, int myRank) {
    assert(numProcesses >= 1 && myRank >= 0 && myRank < numProcesses);

    auto graph = allocateGraphPart(
            numVertices,
            getFirstGraphRowOfProcess(numVertices, numProcesses, myRank),
            getFirstGraphRowOfProcess(numVertices, numProcesses, myRank + 1)
    );

    if (graph == nullptr) {
        return nullptr;
    }

    size_t bufferSize = sizeof(int) * numVertices;

    assert(graph->numVertices > 0 && graph->numVertices == numVertices);
    assert(graph->firstRowIdxIncl >= 0 && graph->lastRowIdxExcl <= graph->numVertices);

    if (myRank == 0) {
        for (int i = graph->firstRowIdxIncl; i < graph->lastRowIdxExcl; i++) {
            initializeGraphRow(graph->data[i], i, graph->numVertices);
        }

        int* buffer;
        buffer = (int*) malloc(bufferSize);

        for (int i = 1; i < numProcesses; i++) {
            int start = getFirstGraphRowOfProcess(numVertices, numProcesses, i);
            int end = getFirstGraphRowOfProcess(numVertices, numProcesses, i+1);
            for (int j = start; j < end; j++) {
                initializeGraphRow(buffer, j, graph->numVertices);
                MPI_Send(buffer, numVertices, MPI_INT, i, j, MPI_COMM_WORLD);
            }
        }

        free(buffer);
    } else {
        for (int i = graph->firstRowIdxIncl; i < graph->lastRowIdxExcl; i++) {
            MPI_Recv(graph->data[i - graph->firstRowIdxIncl], numVertices, MPI_INT, 0, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    return graph;
}

void collectAndPrintGraph(Graph* graph, int numProcesses, int myRank) {
    assert(numProcesses >= 1 && myRank >= 0 && myRank < numProcesses);
    assert(graph->numVertices > 0);
    assert(graph->firstRowIdxIncl >= 0 && graph->lastRowIdxExcl <= graph->numVertices);

    size_t bufferSize = sizeof(int) * graph->numVertices;

    if (myRank == 0) {
        for (int i = graph->firstRowIdxIncl; i < graph->lastRowIdxExcl; i++) {
            printGraphRow(graph->data[i], i, graph->numVertices);
        }

        int* buffer;
        buffer = (int*) malloc(bufferSize);

        for (int i = 1; i < numProcesses; i++) {
            int start = getFirstGraphRowOfProcess(graph->numVertices, numProcesses, i);
            int end = getFirstGraphRowOfProcess(graph->numVertices, numProcesses, i+1);
            for (int j = start; j < end; j++) {
                MPI_Recv(buffer, graph->numVertices, MPI_INT, i, j, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                printGraphRow(buffer, i, graph->numVertices);
            }
        }

        free(buffer);
    } else {
        for (int i = graph->firstRowIdxIncl; i < graph->lastRowIdxExcl; i++) {
            MPI_Send(graph->data[i - graph->firstRowIdxIncl], graph->numVertices, MPI_INT, 0, i, MPI_COMM_WORLD);
        }
    }
}

void destroyGraph(Graph* graph, int numProcesses, int myRank) {
    freeGraphPart(graph);
}
