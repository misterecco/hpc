#include <mpi.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char *argv[]) {
  MPI_Init(&argc,&argv);

  int numProcesses, myRank;
  MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

  if (argc != 3) {
    if (myRank == 0) {
      printf("Usage: %s <try_count> <message_size>\n", argv[0]);
    }
    exit(1);
  }

  int TRY_COUNT = atoi(argv[1]);
  int BUFFER_SIZE = atoi(argv[2]);

  double startTime, endTime, executionTime, sendingTime;

  size_t buffer_size = sizeof(char) * BUFFER_SIZE;
  char* message = malloc(buffer_size);

  if (myRank == 0) {
    startTime = MPI_Wtime();
    MPI_Send(message, BUFFER_SIZE, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
  } else {
    MPI_Recv(message, BUFFER_SIZE, MPI_CHAR, myRank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Send(message, BUFFER_SIZE, MPI_CHAR, (myRank + 1) % numProcesses, 0, MPI_COMM_WORLD);
  }

  if (myRank == 0) {
    MPI_Recv(message, BUFFER_SIZE, MPI_CHAR, numProcesses - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    endTime = MPI_Wtime();
    executionTime = endTime - startTime;
    sendingTime = executionTime / numProcesses;
    printf("%d %d %f\n", TRY_COUNT, BUFFER_SIZE, sendingTime);
  }

  free(message);

  MPI_Finalize(); 
  return 0;
}

