#pragma once

#include <stdio.h>

static void HandleError(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    exit( EXIT_FAILURE );
  }
}

static void* HandleNullPtr(void* ptr, const char *file, int line) {
  if (ptr == nullptr) {
    printf("malloc failed in %s at line %d\n", file, line);
    exit( EXIT_FAILURE );
  }
  return ptr;
}

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__ ))
#define HANDLE_NULLPTR(ptr) (HandleNullPtr(ptr, __FILE__, __LINE__))

