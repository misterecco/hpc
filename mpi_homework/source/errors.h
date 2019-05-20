#pragma once

#include <cstdio>
#include <cstdlib>

static void* HandleNullPtr(void* ptr, const char *file, int line) {
  if (ptr == nullptr) {
    printf("malloc failed in %s at line %d\n", file, line);
    exit( EXIT_FAILURE );
  }
  return ptr;
}

#define HANDLE_NULLPTR(ptr) (HandleNullPtr(ptr, __FILE__, __LINE__))

