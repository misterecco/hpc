#include <iostream>
#include <cstdio>

#include "config.h"
#include "pathfinding.cuh"
#include "slidingpuzzle.cuh"
#include "solver.cuh"

using std::string;

void print_usage(int argc, char** argv) {
    printf("Usage: %s --input-data PATH --output-data PATH --version sliding|pathfinding\n",
        argv[0]);
}

Config parse_args(int argc, char** argv) {
  if (argc != 7) {
    print_usage(argc, argv);
    exit(1);
  }

  Config config;

  for (int i = 1; i <= 6; i+=2) {
    if (!strcmp("--input-data", argv[i])) {
      config.input_data = argv[i+1];
    } else if (!strcmp("--output-data", argv[i])) {
      config.output_data = argv[i+1];
    } else if (!strcmp("--version", argv[i])) {
      if (!strcmp("sliding", argv[i+1])) {
        config.version = Version::SLIDING;
      } else if (!strcmp("pathfinding", argv[i+1])) {
        config.version = Version::PATHFINDING;
      } else {
        printf("Unknown option for --version: %s, valid options: sliding|pathfinding\n", argv[i+1]);
        print_usage(argc, argv);
        exit(1);
      }
    } else {
      printf("Unknown argument: %s\n", argv[i]);
      print_usage(argc, argv);
      exit(1);
    }
  }

  if (config.output_data == "") {
    printf("Missing output data path\n");
    print_usage(argc, argv);
    exit(1);
  } else if (config.input_data == "") {
    printf("Missing input data path\n");
    print_usage(argc, argv);
    exit(1);
  }

  return config;
}

int main(int argc, char** argv) {
  Config config = parse_args(argc, argv);
  /* printf("Config: version: %s, input_data: %s, output_data: %s\n", config.version ==
      Version::SLIDING ? "sliding" : "pathfinding", config.input_data.c_str(),
      config.output_data.c_str()); */
  if (config.version == Version::SLIDING) {
    Solver<SlidingPuzzle, SlidingPuzzle::State, SlidingPuzzle::QState>
      slidingPuzzleSolver(config);
    slidingPuzzleSolver.solve();
  } else {
    Solver<Pathfinding, Pathfinding::State, Pathfinding::QState>
      pathfindingSolver(config);
    pathfindingSolver.solve();
  }

  return 0;
}

