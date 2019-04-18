#pragma once
#include <string>

using std::string;

enum class Version { SLIDING, PATHFINDING };

struct Config {
  Version version;
  string input_data;
  string output_data;
};
