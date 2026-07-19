#pragma once

#include "matrix.hpp"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

inline std::vector<std::string> load_names(const std::string &path) {
  std::ifstream file(path);

  if (!file) {
    std::cerr << "file not found in makemore.hpp\n";
    return {};
  }

  std::vector<std::string> names;
  std::string line;

  while (std::getline(file, line)) {
    if (!line.empty() && line.back() == '\r')
      line.pop_back();

    if (!line.empty())
      names.push_back(line);
  }

  return names;
}

inline i32 stoi_(char c) {
  if (c == '.') {
    return 0;
  } else {
    return (c - 'a' + 1);
  }
}

inline char itos_(i32 ix) {
  if (ix == 0) {
    return '.';
  } else {
    return static_cast<char>(ix + 'a' - 1);
  }
}
