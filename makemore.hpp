#pragma once

#include "matrix.hpp"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

struct Dataset {
  std::vector<std::vector<i32>> X;
  std::vector<i32> Y;
};

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

inline Dataset build_dataset(const std::vector<std::string> &names,
                             i32 block_size) {
  Dataset ds;

  for (const auto &name : names) {
    std::vector<i32> context(block_size, 0);
    std::string s = name + '.';

    for (char ch : s) {
      i32 target = stoi_(ch);

      ds.X.push_back(context);
      ds.Y.push_back(target);

      context.erase(context.begin());
      context.push_back(target);
    }
  }

  return ds;
}
