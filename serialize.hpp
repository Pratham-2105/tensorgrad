#pragma once

#include "matrix.hpp"
#include <cstddef>
#include <fstream>
#include <iomanip>

void save_matrix(const Matrix &m, const std::string &path) {
  std::ofstream ModelFile(path);

  if (!ModelFile) {
    std::cerr << "WeightFile did not open in serealize.hpp";
    return;
  }

  ModelFile << std::setprecision(17);
  ModelFile << m.rows << " " << m.cols << '\n';

  for (auto v : m.matrix) {
    ModelFile << v << " ";
  }

  ModelFile.close();

  return;
}

Matrix load_matrix(const std::string &path) {
  std::ifstream ModelFile(path);
  if (!ModelFile) {
    std::cerr << "load_matrix path failed in serealize.hpp";
    return Matrix(0, 0, false);
  }

  i64 rows, cols;
  ModelFile >> rows >> cols;

  Matrix w(rows, cols, false);

  for (auto &v : w.matrix) {
    ModelFile >> v;
  }

  return w;
}
