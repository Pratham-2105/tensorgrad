#pragma once

#include "matrix.hpp"
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

u32 read_u32_big_endian(std::ifstream &file) {
  unsigned char b[4];

  file.read(reinterpret_cast<char *>(b), 4);

  u32 result =
      (u32(b[0]) << 24) | (u32(b[1]) << 16) | (u32(b[2]) << 8) | u32(b[3]);
  return result;
}

Matrix one_hot(unsigned char label) {
  Matrix label_for_test(10, 1, false);
  int number = (int)label;

  label_for_test.at(number, 0) = 1.0;

  return label_for_test;
}

std::vector<Matrix> load_images(std::string path) {
  std::ifstream data_file(path, std::ios::binary);

  if (!data_file) {
    std::cerr << "can't open data_file in mnist.hpp\n";
    exit(1);
  }

  u32 magic = read_u32_big_endian(data_file);
  u32 count = read_u32_big_endian(data_file);

  if (magic != 2051) {
    std::cerr << "bad image magic\n";
    exit(1);
  }
  std::vector<Matrix> all_digits_in_grid;

  for (u32 c = 0; c < count; ++c) {
    Matrix number_digit_grid(784, 1, false);

    for (u32 i = 0; i < 784; ++i) {
      unsigned char pixel_byte;
      data_file.read(reinterpret_cast<char *>(&pixel_byte), 1);
      number_digit_grid.at(i, 0) = pixel_byte / 255.0;
    }

    all_digits_in_grid.push_back(number_digit_grid);
  }

  return all_digits_in_grid;
}

std::vector<Matrix> load_labels(std::string path) {
  std::ifstream labels_file(path, std::ios::binary);

  if (!labels_file) {
    std::cerr << "can't open labels_file in mnist.hpp\n";
    exit(1);
  }

  u32 magic = read_u32_big_endian(labels_file);
  u32 count = read_u32_big_endian(labels_file);

  if (magic != 2049) {
    std::cerr << "bad image magic\n";
    exit(1);
  }

  std::vector<Matrix> all_labels;

  for (u32 i = 0; i < count; ++i) {
    unsigned char label;
    labels_file.read(reinterpret_cast<char *>(&label), 1);
    all_labels.push_back(one_hot(label));
  }

  return all_labels;
}

i64 argmax(const Matrix &matrix) {
  Scalar best_value = matrix.at(0, 0);
  i64 best_index_row = 0;

  for (i64 i = 0; i < matrix.rows; ++i) {
    Scalar activation = matrix.at(i, 0);
    if (activation > best_value) {
      best_value = activation;
      best_index_row = i;
    }
  }

  return best_index_row;
}
