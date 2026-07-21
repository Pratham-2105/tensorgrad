#pragma once

#include "matrix.hpp"
#include "value.hpp"
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
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

inline Matrix one_hot(i32 idx, i32 vocab) {
  Matrix v(vocab, 1, false);
  v.at(idx, 0) = 1.0;

  return v;
}

inline std::shared_ptr<Value> embed_context(std::shared_ptr<Value> C,
                                            const std::vector<i32> &context,
                                            i32 vocab) {
  std::vector<std::shared_ptr<Value>> lookups;

  for (i32 idx : context) {
    auto oh = std::make_shared<Value>(one_hot(idx, vocab));
    lookups.push_back(matmul(C, oh));
  }

  return concat(lookups);
}

inline i32 sample_row(const std::vector<f32> &row, std::mt19937 &gen) {
  std::uniform_real_distribution<f32> dist(0.0, 1.0);
  f32 r = dist(gen);
  f32 cumulative = 0.0;

  for (size_t i = 0; i < row.size(); ++i) {
    cumulative += row[i];

    if (cumulative > r)
      return static_cast<i32>(i);
  }

  return static_cast<i32>(row.size()) - 1;
}

inline std::string generate(std::shared_ptr<Value> C, std::shared_ptr<Value> W1,
                            std::shared_ptr<Value> b1,
                            std::shared_ptr<Value> W2,
                            std::shared_ptr<Value> b2, i32 vocab, i32 block,
                            std::mt19937 &gen) {

  std::string name;
  std::vector<i32> context(block, 0);

  while (true) {
    auto x = embed_context(C, context, vocab);
    auto h = tanh_(add(matmul(W1, x), b1));
    auto logits = add(matmul(W2, h), b2);

    std::vector<f32> probs(vocab);
    f64 maxl = logits->data.at(0, 0);

    f64 sum = 0.0;

    for (i32 i = 0; i < vocab; ++i) {
      probs[i] = std::exp(logits->data.at(i, 0) - maxl);
      sum += probs[i];
    }

    for (i32 i = 0; i < vocab; ++i) {
      probs[i] /= sum;
    }
    i32 next = sample_row(probs, gen);

    if (next == 0)
      break;

    name += itos_(next);
    context.erase(context.begin());
    context.push_back(next);
  }

  return name;
}
