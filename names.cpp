#include "makemore.hpp"
#include "value.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

Matrix one_hot(i32 idx, i32 vocab) {
  Matrix v(vocab, 1, false);
  v.at(idx, 0) = 1.0;
  return v;
}

std::shared_ptr<Value> embed_context(std::shared_ptr<Value> C,
                                     const std::vector<i32> &context,
                                     i32 vocab) {
  std::vector<std::shared_ptr<Value>> lookups;
  for (i32 idx : context) {
    auto oh = std::make_shared<Value>(one_hot(idx, vocab));
    lookups.push_back(matmul(C, oh));
  }
  return concat(lookups);
}

i32 sample_row(const std::vector<f32> &row, std::mt19937 &gen) {
  std::uniform_real_distribution<f32> dist(0.0, 1.0);
  f32 r = dist(gen);
  f32 cumulative = 0.0;
  for (size_t j = 0; j < row.size(); ++j) {
    cumulative += row[j];
    if (cumulative > r)
      return static_cast<i32>(j);
  }
  return static_cast<i32>(row.size()) - 1;
}

std::string generate(std::shared_ptr<Value> C, std::shared_ptr<Value> W1,
                     std::shared_ptr<Value> b1, std::shared_ptr<Value> W2,
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
    for (i32 i = 0; i < vocab; ++i)
      maxl = std::max(maxl, logits->data.at(i, 0));

    f64 sum = 0.0;
    for (i32 i = 0; i < vocab; ++i) {
      probs[i] = std::exp(logits->data.at(i, 0) - maxl);
      sum += probs[i];
    }
    for (i32 i = 0; i < vocab; ++i)
      probs[i] /= sum;

    i32 next = sample_row(probs, gen);

    if (next == 0)
      break;

    name += itos_(next);
    context.erase(context.begin());
    context.push_back(next);
  }

  return name;
}

int main() {
  auto names = load_names("data/names.txt");
  Dataset ds = build_dataset(names, 3);

  const i32 vocab = 27, embed_dim = 10, block = 3;
  const i32 n_in = embed_dim * block;
  const i32 H = 100;

  auto C = std::make_shared<Value>(Matrix(embed_dim, vocab, true));
  auto W1 = std::make_shared<Value>(Matrix(H, n_in, true));
  auto b1 = std::make_shared<Value>(Matrix(H, 1, false));
  auto W2 = std::make_shared<Value>(Matrix(vocab, H, true));
  auto b2 = std::make_shared<Value>(Matrix(vocab, 1, false));

  std::vector<std::shared_ptr<Value>> params = {C, W1, b1, W2, b2};

  const f64 lr = 0.01;
  const i32 epochs = 10;
  const size_t N = ds.X.size();

  std::mt19937 gen(std::random_device{}());
  std::vector<size_t> order(N);
  std::iota(order.begin(), order.end(), 0);

  for (i32 epoch = 0; epoch < epochs; ++epoch) {
    auto t0 = std::chrono::high_resolution_clock::now();
    f64 running = 0.0;

    std::shuffle(order.begin(), order.end(), gen);

    for (size_t k = 0; k < N; ++k) {
      size_t j = order[k];

      for (auto &p : params)
        zero_grad(p);

      auto x = embed_context(C, ds.X[j], vocab);
      auto h = tanh_(add(matmul(W1, x), b1));
      auto logits = add(matmul(W2, h), b2);
      Matrix target = one_hot(ds.Y[j], vocab);
      auto loss = cross_entropy(logits, target);

      running += loss->data.at(0, 0);

      loss->backward();

      for (auto &p : params)
        p->data = p->data - (p->grad.scalar_multiply(lr));
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    f64 secs = std::chrono::duration<f64>(t1 - t0).count();
    std::cout << "epoch " << epoch << "  avg loss " << running / N << "  ("
              << secs << "s)\n";
  }

  std::cout << "\n--- samples ---\n";
  for (int i = 0; i < 20; ++i)
    std::cout << generate(C, W1, b1, W2, b2, vocab, block, gen) << "\n";

  return 0;
}
