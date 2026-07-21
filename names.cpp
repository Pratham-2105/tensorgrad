#include "makemore.hpp"
#include "serialize.hpp"
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

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

  save_matrix(C->data, "model_names/C.txt");
  save_matrix(W1->data, "model_names/W1.txt");
  save_matrix(b1->data, "model_names/b1.txt");
  save_matrix(W2->data, "model_names/W2.txt");
  save_matrix(b2->data, "model_names/b2.txt");

  std::cout << "saved model to model_names/\n";

  std::cout << "\n--- samples ---\n";
  for (int i = 0; i < 20; ++i)
    std::cout << generate(C, W1, b1, W2, b2, vocab, block, gen) << "\n";

  return 0;
}
