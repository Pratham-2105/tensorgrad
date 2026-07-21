#include "makemore.hpp"
#include "serialize.hpp"
#include <memory>
#include <random>

int main() {
  const i32 vocab = 27, block = 3;

  auto C = std::make_shared<Value>(load_matrix("model_names/C.txt"));
  auto W1 = std::make_shared<Value>(load_matrix("model_names/W1.txt"));
  auto b1 = std::make_shared<Value>(load_matrix("model_names/b1.txt"));
  auto W2 = std::make_shared<Value>(load_matrix("model_names/W2.txt"));
  auto b2 = std::make_shared<Value>(load_matrix("model_names/b2.txt"));

  std::mt19937 gen(std::random_device{}());

  std::cout << "--- generated names ---\n";
  for (int i = 0; i < 20; ++i)
    std::cout << generate(C, W1, b1, W2, b2, vocab, block, gen) << "\n";

  return 0;
}
