#include "matrix.hpp"
#include "mnist.hpp"
#include "serialize.hpp"
#include "value.hpp"
#include <cstdlib>
#include <ctime>
#include <memory>

int main() {

  auto w1 = std::make_shared<Value>(load_matrix("model/w1.txt"));
  auto b1 = std::make_shared<Value>(load_matrix("model/b1.txt"));
  auto w2 = std::make_shared<Value>(load_matrix("model/w2.txt"));
  auto b2 = std::make_shared<Value>(load_matrix("model/b2.txt"));
  auto w3 = std::make_shared<Value>(load_matrix("model/w3.txt"));
  auto b3 = std::make_shared<Value>(load_matrix("model/b3.txt"));

  std::vector<Matrix> test_images = load_images("data/t10k-images-idx3-ubyte");
  std::vector<Matrix> test_labels = load_labels("data/t10k-labels-idx1-ubyte");

  std::mt19937 gen(std::random_device{}());
  std::uniform_int_distribution<size_t> dist(0, test_images.size() - 1);
  size_t idx = dist(gen);

  auto x = std::make_shared<Value>(test_images[idx]);
  auto h1 = tanh_(add(matmul(w1, x), b1));
  auto h2 = tanh_(add(matmul(w2, h1), b2));
  auto o = add(matmul(w3, h2), b3);

  i64 truth = argmax(test_labels[idx]);
  i64 predicted = argmax(o->data);

  std::cout << "image #" << idx << "\n";
  std::cout << "true digit:      " << truth << "\n";
  std::cout << "predicted digit: " << predicted << "\n";
  std::cout << (truth == predicted ? "correct" : "WRONG") << "\n";

  return 0;
}
