#include "matrix.hpp"
#include "mnist.hpp"
#include "value.hpp"
#include <chrono>
#include <memory>

int main() {
  auto start = std::chrono::high_resolution_clock::now();

  std::vector<Matrix> train_images =
      load_images("data/train-images-idx3-ubyte");
  std::vector<Matrix> train_labels =
      load_labels("data/train-labels-idx1-ubyte");

  Matrix W1(128, 784, true);
  Matrix B1(128, 1, false);

  Matrix W2(64, 128, true);
  Matrix B2(64, 1, false);

  Matrix W3(10, 64, true);
  Matrix B3(10, 1, false);

  auto w1 = std::make_shared<Value>(W1);
  auto b1 = std::make_shared<Value>(B1);
  auto w2 = std::make_shared<Value>(W2);
  auto b2 = std::make_shared<Value>(B2);
  auto w3 = std::make_shared<Value>(W3);
  auto b3 = std::make_shared<Value>(B3);

  std::vector<std::shared_ptr<Value>> params = {w1, b1, w2, b2, w3, b3};

  Scalar lr = 0.01;
  i64 epochs = 10;

  size_t N = train_images.size();

  for (i64 epoch = 0; epoch < epochs; ++epoch) {
    Scalar total_loss = 0.0;
    i64 correct = 0;

    for (size_t ex = 0; ex < N; ++ex) {
      Matrix current_image = train_images[ex];
      Matrix current_label = train_labels[ex];

      auto current_img = std::make_shared<Value>(current_image);

      for (auto &p : params) {
        p->grad = Matrix(p->grad.rows, p->grad.cols, false);
      }

      auto h1 = tanh_(add(matmul(w1, current_img), b1));
      auto h2 = tanh_(add(matmul(w2, h1), b2));

      auto logits = add(matmul(w3, h2), b3);

      auto loss = cross_entropy(logits, current_label);
      loss->backward();

      for (auto p : params) {
        p->data = p->data - p->grad.scalar_multiply(lr);
      }

      total_loss += loss->data.at(0, 0);
      if (argmax(logits->data) == argmax(current_label))
        correct++;
    }
    std::cout << "epoch: " << epoch << "   loss: " << total_loss / N
              << "   acc: " << correct * 100.0 / N << "%\n";
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto seconds =
      std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
  std::cout << "training took " << seconds << " seconds\n";

  std::vector<Matrix> test_images = load_images("data/t10k-images-idx3-ubyte");
  std::vector<Matrix> test_labels = load_labels("data/t10k-labels-idx1-ubyte");

  i64 test_correct = 0;
  for (size_t ex = 0; ex < test_images.size(); ++ex) {
    auto x = std::make_shared<Value>(test_images[ex]);
    auto h1 = tanh_(add(matmul(w1, x), b1));
    auto h2 = tanh_(add(matmul(w2, h1), b2));
    auto o = add(matmul(w3, h2), b3); // o is a Value

    if (argmax(o->data) == argmax(test_labels[ex])) // read ->data for argmax
      test_correct++;
  }

  Scalar test_acc = (Scalar)test_correct / test_images.size() * 100;
  std::cout << "TEST accuracy: " << test_acc << "%\n";

  return 0;
}
