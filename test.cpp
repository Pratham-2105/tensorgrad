#include "matrix.hpp"
#include "mnist.hpp"
#include "value.hpp"

int main() {
  /*
    Matrix m(2, 3, false);
    m.at(0, 0) = 1;
    m.at(0, 1) = 2;
    m.at(0, 2) = 3;
    m.at(1, 0) = 4;
    m.at(1, 1) = 5;
    m.at(1, 2) = 6;

    Value v(m);

    std::cout << "data:\n";
    v.data.print_matrix();

    std::cout << "grad:\n";
    v.grad.print_matrix();

    */

  /*
  Matrix A(2, 3, false);
  A.at(0, 0) = 1;
  A.at(0, 1) = 2;
  A.at(0, 2) = 3;
  A.at(1, 0) = 4;
  A.at(1, 1) = 5;
  A.at(1, 2) = 6;

  Matrix B(3, 2, false);
  B.at(0, 0) = 7;
  B.at(0, 1) = 8;
  B.at(1, 0) = 9;
  B.at(1, 1) = 10;
  B.at(2, 0) = 11;
  B.at(2, 1) = 12;

  Matrix C = A * B;

  Matrix c_grad(C.rows, C.cols, false);
  for (auto &v : c_grad.matrix)
    v = 1.0;

  Matrix a_grad = c_grad * (B.transpose());
  Matrix b_grad = (A.transpose()) * c_grad;

  std::cout << "a_grad (want 2x3):\n";
  a_grad.print_matrix();
  std::cout << "b_grad (want 3x2):\n";
  b_grad.print_matrix();

  auto a = std::make_shared<Value>(A);
  auto b = std::make_shared<Value>(B);
  auto c = matmul(a, b);

  for (auto &v : c->grad.matrix)
    v = 1.0; // seed c's grad = ones
  c->_backward();

  std::cout << "a->grad (want 2x3):\n";
  a->grad.print_matrix();
  std::cout << "b->grad (want 3x2):\n";
  b->grad.print_matrix();
  */

  /*
  Matrix A(2, 2, false);
  A.at(0, 0) = 1;
  A.at(0, 1) = 2;
  A.at(1, 0) = 3;
  A.at(1, 1) = 4;
  Matrix B(2, 2, false);
  B.at(0, 0) = 5;
  B.at(0, 1) = 6;
  B.at(1, 0) = 7;
  B.at(1, 1) = 8;

  auto a = std::make_shared<Value>(A);
  auto b = std::make_shared<Value>(B);
  auto c = add(a, b);

  std::cout << "c->data (want 6 8 / 10 12):\n";
  c->data.print_matrix();

  c->grad.at(0, 0) = 1;
  c->grad.at(0, 1) = 2;
  c->grad.at(1, 0) = 3;
  c->grad.at(1, 1) = 4;
  c->_backward();

  std::cout << "a->grad (want 1 2 / 3 4):\n";
  a->grad.print_matrix();
  std::cout << "b->grad (want 1 2 / 3 4):\n";
  b->grad.print_matrix();
  */

  /*
    Matrix A(1, 3, false);
    A.at(0, 0) = 0;
    A.at(0, 1) = 1;
    A.at(0, 2) = -1;

    auto a = std::make_shared<Value>(A);
    auto c = tanh_(a);

    std::cout << "c->data (want 0  0.761594  -0.761594):\n";
    c->data.print_matrix();

    for (auto &v : c->grad.matrix)
      v = 1.0;
    c->_backward();

    std::cout << "a->grad (want 1  0.419974  0.419974):\n";
    a->grad.print_matrix();
  */

  /*
  Matrix A(1, 3, false);
  A.at(0, 0) = 1;
  A.at(0, 1) = 2;
  A.at(0, 2) = 3;

  Matrix W(3, 2, false);
  W.at(0, 0) = 0.1;
  W.at(0, 1) = -0.2;
  W.at(1, 0) = 0.0;
  W.at(1, 1) = 0.1;
  W.at(2, 0) = -0.1;
  W.at(2, 1) = 0.0;

  auto a = std::make_shared<Value>(A);
  auto w = std::make_shared<Value>(W);
  auto c = tanh_(matmul(a, w));

  std::cout << "c->data:\n";
  c->data.print_matrix();
  c->backward();
  std::cout << "a->grad:\n";
  a->grad.print_matrix();
  std::cout << "w->grad:\n";
  w->grad.print_matrix();

  Matrix X(1, 3, false);
  X.at(0, 0) = 0;
  X.at(0, 1) = 1;
  X.at(0, 2) = -1;

  auto x = std::make_shared<Value>(X);
  auto t = tanh_(x);
  auto d = add(t, t);

  std::cout << "d->data:\n";
  d->data.print_matrix();
  d->backward();
  std::cout << "t->grad:\n";
  t->grad.print_matrix();
  std::cout << "x->grad:\n";
  x->;grad.print_matrix()

  */

  /*
  Matrix A(1, 2, false);
  A.at(0, 0) = 3;
  A.at(0, 1) = 5;

  Matrix W(2, 1, false);
  W.at(0, 0) = 2;
  W.at(1, 0) = 4;

  auto a = std::make_shared<Value>(A);
  auto w = std::make_shared<Value>(W);
  auto out = matmul(a, w);
  out->backward();

  std::cout << "a->grad = \n";
  a->grad.print_matrix();
  std::cout << "w->grad = \n";
  w->grad.print_matrix();
  */

  /*
  Matrix logits_data(1, 3, false);
  logits_data.at(0, 0) = 0;
  logits_data.at(0, 1) = 0;
  logits_data.at(0, 2) = 0;

  Matrix target(1, 3, false);
  target.at(0, 0) = 1; // class 0 is the correct one
  target.at(0, 1) = 0;
  target.at(0, 2) = 0;

  auto logits = std::make_shared<Value>(logits_data);
  auto loss = cross_entropy(logits, target);

  std::cout << "loss (want 1.0986):\n";
  loss->data.print_matrix();

  loss->backward();

  std::cout << "logits->grad (want -0.666667  0.333333  0.333333):\n";
  logits->grad.print_matrix();
  */

  return 0;
}
