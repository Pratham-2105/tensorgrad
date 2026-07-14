#include "value.hpp"
#include <iostream>

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

  return 0;
}
