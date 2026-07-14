#include "matrix.hpp"
#include <functional>
#include <memory>
#include <vector>

struct Value {
  Matrix data, grad;

  std::vector<std::shared_ptr<Value>> children;
  std::function<void()> _backward = []() {};

  Value(Matrix data_in) : data(data_in), grad(data_in.rows, data_in.cols, 0) {}

  Value(Matrix data_in, std::vector<std::shared_ptr<Value>> child)
      : data(data_in), grad(data_in.rows, data_in.cols, 0), children(child) {}
};

std::shared_ptr<Value> matmul(std::shared_ptr<Value> a,
                              std::shared_ptr<Value> b) {
  auto out = std::make_shared<Value>(a->data * b->data,
                                     std::vector<std::shared_ptr<Value>>{a, b});

  Value *out_raw = out.get();
  out->_backward = [a, b, out_raw]() {
    a->grad = a->grad + out_raw->grad * b->data.transpose();
    b->grad = b->grad + a->data.transpose() * out_raw->grad;
  };

  return out;
}
