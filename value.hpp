#include "matrix.hpp"
#include <algorithm>
#include <cinttypes>
#include <functional>
#include <memory>
#include <numeric>
#include <set>
#include <vector>

struct Value {
  Matrix data, grad;

  std::vector<std::shared_ptr<Value>> children;
  std::function<void()> _backward = []() {};

  Value(Matrix data_in) : data(data_in), grad(data_in.rows, data_in.cols, 0) {}

  Value(Matrix data_in, std::vector<std::shared_ptr<Value>> child)
      : data(data_in), grad(data_in.rows, data_in.cols, 0), children(child) {}

  void build_topo(std::vector<Value *> &topo, std::set<Value *> &visited) {

    if (visited.count(this))
      return;

    visited.insert(this);

    for (auto &child : this->children) {
      child->build_topo(topo, visited);
    }

    topo.push_back(this);
  }

  void backward() {
    std::vector<Value *> topo;
    std::set<Value *> visited;
    build_topo(topo, visited);

    for (auto &v : this->grad.matrix)
      v = 1.0;

    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
      (*it)->_backward();
    }
  }
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

std::shared_ptr<Value> add(std::shared_ptr<Value> a, std::shared_ptr<Value> b) {
  auto out = std::make_shared<Value>(a->data + b->data,
                                     std::vector<std::shared_ptr<Value>>{a, b});

  Value *out_raw = out.get();
  out->_backward = [a, b, out_raw]() {
    a->grad = a->grad + out_raw->grad;
    b->grad = b->grad + out_raw->grad;
  };

  return out;
}

std::shared_ptr<Value> tanh_(std::shared_ptr<Value> a) {
  auto out = std::make_shared<Value>(
      a->data.apply([](double x) { return std::tanh(x); }),
      std::vector<std::shared_ptr<Value>>{a});

  Value *out_raw = out.get();
  out->_backward = [a, out_raw]() {
    a->grad = a->grad + out_raw->data.apply([](double x) { return 1 - x * x; })
                            .hadamard(out_raw->grad);
  };

  return out;
}

std::shared_ptr<Value> cross_entropy(std::shared_ptr<Value> logits,
                                     Matrix target) {
  // find the max_logit and subtract it from all the logits;
  // then .apply() the softmax logic on logits;

  // v[i] = e^(v[i] - max) / sum[v];

  Scalar max_logit = logits->data.matrix[0];

  for (Scalar x : logits->data.matrix) {
    max_logit = std::max(max_logit, x);
  }

  Matrix exp_logits = logits->data.apply(
      [max_logit](Scalar x) { return std::exp(x - max_logit); });

  Scalar sum_of_exp =
      std::accumulate(exp_logits.matrix.begin(), exp_logits.matrix.end(), 0.0);

  Matrix probs =
      exp_logits.apply([sum_of_exp](Scalar x) { return x / sum_of_exp; });

  Scalar loss = 0.0;

  for (size_t i = 0; i < target.matrix.size(); ++i) {
    if (target.matrix[i] == 1.0) {
      loss = -std::log(probs.matrix[i]);
      break;
    }
  }

  Matrix loss_mat(1, 1, false);
  loss_mat.at(0, 0) = loss;

  auto out = std::make_shared<Value>(
      loss_mat, std::vector<std::shared_ptr<Value>>{logits});

  Value *out_raw = out.get();
  out->_backward = [logits, probs, target, out_raw]() {
    logits->grad = logits->grad + (probs - target);
  };

  return out;
}
