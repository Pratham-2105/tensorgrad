# tensorgrad

A tensor automatic differentiation engine written from scratch in C++. No libraries, no frameworks — just `std::shared_ptr`, a graph of nodes, and the chain rule, except every node holds a whole matrix instead of a single number. It trains a real network on MNIST, saves the trained model to disk, and loads it back to make predictions on demand.

This is the third in a line, and the progression is the whole point:

* CNeural — backpropagation derived by hand and written as fixed code for one specific network. Matrices, but no autograd.
* micrograd — an engine that works out backpropagation automatically, at runtime, for any expression you build — but one scalar per node, far too small to train anything real.
* tensorgrad — the same automatic engine as micrograd, but every node holds a matrix. A matrix multiply becomes a single node with one backward rule instead of thousands of scalar nodes. That's the jump from toy to usable: enough to actually train a model.

CNeural hand-coded the gradients; micrograd automated them for scalars; tensorgrad automates them for tensors — and then uses that to train a real network, with no ML libraries anywhere in the stack.

## The idea

Same core as micrograd — a graph of operations walked backward with the chain rule — with one change that carries all the weight: each `Value` node stores a `Matrix data` and a `Matrix grad` instead of a single number.

The graph machinery is unchanged from micrograd — the `shared_ptr` edges, the topological sort, the reverse-walk backward pass, the accumulation of gradients — because none of it cares what a node holds. What changes is every operation's backward rule, which becomes matrix calculus instead of scalar arithmetic. Those are the same matrix backprop rules I derived by hand in CNeural (BP1–BP4), now attached automatically to each node the moment it's built.

## Results

The engine was validated by training a classifier on MNIST — a 784 → 128 → 64 → 10 network, trained entirely through the automatic backward pass instead of hand-coded gradients. Eight epochs over 60,000 images:

```
epoch: 0   loss: 0.3003   acc: 90.91%
epoch: 4   loss: 0.0578   acc: 98.23%
epoch: 7   loss: 0.0284   acc: 99.14%
training took 399 seconds
TEST accuracy: 96.11%
```

The point wasn't the accuracy — CNeural already reached it by hand. The point is that this time no gradients were derived or written by hand. The engine computed every one of them automatically, and the network still learned to recognize digits. That's what proves the autograd is correct end to end: the same result, but the calculus is now the program's job instead of mine.

## Running it

You need the four MNIST files (train-images, train-labels, t10k-images, t10k-labels) in a `data/` folder. Then to train:

```
make run
```

This trains the network and writes the six weight matrices to `model/`. Once a model is saved, you can make single predictions without retraining:

```
make predict
```

`predict` loads the saved weights, picks a random test image, and prints what the network thinks it is versus the true label. A trained `model/` is included, so `make predict` works straight after cloning — no need to train first.

Only a C++17 compiler is required. Nothing else.

## How it's put together

The code is header-only, each file doing one job:

* **matrix.hpp** — the math, carried over from CNeural. A `Matrix` stored in a flat row-major array, with the operations backprop needs: matrix multiply, add, subtract, transpose, element-wise apply, and Hadamard product. Every operation returns a new matrix instead of modifying the old one.
* **value.hpp** — the engine. A `Value` node holding its data and gradient (both matrices), the child nodes it was built from, and a closure (`_backward`) that pushes gradient to those children. The operations — `matmul`, `add`, `tanh_`, and a fused `cross_entropy` (softmax + cross-entropy loss) — each build a new node, record its children, and attach the right matrix backward rule. `backward()` topologically sorts the graph and walks it in reverse, firing every closure.
* **mnist.hpp** — reading the data, reused from CNeural. Parses the raw binary MNIST files (big-endian headers), loads images as 784×1 columns scaled to 0–1, one-hot encodes the labels, and picks the network's best guess with argmax.
* **serialize.hpp** — saving and loading. `save_matrix` writes a matrix as plain text (its shape, then its values at full precision); `load_matrix` reads it back. That's all a saved model is: six matrices on disk.
* **main.cpp** — the training program. Loads the data, builds the parameters, runs the loop (zero the gradients, forward pass, `backward()`, step every parameter against its gradient), reports loss and accuracy per epoch, then saves the trained weights to `model/`.
* **predict.cpp** — inference. Loads the saved weights, runs a single forward pass on one test image, and reports the prediction. No training, no backward pass — just the trained network answering.

## How the learning works

Every operation on a `Value` builds a node that remembers its inputs and how it was made, so a whole network becomes one big graph — for MNIST that's `cross_entropy(add(matmul(w3, tanh_(add(matmul(w2, tanh_(add(matmul(w1, x), b1))), b2))), b3), target)`, a two-hidden-layer classifier written as a single expression. `backward()` seeds the final loss node's gradient to 1, walks the graph in reverse, and at each node applies the chain rule using that operation's local matrix rule:

| Operation | Backward rule |
| --- | --- |
| matmul | `A.grad += C.grad · Bᵀ`, `B.grad += Aᵀ · C.grad` |
| add | the gradient passes straight through to both inputs |
| tanh | `A.grad += (1 − C²) ⊙ C.grad` |
| softmax + cross-entropy | `logits.grad += softmax(logits) − target` |

Those are CNeural's BP1–BP4, generalized and wired into the graph automatically instead of hand-coded for one fixed network. Training is then just: build the loss as one `Value`, call `backward()` to fill in every parameter's gradient, and step each parameter a little against its gradient. Do that over 60,000 images a few times and the random starting weights turn into something that recognizes digits.

## Notes

I built this one piece at a time, checking each operation on small matrices I could work out by hand before moving on — the matmul backward against a hand-computed gradient, the shared-node case to confirm gradients accumulate, and the whole engine cross-checked against micrograd (the scalar version) on the same tiny expression, since two engines that share no code agreeing is the strongest proof it's right. The rule throughout was to look up how something works when I got stuck, but never to copy a finished solution.

tensorgrad extends the micrograd idea (from Andrej Karpathy's *Neural Networks: Zero to Hero*) from scalars to tensors. The matrix backpropagation comes from my own CNeural.
