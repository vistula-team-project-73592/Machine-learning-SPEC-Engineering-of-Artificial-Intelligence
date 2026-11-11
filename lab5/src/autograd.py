import math
import random
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_moons

class Value:
    def __init__(self, data, _children=()):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other))

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other))

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers"
        out = Value(self.data ** other, (self,))

        def _backward():
            self.grad += (other * (self.data ** (other - 1))) * out.grad
        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return Value(other) + (-self)

    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, (self,))

        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + math.exp(-self.data))
        out = Value(s, (self,))

        def _backward():
            self.grad += (s * (1.0 - s)) * out.grad
        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data + 1e-15), (self,))

        def _backward():
            self.grad += (1.0 / (self.data + 1e-15)) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

print("--- Running Step 4: Verification ---")
a_val, b_val, c_val = 1.5, -2.0, 3.5

a = Value(a_val)
b = Value(b_val)
c = Value(c_val)
out_custom = ((a * b + c) ** 2).relu()
out_custom.backward()

a_pt = torch.tensor(a_val, requires_grad=True, dtype=torch.float64)
b_pt = torch.tensor(b_val, requires_grad=True, dtype=torch.float64)
c_pt = torch.tensor(c_val, requires_grad=True, dtype=torch.float64)
out_pt = ((a_pt * b_pt + c_pt) ** 2).relu()
out_pt.backward()

eps = 1e-7
def f(av, bv, cv):
    return max(0.0, (av * bv + cv) ** 2)

grad_a_num = (f(a_val + eps, b_val, c_val) - f(a_val - eps, b_val, c_val)) / (2 * eps)
grad_b_num = (f(a_val, b_val + eps, c_val) - f(a_val, b_val - eps, c_val)) / (2 * eps)
grad_c_num = (f(a_val, b_val, c_val + eps) - f(a_val, b_val, c_val - eps)) / (2 * eps)

diff_pt = max(abs(a.grad - a_pt.grad.item()), abs(b.grad - b_pt.grad.item()), abs(c.grad - c_pt.grad.item()))
diff_num = max(abs(a.grad - grad_a_num), abs(b.grad - grad_b_num), abs(c.grad - grad_c_num))

print(f"Max diff vs PyTorch: {diff_pt:.2e}")
print(f"Max diff vs Numerical: {diff_num:.2e}")
assert diff_pt < 1e-6 and diff_num < 1e-6, "Gradients do not match!"
print("Verification Passed successfully!\n")

print("--- Running Step 5: Training ---")
X_np, y_np = make_moons(n_samples=100, noise=0.1, random_state=42)

class CustomMLP:
    def __init__(self):
        self.w1 = [[Value(random.uniform(-0.5, 0.5)) for _ in range(2)] for _ in range(8)]
        self.b1 = [Value(0.0) for _ in range(8)]
        self.w2 = [Value(random.uniform(-0.5, 0.5)) for _ in range(8)]
        self.b2 = Value(0.0)

    def parameters(self):
        params = []
        for row in self.w1: params.extend(row)
        params.extend(self.b1)
        params.extend(self.w2)
        params.append(self.b2)
        return params

    def forward(self, x):
        hidden = []
        for i in range(8):
            act = self.w1[i][0] * x[0] + self.w1[i][1] * x[1] + self.b1[i]
            hidden.append(act.relu())
        out = self.b2
        for i in range(8):
            out = out + self.w2[i] * hidden[i]
        return out.sigmoid()

def train_custom():
    model = CustomMLP()
    lr = 0.1
    t0 = time.perf_counter()
    for epoch in range(15):
        total_loss = Value(0.0)
        for xi, yi in zip(X_np, y_np):
            pred = model.forward([Value(xi[0]), Value(xi[1])])
            loss = -(Value(yi) * pred.log() + Value(1 - yi) * (Value(1.0) - pred).log())
            total_loss = total_loss + loss

        for p in model.parameters():
            p.grad = 0.0
        total_loss.backward()

        for p in model.parameters():
            p.data -= lr * p.grad
    return time.perf_counter() - t0

class PyTorchMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8), nn.ReLU(),
            nn.Linear(8, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

def train_pytorch():
    model = PyTorchMLP()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.BCELoss()
    X_t = torch.tensor(X_np, dtype=torch.float32)
    y_t = torch.tensor(y_np, dtype=torch.float32).unsqueeze(1)

    t0 = time.perf_counter()
    for epoch in range(15):
        optimizer.zero_grad()
        preds = model(X_t)
        loss = criterion(preds, y_t)
        loss.backward()
        optimizer.step()
    return time.perf_counter() - t0

custom_times = [train_custom() for _ in range(5)]
pytorch_times = [train_pytorch() for _ in range(5)]

print(f"Custom Engine Median Time: {np.median(custom_times):.4f} seconds")
print(f"PyTorch Median Time:       {np.median(pytorch_times):.4f} seconds")