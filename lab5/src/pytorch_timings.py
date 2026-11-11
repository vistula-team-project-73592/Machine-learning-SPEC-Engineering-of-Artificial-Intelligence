import torch
import time

print("--- Part A: Eager vs Traced Execution ---")

model = torch.nn.Sequential(
    torch.nn.Linear(256, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 128)
)
x = torch.randn(1024, 256)

traced_model = torch.jit.trace(model, x)

for _ in range(10):
    _ = model(x)
    _ = traced_model(x)

t0 = time.perf_counter()
for _ in range(100):
    _ = model(x)
eager_time = time.perf_counter() - t0

t0 = time.perf_counter()
for _ in range(100):
    _ = traced_model(x)
traced_time = time.perf_counter() - t0

print(f"Eager Execution Time:  {eager_time:.5f} seconds")
print(f"Traced Execution Time: {traced_time:.5f} seconds\n")

print("--- Part B: Tensor Contiguity ---")

tensor_contig = torch.randn(5000, 5000)

print("Original Tensor:")
print(f"  Shape: {tensor_contig.shape}")
print(f"  Dtype: {tensor_contig.dtype}")
print(f"  Is contiguous? {tensor_contig.is_contiguous()}")

tensor_non_contig = tensor_contig.t()

print("\nTransposed Tensor View:")
print(f"  Shape: {tensor_non_contig.shape}")
print(f"  Dtype: {tensor_non_contig.dtype}")
print(f"  Is contiguous? {tensor_non_contig.is_contiguous()}\n")

t0 = time.perf_counter()
for _ in range(100):
    _ = tensor_contig.sum(dim=1)
time_contig = time.perf_counter() - t0

t0 = time.perf_counter()
for _ in range(100):
    _ = tensor_non_contig.sum(dim=1)
time_non_contig = time.perf_counter() - t0

print(f"Contiguous Sum Time:     {time_contig:.5f} seconds")
print(f"Non-Contiguous Sum Time: {time_non_contig:.5f} seconds")