import time
import numpy as np
import torch
import onnxruntime as ort
from models import BigCNN, measure_latency_ms, get_file_size_mb

def main():
    model = BigCNN()
    model.load_state_dict(torch.load("baseline.pth"))
    model.eval()

    dummy_input = torch.randn(1, 1, 28, 28)

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        "baseline.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=13
    )

    # Run in PyTorch
    with torch.no_grad():
        pt_out = model(dummy_input).numpy()

    # Run in ONNX Runtime
    session = ort.InferenceSession("baseline.onnx", providers=["CPUExecutionProvider"])
    ort_inputs = {"input": dummy_input.numpy()}
    ort_out = session.run(None, ort_inputs)[0]

    # Validate numerical agreement (< 1e-4)
    max_diff = np.max(np.abs(pt_out - ort_out))
    print(f"Max absolute difference (PyTorch vs ONNX): {max_diff:.2e}")
    assert max_diff < 1e-4, "ONNX output differs too much from PyTorch!"
    print("ONNX Validation Passed (< 1e-4)!")

    # Measure ONNX Runtime latency
    for _ in range(10):  # warm-up
        session.run(None, ort_inputs)
    t0 = time.perf_counter()
    for _ in range(100):
        session.run(None, ort_inputs)
    ort_latency = ((time.perf_counter() - t0) / 100) * 1000.0

    print(f"PyTorch Latency:      {measure_latency_ms(model):.3f} ms/sample")
    print(f"ONNX Runtime Latency: {ort_latency:.3f} ms/sample")
    print(f"ONNX File Size:       {get_file_size_mb('baseline.onnx'):.2f} MB")

if __name__ == "__main__":
    main()