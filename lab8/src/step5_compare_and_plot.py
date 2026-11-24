import platform
import torch
import torch.nn as nn
import torch.ao.quantization as quant
import matplotlib.pyplot as plt
from models import BigCNN, TinyCNN, get_dataloaders, evaluate_accuracy, measure_latency_ms, get_file_size_mb

def main():
    backend = "qnnpack" if platform.machine() == "arm64" else "fbgemm"
    torch.backends.quantized.engine = backend
    _, testloader = get_dataloaders()

    # Load existing models
    baseline = BigCNN(); baseline.load_state_dict(torch.load("baseline.pth"))
    pruned = BigCNN(); pruned.load_state_dict(torch.load("pruned_90_dense.pth"))
    distilled = TinyCNN(); distilled.load_state_dict(torch.load("student_distilled.pth"))

    # Create combined variants
    dyn_quant = quant.quantize_dynamic(baseline, {nn.Linear}, dtype=torch.qint8)
    pruned_quant = quant.quantize_dynamic(pruned, {nn.Linear}, dtype=torch.qint8)
    distilled_quant = quant.quantize_dynamic(distilled, {nn.Linear}, dtype=torch.qint8)

    torch.save(pruned_quant.state_dict(), "pruned_quant.pth")
    torch.save(distilled_quant.state_dict(), "distilled_quant.pth")

    # Note: Static quant model is evaluated in step2_3_quantization.py; plug its numbers in below if needed
    variants = [
        ("Baseline", baseline, "baseline.pth"),
        ("Pruned (90%)", pruned, "pruned_90_dense.pth"),
        ("Distilled", distilled, "student_distilled.pth"),
        ("Dynamic Quant", dyn_quant, "baseline_dynamic_quant.pth"),
        ("Pruned+Quant", pruned_quant, "pruned_quant.pth"),
        ("Distilled+Quant", distilled_quant, "distilled_quant.pth"),
    ]

    names, accs, lats, sizes = [], [], [], []
    print(f"{'Variant':<18} | {'Acc (%)':<8} | {'Latency (ms)':<13} | {'Size (MB)':<10}")
    print("-" * 58)
    for name, model, path in variants:
        acc = evaluate_accuracy(model, testloader)
        lat = measure_latency_ms(model)
        size = get_file_size_mb(path)
        names.append(name); accs.append(acc); lats.append(lat); sizes.append(size)
        print(f"{name:<18} | {acc:<8.2f} | {lat:<13.3f} | {size:<10.2f}")

    # Plot Accuracy vs Latency and Accuracy vs Size
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.scatter(lats, accs, color="blue")
    for i, txt in enumerate(names):
        ax1.annotate(txt, (lats[i], accs[i]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax1.set_xlabel("Latency (ms/sample)")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Accuracy vs. Latency")
    ax1.grid(True)

    ax2.scatter(sizes, accs, color="green")
    for i, txt in enumerate(names):
        ax2.annotate(txt, (sizes[i], accs[i]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax2.set_xlabel("Size (MB)")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy vs. Model Size")
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("variant_comparison.png")
    print("\nSaved plot to variant_comparison.png")

if __name__ == "__main__":
    main()