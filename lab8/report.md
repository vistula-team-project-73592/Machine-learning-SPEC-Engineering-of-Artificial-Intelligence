# Lab 8 Report: Quantization and Export

## Step 1: Manual Quantization Error Analysis
- 8-bit quantization produces very low MSE, 4-bit increases error noticeably, and 2-bit causes severe degradation because only 3 discrete levels (-1, 0, 1) exist.
- **Per-channel scaling** consistently achieves lower MSE than **per-tensor scaling** because each filter gets its own scale factor instead of one outlier channel forcing a large scale on all other channels.

## Step 3: Why Calibration Data Must Represent Real Data
Static quantization observes activation ranges (min/max values) during calibration to compute fixed scales and zero-points. If the 500-image calibration subset does not represent real input data (e.g., missing classes or wrong brightness), the chosen scales will either clip real activations or waste quantization bins, causing a sharp drop in test accuracy.

## Step 5: Variant Comparison & Deployment Selection

| Variant | Accuracy (%) | Latency (ms) | Size (MB) |
| :--- | :--- | :--- | :--- |
| Baseline (FP32) | [Fill] | [Fill] | [Fill] |
| Pruned (90%) | [Fill] | [Fill] | [Fill] |
| Distilled (100k) | [Fill] | [Fill] | [Fill] |
| Dynamic Quant | [Fill] | [Fill] | [Fill] |
| Static Quant | [Fill] | [Fill] | [Fill] |
| Pruned + Quant | [Fill] | [Fill] | [Fill] |
| Distilled + Quant | [Fill] | [Fill] | [Fill] |

### Deployment Choices
- **Cloud:** Baseline (FP32) or ONNX FP32 — abundant compute/memory, so maximum accuracy is the priority.
- **Edge (e.g., Jetson/Raspberry Pi):** Static Quantized Baseline or ONNX — strong balance of high accuracy and fast CPU/NPU execution.
- **Mobile:** Distilled + Quantized — small download size (<1 MB), low latency, and low battery usage.
- **TinyML (Microcontrollers):** Distilled + Static INT8 Quantized — fits inside strict kilobyte-level RAM/flash limits with integer-only math.