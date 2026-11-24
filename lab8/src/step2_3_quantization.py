import platform
import torch
import torch.nn as nn
import torch.ao.quantization as quant
from torch.utils.data import Subset, DataLoader
from models import BigCNN, get_dataloaders, evaluate_accuracy, measure_latency_ms, get_file_size_mb

# Static quantization requires QuantStub and DeQuantStub around the network
class QuantizableBigCNN(BigCNN):
    def __init__(self):
        super().__init__()
        self.quant = quant.QuantStub()
        self.dequant = quant.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return self.dequant(x)

def main():
    # Select backend: fbgemm for Intel/AMD, qnnpack for Apple Silicon
    backend = "qnnpack" if platform.machine() == "arm64" else "fbgemm"
    torch.backends.quantized.engine = backend
    print(f"Using quantization backend: {backend}\n")

    trainloader, testloader = get_dataloaders()
    base_model = BigCNN()
    base_model.load_state_dict(torch.load("baseline.pth"))
    base_model.eval()

    # --- STEP 2: Post-Training Dynamic Quantization ---
    dyn_model = quant.quantize_dynamic(base_model, {nn.Linear}, dtype=torch.qint8)
    torch.save(dyn_model.state_dict(), "baseline_dynamic_quant.pth")
    print("--- Step 2: Dynamic Quantization ---")
    print(f"Accuracy: {evaluate_accuracy(dyn_model, testloader):.2f}%")
    print(f"Size:     {get_file_size_mb('baseline_dynamic_quant.pth'):.2f} MB")
    print(f"Latency:  {measure_latency_ms(dyn_model):.3f} ms/sample\n")

    # --- STEP 3: Post-Training Static Quantization ---
    stat_model = QuantizableBigCNN()
    stat_model.load_state_dict(torch.load("baseline.pth"))
    stat_model.eval()
    stat_model.qconfig = quant.get_default_qconfig(backend)
    quant.prepare(stat_model, inplace=True)

    # Calibrate on 500 random training images
    indices = torch.randperm(len(trainloader.dataset))[:500]
    calib_loader = DataLoader(Subset(trainloader.dataset, indices), batch_size=64)
    with torch.no_grad():
        for images, _ in calib_loader:
            stat_model(images)

    quant.convert(stat_model, inplace=True)
    torch.save(stat_model.state_dict(), "baseline_static_quant.pth")
    print("--- Step 3: Static Quantization (500-image calibration) ---")
    print(f"Accuracy: {evaluate_accuracy(stat_model, testloader):.2f}%")
    print(f"Size:     {get_file_size_mb('baseline_static_quant.pth'):.2f} MB")
    print(f"Latency:  {measure_latency_ms(stat_model):.3f} ms/sample")

if __name__ == "__main__":
    main()