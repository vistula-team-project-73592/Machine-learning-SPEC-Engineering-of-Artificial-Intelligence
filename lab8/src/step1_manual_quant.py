import torch
from models import BigCNN

def quantize_dequantize(w, num_bits, per_channel=False):
    qmax = (2 ** (num_bits - 1)) - 1  # 127 for 8-bit, 7 for 4-bit, 1 for 2-bit
    
    if per_channel:
        # One scale per output channel (dim 0)
        dims = tuple(range(1, w.ndim))
        scale = w.abs().amax(dim=dims, keepdim=True) / qmax
    else:
        # One scale for the entire tensor
        scale = w.abs().max() / qmax

    scale = torch.clamp(scale, min=1e-8)  # avoid divide-by-zero
    
    # Map to int, clamp, and map back to float
    q = torch.clamp(torch.round(w / scale), -qmax, qmax)
    w_reconstructed = q * scale
    mse = torch.mean((w - w_reconstructed) ** 2).item()
    return mse

def main():
    model = BigCNN()
    model.load_state_dict(torch.load("baseline.pth"))
    w = model.conv2.weight.data  #

    print(f"Weight tensor shape: {tuple(w.shape)}\n")
    print(f"{'Bits':<6} | {'Per-Tensor MSE':<16} | {'Per-Channel MSE':<16}")
    print("-" * 44)
    for bits in [8, 4, 2]:
        mse_tensor = quantize_dequantize(w, bits, per_channel=False)
        mse_channel = quantize_dequantize(w, bits, per_channel=True)
        print(f"{bits:<6} | {mse_tensor:<16.8f} | {mse_channel:<16.8f}")

if __name__ == "__main__":
    main()