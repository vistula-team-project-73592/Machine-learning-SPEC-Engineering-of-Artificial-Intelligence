import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import mlflow

# ==========================================
# 1. LOAD YAML CONFIGURATION
# ==========================================
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

torch.manual_seed(config["seed"])

# ==========================================
# 2. MODEL DEFINITION (Lab 3 Best Model)
# ==========================================
class Lab3Model(nn.Module):
    def __init__(self):
        super().__init__()
        # Replace this with your actual Lab 3 architecture
        self.net = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(28*28, 128), 
            nn.ReLU(), 
            nn.Linear(128, 10)
        )
    def forward(self, x): 
        return self.net(x)

# ==========================================
# 3. CHECKPOINT HELPER FUNCTIONS
# ==========================================
def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)

def load_checkpoint(model, optimizer, path):
    if os.path.exists(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Resuming training from epoch {checkpoint['epoch']}")
        return checkpoint['epoch']
    return 0

# ==========================================
# 4. MAIN TRAINING PIPELINE
# ==========================================
def train():
    # Setup Data (Replace FakeData with your Lab 3 dataset like MNIST/CIFAR)
    transform = transforms.ToTensor()
    dataset = datasets.FakeData(size=1000, image_size=(1, 28, 28), num_classes=10, transform=transform)
    loader = DataLoader(dataset, batch_size=config["batch_size"], num_workers=config["num_workers"])
    
    model = Lab3Model()
    
    # NOTE FOR STEP 5: Change Adam to SGD here when comparing optimizers
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.CrossEntropyLoss()
    
    # Attempt to load an existing checkpoint (Step 7)
    start_epoch = load_checkpoint(model, optimizer, config["checkpoint_path"])

    mlflow.set_experiment("Lab6_Pipeline")
    with mlflow.start_run(run_name=config["run_name"]):
        mlflow.log_params(config) 

        # Setup PyTorch Profiler (Step 2)
        prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/profiler'),
            record_shapes=True
        )
        
        prof.start()
        
        for epoch in range(start_epoch, config["epochs"]):
            model.train()
            optimizer.zero_grad()
            epoch_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(loader):
                output = model(data)
                loss = criterion(output, target)
                
                # Gradient Accumulation (Step 6)
                loss = loss / config["grad_accum_steps"]
                loss.backward()
                
                if (batch_idx + 1) % config["grad_accum_steps"] == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                
                epoch_loss += loss.item() * config["grad_accum_steps"]
                prof.step()
            
            # Save checkpoint at the end of each epoch (Step 1 & 7)
            save_checkpoint(model, optimizer, epoch + 1, config["checkpoint_path"])
            
            # Log metrics to MLflow
            avg_loss = epoch_loss / len(loader)
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            print(f"Epoch {epoch+1}/{config['epochs']} complete. Avg Loss: {avg_loss:.4f}")
            
            # Simulate interruption for Step 7 testing
            if epoch + 1 == 3 and start_epoch == 0:
                print("--- Simulating interruption at Epoch 3! Run train.py again to resume. ---")
                break
                
        prof.stop()

if __name__ == "__main__":
    train()