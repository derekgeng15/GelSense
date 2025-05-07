from  GraspInf.grasp_inf import GraspInf
from GraspInf.gel_dataset import GelDataset
import torch
from torch.utils.data import DataLoader

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Initialize dataset and dataloader
dataset = GelDataset("/n/fs/pvl-franka/dg9272/GelSense/h5")
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Initialize GraspInf model
grasp_inf = GraspInf(pretrained=False).to(device=device)

chkpt = torch.load('/n/fs/pvl-franka/dg9272/GelSense/checkpoints/gel_sense/best_model.pth', map_location=device)
grasp_inf.load_state_dict(chkpt)
# Initialize counters
total_correct = 0
total_samples = 0

# Select a random batch

for batch in dataloader:
    vis_b   = batch['cam_before'].to(device, non_blocking=True)
    vis_a   = batch['cam_during'].to(device, non_blocking=True)
    delta_l = batch['delta_gelA'].to(device, non_blocking=True)
    delta_r = batch['delta_gelB'].to(device, non_blocking=True)
    labels  = batch['is_gripping'].float().to(device, non_blocking=True)


    outputs = grasp_inf(vis_b, vis_a, delta_l, delta_r)
    print(outputs)
    preds          = (outputs >= 0.5).float()
    total_correct += (preds == labels).sum().item()
# inputs = (vis_b, vis_a, delta_l, delta_r)
# outputs = grasp_inf.predict(*inputs)

# Test GraspInf on the batch
# outputs = grasp_inf.predict(inputs)

# Print results
# print("Inputs:", inputs)
print(total_correct)
print(total_samples)
print("Predicted Outputs:", outputs)
print("Ground Truth Labels:", labels)
