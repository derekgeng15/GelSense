import torch
import torchvision.transforms as T
import torchvision.io as io  # Import torchvision.io for reading images
from GraspInf.grasp_inf import GraspInf, GraspInf_GelOnly
from PIL import Image
import numpy as np
import json 
import os
import random
from tqdm import tqdm
from torchvision.io import read_video

def normalize_ims(im):
    return -1. + (2./255) * im


def grab_init_frame(mp4_path: str) -> np.ndarray:
    """
    Extracts the first frame from a given mp4 video file and returns
    it as an H×W×3 uint8 numpy array in RGB order.
    """
    # read_video returns T×H×W×3 in RGB uint8
    video, _, _ = read_video(mp4_path,
                             start_pts=0.0,
                             end_pts=0.04,
                             pts_unit="sec")
    if video.size(0) == 0:
        raise ValueError(f"Unable to read first frame from {mp4_path}")

    # grab the first frame, move to CPU & convert to numpy
    frame = video[0]    # shape H×W×3, dtype uint8
    return frame

data = '/n/fs/pvl-franka/ViTac/InfiniViTac/real_dataset/valid_objs'

device = "cuda" if torch.cuda.is_available() else "cpu"
model = GraspInf(pretrained=False)
model_gel = GraspInf_GelOnly(pretrained=False)

chkpt = torch.load('/n/fs/pvl-franka/dg9272/GelSense/checkpoints/gel_sense/best_model.pth', map_location=device)
chkpt_gel = torch.load('/n/fs/pvl-franka/dg9272/GelSense/checkpoints/gel_sense_gel_only/best_model.pth', map_location=device)

model.load_state_dict(chkpt)
model_gel.load_state_dict(chkpt_gel)
model.to(device)
model_gel.to(device)
model.eval()
model_gel.eval()

real_path = '/n/fs/pvl-franka/dg9272/GelSense/InfiniViTac/real_dataset/in_the_wild_data/frames'

success = '/n/fs/pvl-franka/dg9272/GelSense/InfiniViTac/real_dataset/in_the_wild_data/frames/success'
failure = '/n/fs/pvl-franka/dg9272/GelSense/InfiniViTac/real_dataset/in_the_wild_data/frames/failure'
transform = T.Compose([
    T.Resize((224, 224)),      # Resizes the C×H×W tensor
    # T.Lambda(lambda x: x * 255)  # Scale back to [0, 255]
])
# success frames
res_gel = []
res = []
def eval(path, label):
    data_path = os.path.join(path, label)
    for sample in tqdm(os.listdir(data_path)):
        if "DS" in sample:
            continue
        sample_path = os.path.join(data_path, sample)
        vid_path = os.path.join(os.path.join('/n/fs/pvl-franka/dg9272/GelSense/videos/', label), sample)
        
        # Check if all paths exist
        if not os.path.exists(os.path.join(vid_path, "Eye_in_hand.mp4")):
            print(f"Missing file: {os.path.join(vid_path, 'Eye_in_hand.mp4')}")
            continue
        if not os.path.exists(os.path.join(vid_path, "Gelsight_left.mp4")):
            print(f"Missing file: {os.path.join(vid_path, 'Gelsight_left.mp4')}")
            continue
        if not os.path.exists(os.path.join(vid_path, "Gelsight_right.mp4")):
            print(f"Missing file: {os.path.join(vid_path, 'Gelsight_right.mp4')}")
            continue
        if not os.path.exists(os.path.join(sample_path, "Eye_in_hand.png")):
            print(f"Missing file: {os.path.join(sample_path, 'Eye_in_hand.png')}")
            continue
        if not os.path.exists(os.path.join(sample_path, "Gelsight_left.png")):
            print(f"Missing file: {os.path.join(sample_path, 'Gelsight_left.png')}")
            continue
        if not os.path.exists(os.path.join(sample_path, "Gelsight_right.png")):
            print(f"Missing file: {os.path.join(sample_path, 'Gelsight_right.png')}")
            continue

        cam_before = grab_init_frame(os.path.join(vid_path, "Eye_in_hand.mp4")).permute(2, 0, 1)
        lGel_before = grab_init_frame(os.path.join(vid_path, "Gelsight_left.mp4")).permute(2, 0, 1)
        rGel_before = grab_init_frame(os.path.join(vid_path, "Gelsight_right.mp4")).permute(2, 0, 1)
        
        # Use torchvision.io.read_image to read images in RGB format
        cam_during = io.read_image(os.path.join(sample_path, "Eye_in_hand.png")).permute(1, 2, 0).permute(2, 0, 1)      # uint8 [0–255]
        lGel_during = io.read_image(os.path.join(sample_path, "Gelsight_left.png")).permute(1, 2, 0).permute(2, 0, 1) 
        rGel_during = io.read_image(os.path.join(sample_path, "Gelsight_right.png")).permute(1, 2, 0).permute(2, 0, 1)
        
        
        output_dir = "output_images"
        
        # Convert BGR to RGB as OpenCV loads images in BGR format by default
        # cam_during = cv2.cvtColor(cam_during, cv2.COLOR_BGR2RGB)
        # lGel_during = cv2.cvtColor(lGel_during, cv2.COLOR_BGR2RGB)
        # rGel_during = cv2.cvtColor(rGel_during, cv2.COLOR_BGR2RGB)

        # Ensure the images are in the range [0, 255] as numpy arrays
        # cam_during = cam_during.astype(np.float16)
        # lGel_during = lGel_during.astype(np.float16)
        # rGel_during = rGel_during.astype(np.float16)

        # print(f"Shape before transform:")
        # print(f"cam_before: {cam_before.shape}")
        # print(f"cam_during: {cam_during.shape}")
        # print(f"gelA_before: {lGel_before.shape}")
        # print(f"gelA_during: {lGel_during.shape}")
        # print(f"gelB_before: {rGel_before.shape}")
        # print(f"gelB_during: {rGel_during.shape}")

        cam_before = transform(cam_before)
        cam_during = transform(cam_during)
        gelA_before = transform(lGel_before)
        gelA_during = transform(lGel_during)
        gelB_before = transform(rGel_before)
        gelB_during = transform(rGel_during)

        # Print min and max after transform
        # print("After transform:")
        # print(f"cam_before - min: {cam_before.min()}, max: {cam_before.max()}")
        # print(f"cam_during - min: {cam_during.min()}, max: {cam_during.max()}")
        # print(f"gelA_before - min: {gelA_before.min()}, max: {gelA_before.max()}")
        # print(f"gelA_during - min: {gelA_during.min()}, max: {gelA_during.max()}")
        # print(f"gelB_before - min: {gelB_before.min()}, max: {gelB_before.max()}")
        # print(f"gelB_during - min: {gelB_during.max()}, max: {gelB_during.max()}")

        # print(f"Shape after transform:")

        # Print shapes for debugging
        # print(f"cam_before shape: {cam_before.shape}")
        # print(f"cam_during shape: {cam_during.shape}")
        # print(f"gelA_before shape: {gelA_before.shape}")
        # print(f"gelA_during shape: {gelA_during.shape}")
        # print(f"gelB_before shape: {gelB_before.shape}")
        # print(f"gelB_during shape: {gelB_during.shape}")
        
        
        # concat before and during images
        cam_combined = torch.cat((cam_before, cam_during), dim=0)
        gelA_combined = torch.cat((gelA_before, gelA_during), dim=0)
        gelB_combined = torch.cat((gelB_before, gelB_during), dim=0)
        
        # randomly sample 224 x 224 crop
        # _, h, w = cam_combined.shape
        # top = random.randint(0, h - 224)
        # left = random.randint(0, w - 224)
        # cam_combined = cam_combined[:, top:top + 224, left:left + 224]
        # gelA_combined = gelA_combined[:, top:top + 224, left:left + 224]
        # gelB_combined = gelB_combined[:, top:top + 224, left:left + 224]
        
        # Separate gel combined back into before and during
        gelA_before = gelA_combined[:3, :, :]
        gelA_during = gelA_combined[3: :, :,]
        gelB_before = gelB_combined[:3, :, :,]
        gelB_during = gelB_combined[3:, :, :]
        
        # compute delta img
        dIA = gelA_during - gelA_before
        dIB = gelB_during - gelB_before
        
        # Separate cam_combined back into before and during
        cam_before = cam_combined[:3, :, :]
        cam_during = cam_combined[3:, :, :]
        
        # Save gelA, gelB, and cam images as PNGs
        output_dir = "output_images"
        os.makedirs(output_dir, exist_ok=True)

        # Save gelA images
        images = [
            (cam_before, "camera_before.png"),
            (cam_during, "camera_during.png"),
            (gelA_before, "gelA_before.png"),
            (gelA_during, "gelA_during.png"),
            (gelB_before, "gelB_before.png"),
            (gelB_during, "gelB_during.png"),
        ]

        for img, filename in images:
            # Convert tensor to numpy array and ensure it has a valid shape (H×W×C)
            # Save the image using PIL
            # print(img.shape)
            io.write_png(img, os.path.join(output_dir, filename))
    
    
        # Normalize the images
        # Print min and max before normalization
        print("Before normalization:")
        print(f"dIA - min: {dIA.min()}, max: {dIA.max()}")
        print(f"dIB - min: {dIB.min()}, max: {dIB.max()}")
        print(f"cam_combined - min: {cam_combined.min()}, max: {cam_combined.max()}")

        # Normalize the images
        dIA = normalize_ims(dIA)
        dIB = normalize_ims(dIB)
        cam_combined = normalize_ims(cam_combined)

        # Print min and max after normalization
        print("After normalization:")
        print(f"dIA - min: {dIA.min()}, max: {dIA.max()}")
        print(f"dIB - min: {dIB.min()}, max: {dIB.max()}")
        print(f"cam_combined - min: {cam_combined.min()}, max: {cam_combined.max()}")

        cam_before = cam_combined[:3, :, :]
        cam_during = cam_combined[3:, :, :]
        
        # Add batch dimension to the tensors
        # print(f"cam_before shape: {cam_before.shape}")
        # print(f"cam_during shape: {cam_during.shape}")
        # print(f"dIA shape: {dIA.shape}")
        # print(f"dIB shape: {dIB.shape}")
        
        cam_before = cam_before[None, :, :, :].float().to(device).permute(0, 2, 3, 1)
        cam_during = cam_during[None, :, :, :].float().to(device).permute(0, 2, 3, 1)
        dIA = dIA[None, :, :, :].float().to(device).permute(0, 2, 3, 1)
        dIB = dIB[None, :, :, :].float().to(device).permute(0, 2, 3, 1)
        model_output = model(cam_before, cam_during, dIB, dIA)
        model_gel_output = model_gel(dIB, dIA)
        print(f"Model output: {model_output}")
        print(f"Model Gel output: {model_gel_output}")
        if label == "success":
            res.append((model_output >= 0.5).float())
            res_gel.append((model_gel_output >= 0.5).float())
        else:
            res.append((model_output <= 0.5).float())
            res_gel.append((model_gel_output <= 0.5).float())
        # return
eval(real_path, "success")
print(f"Accuracy: {sum(res) / len(res)}")
print(f"Accuracy: {sum(res_gel) / len(res_gel)}")
eval(real_path, "failure")

print(f"Accuracy: {sum(res) / len(res)}")
print(f"Accuracy: {sum(res_gel) / len(res_gel)}")