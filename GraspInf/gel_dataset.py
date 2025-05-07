import os
import h5py
import torch.distributed
from torch.utils.data import Dataset
import torchvision.transforms as T
import numpy as np
import torch
import random
from PIL import Image
    
def crop_kinect(im):
    # the bounds of the table, plus some padding above for tall objects
    bounds = np.array([[ 0.28602991, 0.07516428],
                        [ 0.76788474, 0.06441159],
                        [ 0.97554603, 0.59487754],
                        [ 0.13482023, 0.60563023]])
    d = np.array([im.shape[0], im.shape[1]])
    x0, y0 = map(int, bounds.min(0) * d)
    x1, y1 = map(int, bounds.max(0) * d)
    return im[x0 : x1, y0 : y1]
def normalize_ims(im):
    return -1. + (2./255) * im
class GelDataset(Dataset):
    def __init__(self, path, swmr: bool = False):
        """
        Args:
            path: directory containing .h5 files
            swmr: whether to open each file in SWMR mode (if your files were written with SWMR)
        """
        self.path = path
        # 1) find all .h5 files
        self.files = sorted([
            os.path.join(path, fn)
            for fn in os.listdir(path)
            if fn.endswith('.h5')
        ])
        # 2) build a flat list of (file_idx, group_name)
        self.index = []
        for i, fp in enumerate(self.files):
            with h5py.File(fp, 'r') as f:
                for grp_name in f['data'].keys():
                    self.index.append((i, grp_name))

        # 3) placeholder for one open handle _per_ file
        self._handles = [None] * len(self.files)
        self._swmr    = swmr
        # if torch.distributed.get_rank() == 0:
        #     print(f"loaded {len(self.index)} samples")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        file_idx, grp_name = self.index[idx]
        # Open & cache the file handle on first access
        if self._handles[file_idx] is None:
            self._handles[file_idx] = h5py.File(
                self.files[file_idx],
                'r',
                libver='latest',
                swmr=self._swmr
            )
        f = self._handles[file_idx]
        grp = f['data'][grp_name]
        
        cam_before = grp['kinectA_rgb_before'][()]
        cam_during = grp['kinectA_rgb_during'][()]
        
        gelA_before = grp['gelsightA_before'][()]
        gelA_during = grp['gelsightA_during'][()]
        
        gelB_before = grp['gelsightB_before'][()]
        gelB_during = grp['gelsightB_during'][()]
        
        # crop table
        cam_before = crop_kinect(cam_before)
        cam_during = crop_kinect(cam_during)
                
        # resize + sample random image 
        transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Lambda(lambda x: x * 255)  # Scale back to [0, 255]
        ])
        cam_before = transform(cam_before).permute(1, 2, 0)
        cam_during = transform(cam_during).permute(1, 2, 0)
        gelA_before = transform(gelA_before).permute(1, 2, 0)
        gelA_during = transform(gelA_during).permute(1, 2, 0)
        gelB_before = transform(gelB_before).permute(1, 2, 0)
        gelB_during = transform(gelB_during).permute(1, 2, 0)
        # Write out all images for visualization as PNG files

        
        # concat before and during images
        cam_combined = torch.cat((cam_before, cam_during), dim=2)
        gelA_combined = torch.cat((gelA_before, gelA_during), dim=2)
        gelB_combined = torch.cat((gelB_before, gelB_during), dim=2)
        
        # randomly sample 224 x 224 crop
        h, w, _ = cam_combined.shape
        top = random.randint(0, h - 224)
        left = random.randint(0, w - 224)
        cam_combined = cam_combined[top:top + 224, left:left + 224, :]
        gelA_combined = gelA_combined[top:top + 224, left:left + 224, :]
        gelB_combined = gelB_combined[top:top + 224, left:left + 224, :]
        
        
        # Apply random horizontal flip
        if random.random() > 0.5:
            cam_combined = torch.flip(cam_combined, dims=[1])
            gelA_combined = torch.flip(gelA_combined, dims=[1])
            gelB_combined = torch.flip(gelB_combined, dims=[1])
            
        # random vertical flip across gels
        if random.random() > 0.5:
            gelA_combined = torch.flip(gelA_combined, dims=[0])
            gelB_combined = torch.flip(gelB_combined, dims=[0])
            
        # Separate gel combined back into before and during
        gelA_before = gelA_combined[:, :, :3]
        gelA_during = gelA_combined[:, :, 3:]
        gelB_before = gelB_combined[:, :, :3]
        gelB_during = gelB_combined[:, :, 3:]
        
        # compute delta img
        dIA = gelA_during - gelA_before
        dIB = gelB_during - gelB_before
        
        
        # Visualize and write out dIA, dIB, cam_combined using PIL

        # Convert tensors to numpy arrays for saving
        # Separate cam_combined back into before and during
        cam_before = cam_combined[:, :, :3]
        cam_during = cam_combined[:, :, 3:]

        # # Convert tensors to numpy arrays for saving
        images = [
            (cam_before, "camera_before.png"),
            (cam_during, "camera_during.png"),
            (gelA_before, "gelA_before.png"),
            (gelA_during, "gelA_during.png"),
            (gelB_before, "gelB_before.png"),
            (gelB_during, "gelB_during.png"),
        ]

        # Concatenate all images horizontally
        all_images = torch.cat([img.permute(2, 0, 1) for img, _ in images], dim=2).permute(1, 2, 0)
        
        # Convert tensor to numpy array and scale to [0, 255]
        all_images_np = (all_images.numpy()).astype(np.uint8)
        
        # Save the concatenated image using PIL with the index as the filename
        Image.fromarray(all_images_np).save(f"concatenated_{idx}.png")
        # Normalize the images
        # Print min, max values before normalization
        # print("Before normalization:")
        # print(f"dIA min: {dIA.min()}, max: {dIA.max()}")
        # print(f"dIB min: {dIB.min()}, max: {dIB.max()}")
        # print(f"cam_combined min: {cam_combined.min()}, max: {cam_combined.max()}")

        # Normalize the images
        dIA = normalize_ims(dIA)
        dIB = normalize_ims(dIB)
        cam_combined = normalize_ims(cam_combined)

        # Print min, max values after normalization
        # print("After normalization:")
        # print(f"dIA min: {dIA.min()}, max: {dIA.max()}")
        # print(f"dIB min: {dIB.min()}, max: {dIB.max()}")
        # print(f"cam_combined min: {cam_combined.min()}, max: {cam_combined.max()}")
        
        cam_before = cam_combined[:, :, :3]
        cam_during = cam_combined[:, :, 3:]
        
        # Now read exactly what you need into memory
        return {
            'is_gripping':        int(grp.attrs['is_gripping']),
            'cam_before':         cam_before,
            'cam_during':         cam_during,
            'delta_gelA':         dIA,
            'delta_gelB':         dIB
        }

    
if __name__ == "__main__":
    dataset = GelDataset('/n/fs/pvl-franka/dg9272/GelSense/h5')
    print(len(dataset))
    for i in range(4):
        idx = random.randint(0, len(dataset)- 1)
        import time 
        startime = time.time()
        dataset[idx]
        print(time.time() - startime)
    