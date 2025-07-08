import os
import torch
import torchvision.transforms as transforms
from torchvision.models import inception_v3
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance

# Folder paths
real_folder1 = ".../test"
real_folder2 = ".../validation"
real_folder3 = ".../train"
syn_folder = "/project/outputs/inference/.../PIL"

# Function to load images from a folder
def load_images_from_folder(folder, transform, device):
    images = []
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        try:
            img = Image.open(filepath).convert("RGB")
            img = transform(img).unsqueeze(0).to(device)
            images.append(img)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    return torch.cat(images, dim=0)

# Transformations for model input
transform = transforms.Compose([
    transforms.Resize((512, 512)),  # Required size for Inception-v3
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.Lambda(lambda x: (x * 255).byte()),
])

# Load images
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Rimages1 = load_images_from_folder(real_folder1, transform, device)
Rimages2 = load_images_from_folder(real_folder2, transform, device)
Rimages3 = load_images_from_folder(real_folder3, transform, device)
Simages = load_images_from_folder(syn_folder, transform, device)

# FID calculation
fid_calculator_rr = FrechetInceptionDistance(feature=2048).to(device)
fid_calculator_rs = FrechetInceptionDistance(feature=2048).to(device)

# Add images to FID calculations
fid_calculator_rr.update(Rimages1, real=True)
fid_calculator_rr.update(Rimages2, real=True)
fid_calculator_rs.update(Rimages3, real=True)
fid_calculator_rs.update(Simages, real=False)

# Compute FID value
fid_value_rr = fid_calculator_rr.compute()
fid_value_rs = fid_calculator_rs.compute()
ratio = 1 - ((fid_value_rs.item() - fid_value_rr.item()) / fid_value_rs.item())
print(f"FID Score: {ratio:.4f}")
