import os
import torch
import torch.nn.functional as F
from torchvision.models import inception_v3
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from torchvision.datasets.folder import default_loader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.utils.data import Dataset, DataLoader

# Function to calculate the Inception Score
def inception_score(images, inception_model, batch_size=32, splits=10):
    """
    Calculates the Inception Score.
    :param images: DataLoader containing the images.
    :param inception_model: Pre-trained Inception-v3 model.
    :param batch_size: Batch size.
    :param splits: Number of splits to calculate the variance.
    :return: Mean and standard deviation of the Inception Score.
    """
    # Set the model to eval mode
    inception_model.eval()
    preds = []

    # Compute probabilities with the model
    with torch.no_grad():
        for batch in tqdm(images, desc="Calculating predictions"):
            batch = batch.to(next(inception_model.parameters()).device)
            pred = F.softmax(inception_model(batch), dim=1)
            preds.append(pred.cpu().numpy())

    # Combine all predictions
    preds = np.concatenate(preds, axis=0)

    # Calculate the Inception Score
    split_scores = []
    for k in range(splits):
        part = preds[k * (preds.shape[0] // splits): (k + 1) * (preds.shape[0] // splits), :]
        py = np.mean(part, axis=0)  # Marginal mean
        split_scores.append(np.exp(np.mean(np.sum(part * (np.log(part) - np.log(py + 1e-10)), axis=1))))

    return np.mean(split_scores), np.std(split_scores)

# Settings
image_folder_path = "project/outputs/inference/experiment/actually_ddim/gs1_cgs1p6_200steps/PIL"
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Custom dataset
class FlatImageDataset(Dataset):
    def __init__(self, image_folder_path, transform=None):
        self.image_folder_path = image_folder_path
        self.transform = transform
        self.image_paths = [os.path.join(image_folder_path, img) for img in os.listdir(image_folder_path) if img.endswith(('.jpg', '.png'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = default_loader(image_path)  # Load the image
        if self.transform:
            image = self.transform(image)
        return image

transform = Compose([
    Resize((512, 512)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
dataset = FlatImageDataset(image_folder_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

# Load the pre-trained Inception-v3 model
inception_model = inception_v3(pretrained=True).to(device)

# Calculate the Inception Score
mean_is, std_is = inception_score(dataloader, inception_model, batch_size=batch_size, splits=10)
print(f"Inception Score: {mean_is:.4f} ± {std_is:.4f}")
