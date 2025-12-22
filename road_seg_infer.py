import os
import matplotlib.pyplot as plt
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3) #dropout 30%
        )

    def forward(self, x):
        return self.conv(x)
    
### MODEL ARCHITECTURE (U-NET) ###

class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()

        #encoder
        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)
        self.pool = nn.MaxPool2d(2)

        #bottleneck
        self.bottleneck = ConvBlock(512, 1024)

        #decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 =ConvBlock(128, 64)

        #output
        self.conv_final = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def forward(self, x):

        #encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        #bottleneck
        b = self.bottleneck(self.pool(e4))

        #decoder with skip connections
        d4 = self.upconv4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        #output
        out = self.conv_final(d1)

        return torch.sigmoid(out) #binary segmentation
    
inference_transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])  
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = inference_transform(image)
    return image, image_tensor.unsqueeze(0)  # add batch dimension

def predict_mask(model, image_tensor, threshold=0.5):
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        output = model(image_tensor)  # (1, 1, H, W)
    
    mask = output[0, 0].cpu().numpy()
    binary_mask = (mask > threshold).astype(np.uint8)
    return binary_mask

def visualize_result(original_image, predicted_mask):
    plt.figure(figsize=(12,4))

    plt.subplot(1,3,1)
    plt.imshow(original_image)
    plt.title("Input Image")
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.imshow(predicted_mask, cmap="gray")
    plt.title("Predicted Mask")
    plt.axis("off")
    
    plt.subplot(1,3,3)
    plt.imshow(original_image)
    plt.imshow(predicted_mask, cmap="Reds", alpha=0.5)
    plt.title("Overlay")
    plt.axis("off")

    plt.show()  

model = UNet(in_channels=3, out_channels=1)
model.load_state_dict(torch.load("best_road_seg_unet.pth", map_location=device))
model.to(device)
model.eval()

def main():
    image_path = "infer_satellite_image.tiff"  # ANY satellite image

    original_image, image_tensor = load_image(image_path)
    predicted_mask = predict_mask(model, image_tensor)

    visualize_result(original_image, predicted_mask)
    mask_img = Image.fromarray(predicted_mask * 255)
    mask_img.save("infer_predicted_road_mask.png")

    return "infer_predicted_road_mask.png"
