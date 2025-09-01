import os
import torch
import requests
import numpy as np
from PIL import Image
from pathlib import Path
from random import shuffle
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyyoutube import Api

# ---------- CONFIGURATION ----------
API_KEY = ''  
CHANNEL_NAMES = [
    "veritasium", "crashcourse", "MKBHD", "Vox", "GrahamStephan",
    "jacksepticeye", "chloeting", "BingingWithBabish", "CollegeHumor", "5minutecrafts"
]
THUMBNAIL_FILE = "thumbnails.pt"

# original small-CNN image size
IMAGE_SIZE = 64

# ResNet settings
USE_RESNET = True               # <-- set True to use ResNet transfer learning
RESNET_NAME = "resnet18"        # resnet18 is used; can change to resnet50 (requires more memory)
RESNET_INPUT_SIZE = 224         # pretrained ResNets expect ~224x224
RESNET_FREEZE = True            # if True, freeze backbone and train final fc only

BATCH_SIZE = 16
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms
# small CNN transform (keeps IMAGE_SIZE)
base_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

# ResNet / ImageNet-like transform
resnet_transform = transforms.Compose([
    transforms.Resize((RESNET_INPUT_SIZE, RESNET_INPUT_SIZE)),
    transforms.ToTensor(),
    # ImageNet mean/std (important for pretrained models)
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ---------- DATA COLLECTION ----------
class Thumbnail:
    """Stores a PIL image and a label index (picklable)."""
    def __init__(self, pil_image, label):
        self.image = pil_image
        self.label = label

def fetch_thumbnails():
    """
    Downloads thumbnails and caches them as PIL Images with labels.
    This allows switching transform pipelines later without re-downloading.
    """
    if Path(THUMBNAIL_FILE).exists():
        print("✔️ Using cached thumbnails.")
        return torch.load(THUMBNAIL_FILE)

    print("📥 Downloading thumbnails from YouTube...")
    api = Api(api_key=API_KEY)
    thumbnails = []
    label_map = {}
    for idx, name in enumerate(CHANNEL_NAMES):
        print(f"🔍 Searching for channel: {name}")
        response = api.search_by_keywords(q=name, search_type="channel", count=1)
        if not response.items:
            print(f"⚠️ Channel {name} not found; skipping.")
            continue
        channel_id = response.items[0].id.channelId
        label_map[channel_id] = idx
        channel_info = api.get_channel_info(channel_id=channel_id)
        uploads_playlist = channel_info.items[0].to_dict()['contentDetails']['relatedPlaylists']['uploads']
        playlist_items = api.get_playlist_items(playlist_id=uploads_playlist, count=400)
        for item in playlist_items.items:
            try:
                video_id = item.snippet.resourceId.videoId
                video = api.get_video_by_id(video_id=video_id).items[0].to_dict()
                thumbnail_url = video['snippet']['thumbnails']['medium']['url']
                img = Image.open(requests.get(thumbnail_url, stream=True).raw).convert("RGB")
                thumbnails.append(Thumbnail(img, idx))
            except Exception as e:
                print("⚠️ Failed to process thumbnail:", e)

    torch.save(thumbnails, THUMBNAIL_FILE)
    print(f"✅ Saved {len(thumbnails)} thumbnails.")
    return thumbnails

# ---------- DATASET ----------
class ThumbnailDataset(Dataset):
    def __init__(self, data, transform):
        """
        data: list of Thumbnail (with .image PIL and .label)
        transform: torchvision transform to be applied on-the-fly
        """
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pil_img = self.data[idx].image
        label = self.data[idx].label
        img_tensor = self.transform(pil_img)
        return img_tensor, label

# ---------- MODELS ----------
class SmallCNN(nn.Module):
    def __init__(self, num_classes, image_size=IMAGE_SIZE):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * (image_size // 4) * (image_size // 4), 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)

def build_resnet(num_classes, resnet_name="resnet18", freeze_backbone=True):
    """
    Loads a pretrained ResNet, replaces final fc layer with num_classes output.
    freeze_backbone: if True, set requires_grad=False for backbone params.
    """
    if resnet_name == "resnet18":
        model = models.resnet18(pretrained=True)
    elif resnet_name == "resnet50":
        model = models.resnet50(pretrained=True)
    else:
        raise ValueError("Unsupported resnet_name")

    # freeze backbone if requested
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, num_classes)   # replace final fc
    return model

# ---------- TRAINING ----------
def train_model(model, train_loader, test_loader, optimizer=None, criterion=None, epochs=EPOCHS):
    if optimizer is None:
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    model.to(DEVICE)
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / max(1, len(train_loader))
        print(f"📚 Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
        evaluate(model, test_loader)

def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            out = model(x)
            preds = out.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
    print(classification_report(all_labels, all_preds, target_names=CHANNEL_NAMES))

# ---------- MAIN ----------
def main():
    thumbnails = fetch_thumbnails()
    shuffle(thumbnails)

    # choose transform and model based on USE_RESNET
    if USE_RESNET:
        transform = resnet_transform
        model = build_resnet(num_classes=len(CHANNEL_NAMES),
                             resnet_name=RESNET_NAME,
                             freeze_backbone=RESNET_FREEZE)
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.Adam(trainable_params, lr=1e-4)
    else:
        transform = base_transform
        model = SmallCNN(num_classes=len(CHANNEL_NAMES), image_size=IMAGE_SIZE)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # split and dataloaders (transform applied by dataset)
    train_data, test_data = train_test_split(thumbnails, test_size=0.2, random_state=42)
    train_loader = DataLoader(ThumbnailDataset(train_data, transform), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(ThumbnailDataset(test_data, transform), batch_size=BATCH_SIZE)

    model = model.to(DEVICE)
    train_model(model, train_loader, test_loader, optimizer=optimizer, epochs=EPOCHS)

if __name__ == "__main__":
    main()
