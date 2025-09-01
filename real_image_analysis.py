import os
import torch
import requests
import numpy as np
from PIL import Image
from pathlib import Path
from random import shuffle
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyyoutube import Api

# ---------- CONFIGURATION ----------
API_KEY = '' 
CHANNEL_NAMES = [
"veritasium", "crashcourse", "MKBHD", "Vox", "GrahamStephan", "jacksepticeye", "chloeting", "BingingWithBabish", "CollegeHumor", "5minutecrafts"
]
THUMBNAIL_FILE = "thumbnails.pt"
IMAGE_SIZE = 64
BATCH_SIZE = 16
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])


# ---------- DATA COLLECTION ----------
class Thumbnail:
    def __init__(self, tensor, label):
        self.tensor = tensor
        self.label = label


def fetch_thumbnails():
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
                img_tensor = transform(img)
                thumbnails.append(Thumbnail(img_tensor, idx))
            except Exception as e:
                print("⚠️ Failed to process thumbnail:", e)

    torch.save(thumbnails, THUMBNAIL_FILE)
    print(f"✅ Saved {len(thumbnails)} thumbnails.")
    return thumbnails


# ---------- DATASET ----------
class ThumbnailDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx].tensor, self.data[idx].label


# ---------- MODEL ----------
class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * (IMAGE_SIZE // 4) * (IMAGE_SIZE // 4), 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)


# ---------- TRAINING ----------
def train_model(model, train_loader, test_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"📚 Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f}")
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

    train_data, test_data = train_test_split(thumbnails, test_size=0.2, random_state=42)
    train_loader = DataLoader(ThumbnailDataset(train_data), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(ThumbnailDataset(test_data), batch_size=BATCH_SIZE)

    model = CNN(num_classes=len(CHANNEL_NAMES)).to(DEVICE)
    train_model(model, train_loader, test_loader)


if __name__ == "__main__":
    main()
