#!/usr/bin/env python3
"""
Thumbnail CNN classifier for N channels (default NUM_CLASSES=10).
Fill CHANNELS with the 10 channel names you want to classify.
"""

import os
import argparse
from random import shuffle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage import io, exposure

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset
from torch.optim.lr_scheduler import StepLR

from sklearn.metrics import precision_score, recall_score, f1_score

from pyyoutube import Api


# ----- CONFIG: Put the exact 10 channel search names here -----
CHANNELS = [
    "MKBHD",
    "crashcourse",
    "Veritasium",
]
NUM_CLASSES = len(CHANNELS)
NBINS = 25
THUMBNAIL_CACHE = "thumbnails.npz"


# --------------------- YouTube crawler -------------------------
class YouTubeCrawler:
    def __init__(self, names=None):
        self.names = names if names is not None else CHANNELS
        self.channels = []
        api_key = os.getenv("YOUTUBE_API_KEY")
        if not api_key:
            raise RuntimeError("Set YOUTUBE_API_KEY environment variable")
        self.api = Api(api_key=api_key)

    def get_channel_ids(self, names):
        for search_term in names:
            search_response = self.api.search_by_keywords(
                q=search_term, search_type="channel", count=1
            )
            if search_response.items:
                channel_id = search_response.items[0].id.channelId
                print(f"🔎 Found channel ID for '{search_term}': {channel_id}")
                self.channels.append(channel_id)
            else:
                raise ValueError(f"No channel found for search term: {search_term}")
        print("Channels ids found are %s" % (self.channels))

    def fetchThumbnails(self, max_videos_per_channel=400):
        if not self.needsDownload():
            print("You already downloaded the thumbnails (remove thumbnails.npz to re-download)")
            return

        thumbnailList = []
        self.get_channel_ids(self.names)

        for idx, channel in enumerate(self.channels):
            print('Visiting channel ' + channel)
            channelInfo = self.api.get_channel_info(channel_id=channel)
            if not channelInfo:
                raise RuntimeError("Failed to fetch channel info")
            channel_data = channelInfo.items[0].to_dict()
            playlistUploads = channel_data['contentDetails']['relatedPlaylists']['uploads']

            print("📺 Channel Title:", channel_data["snippet"]["title"])
            print("📈 Subscriber Count:", channel_data["statistics"].get("subscriberCount", "N/A"))
            print("📹 Video Count:", channel_data["statistics"].get("videoCount", "N/A"))

            playlistItens = self.api.get_playlist_items(playlist_id=playlistUploads, count=400)

            c = 0
            for k, item in enumerate(playlistItens.items):
                if c >= max_videos_per_channel:
                    break
                try:
                    videoId = item.snippet.resourceId.videoId
                    print(f'Getting thumbnail of video {videoId} (channel idx {idx})')
                    response = self.api.get_video_by_id(video_id=videoId)
                    video = response.items[0]
                    # Prefer medium thumbnail, fallback to default
                    thumbs = video.to_dict()['snippet']['thumbnails']
                    thumbnailUrl = thumbs.get('medium', thumbs.get('high', thumbs.get('default')))['url']

                    img = skimage.img_as_float(io.imread(thumbnailUrl))
                    # compute normalized histograms (NBINS)
                    histograms = [exposure.histogram(img[:, :, i], nbins=NBINS, normalize=True)[0]
                                  for i in range(img.shape[-1])]
                    thumbnail = Thumbnail(thumbnailUrl, histograms, self.names[idx], idx)
                    thumbnailList.append(thumbnail)
                    c += 1
                except Exception as e:
                    print("Warning: skipped item due to:", e)
                    continue

        print("Saving %d thumbnails to %s" % (len(thumbnailList), THUMBNAIL_CACHE))
        np.savez_compressed(THUMBNAIL_CACHE, thumbnails=thumbnailList)

    def needsDownload(self):
        return not Path(THUMBNAIL_CACHE).is_file()


# --------------------- Thumbnail & Data ------------------------
class Thumbnail:
    def __init__(self, url, rgbColors, classification, classificationNumber):
        self.url = url
        # ensure arrays of consistent dtype
        red = np.array(rgbColors[0], dtype=np.float32)
        green = np.array(rgbColors[1], dtype=np.float32)
        blue = np.array(rgbColors[2], dtype=np.float32)
        self.rgbColors = [red, green, blue, red + green + blue]
        self.classification = classification
        self.classificationNumber = classificationNumber

    def plot(self):
        img = skimage.img_as_float(io.imread(self.url))
        plt.figure()
        plt.imshow(img)
        plt.axis("off")

        fig, axs = plt.subplots(1, 3, figsize=(20, 5))
        axs[0].imshow(img[:, :, 0], cmap='Reds'); axs[0].grid(True)
        axs[1].imshow(img[:, :, 1], cmap='Greens'); axs[1].grid(True)
        axs[2].imshow(img[:, :, 2], cmap='Blues'); axs[2].grid(True)

        histograms = [exposure.histogram(img[:, :, i], nbins=NBINS, normalize=True)[0] for i in range(img.shape[-1])]
        fig, axs = plt.subplots(1, 3, figsize=(20, 5), sharey=True)
        axs[0].bar(np.arange(len(histograms[0])), histograms[0])
        axs[1].bar(np.arange(len(histograms[1])), histograms[1])
        axs[2].bar(np.arange(len(histograms[2])), histograms[2])
        plt.show()

    @staticmethod
    def loadThumbnails():
        data = np.load(THUMBNAIL_CACHE, allow_pickle=True)
        thumbList = data['thumbnails'].tolist()
        thumbnailList = []
        # just reconstruct Thumbnail objects (trust cache)
        for t in thumbList:
            thumbnailList.append(
                Thumbnail(t.url, t.rgbColors, t.classification, t.classificationNumber)
            )
        shuffle(thumbnailList)
        print("Loaded %d thumbnails from cache." % len(thumbnailList))
        return thumbnailList

    @staticmethod
    def getClassNameByNumber(classNumber):
        # safe: use CHANNELS mapping
        if 0 <= classNumber < len(CHANNELS):
            return CHANNELS[classNumber]
        return f"Class_{classNumber}"


class ThumbnailData(Dataset):
    def __init__(self, thumbnails):
        self.thumbnails = thumbnails

    def __len__(self):
        return len(self.thumbnails)

    def __getitem__(self, idx):
        # returns (torch.FloatTensor, torch.LongTensor)
        x = np.asarray(self.thumbnails[idx].rgbColors, dtype=np.float32)  # shape (4, NBINS)
        y = int(self.thumbnails[idx].classificationNumber)
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


# --------------------- Model ------------------------
class Net(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(Net, self).__init__()
        # Input shape = (batch, 1, 4, NBINS)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=2, stride=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=2, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

        # Keep fc sizes similar but make final layer num_classes
        # Note: flatten size depends on NBINS; the original code used 1408. If you change convs or NBINS,
        # you may need to recompute this value.
        self.fc1 = nn.Linear(1408, 400)
        self.fc2 = nn.Linear(400, 100)
        self.fc3 = nn.Linear(100, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


# --------------------- Training / Testing / Predict ------------------------
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # data: (B, 4, NBINS) -> want (B,1,4,NBINS) and float32
        data = data.unsqueeze(dim=1).to(device=device, dtype=torch.float32)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print("Args log interval %s %s" % (batch_idx, args.log_interval))
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}'
                  f' ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        if args.dry_run:
            break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    y_validation_list = []
    y_validation_pred_list = []

    with torch.no_grad():
        for data, target in test_loader:
            data = data.unsqueeze(dim=1).to(device=device, dtype=torch.float32)
            target = target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            y_validation_pred_list.extend(pred.view(-1).cpu().numpy().tolist())
            y_validation_list.extend(target.view(-1).cpu().numpy().tolist())

    test_loss /= len(test_loader.dataset)
    precision_validation = precision_score(y_validation_list, y_validation_pred_list, average="macro", zero_division=0)
    recall_validation = recall_score(y_validation_list, y_validation_pred_list, average="macro", zero_division=0)
    f1_validation = f1_score(y_validation_list, y_validation_pred_list, average="macro", zero_division=0)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%), Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}\n'
          .format(test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset),
                  precision_validation, recall_validation, f1_validation))


def predict(thumbnail_url, model, device):
    img = skimage.img_as_float(io.imread(thumbnail_url))
    plt.figure(); plt.imshow(img); plt.axis("off")

    histograms = [exposure.histogram(img[:, :, i], nbins=NBINS, normalize=True)[0] for i in range(img.shape[-1])]
    thumbnail_to_pred = Thumbnail(thumbnail_url, histograms, None, None)
    data = np.asarray(thumbnail_to_pred.rgbColors, dtype=np.float32)
    data = torch.from_numpy(data).unsqueeze(0).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        out = model(data)
        pred = out.argmax(dim=1).item()

    plt.title(Thumbnail.getClassNameByNumber(pred), fontsize=16)
    plt.show()


# --------------------- CLI / Main ------------------------
def main():
    parser = argparse.ArgumentParser(description='PyTorch Thumbnail Classifier (multi-channel)')
    parser.add_argument('--batch-size', type=int, default=10, metavar='N')
    parser.add_argument('--test-batch-size', type=int, default=10, metavar='N')
    parser.add_argument('--epochs', type=int, default=10, metavar='N')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M')
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--dry-run', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=1, metavar='S')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N')
    parser.add_argument('--save-model', action='store_true', default=True)
    parser.add_argument('--max-videos-per-channel', type=int, default=200)
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    crawler = YouTubeCrawler(names=CHANNELS)
    crawler.fetchThumbnails(max_videos_per_channel=args.max_videos_per_channel)

    thumbnails = Thumbnail.loadThumbnails()
    if len(thumbnails) == 0:
        raise RuntimeError("No thumbnails found in cache. Run crawler or check thumbnails.npz")

    # quick preview
    thumbnails[0].plot()
    dataset = ThumbnailData(thumbnails)

    # split
    validation_split = 0.15
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_loader = torch.utils.data.DataLoader(Subset(dataset, train_indices),
                                               batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(Subset(dataset, val_indices),
                                              batch_size=args.test_batch_size, shuffle=False)

    model = Net(num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "thumbnail_cnn_multi.pth")
        print("Saved model to thumbnail_cnn_multi.pth")

    # demo predict on first thumbnail
    predict(thumbnails[0].url, model, device)


if __name__ == '__main__':
    main()
