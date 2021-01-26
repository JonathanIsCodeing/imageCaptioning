import json
import os
from random import randint

import torch
import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms


class ImageCaptionDataset(Dataset):
    def __init__(self, json_file, data_dir, vocab, max_caption_len=20, transform=None, num_test_images=0):
        self.data_dir = data_dir
        self.vocab = vocab
        self.max_caption_len = max_caption_len

        # Add minimally needed transformation if not given as input
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        # Read json file
        self.captions = []
        self.images_dir = []
        with open(json_file) as json_file:
            data = json.load(json_file)
            for img in tqdm.tqdm(data['images']):
                file_name = img['file_name']
                if os.path.exists(os.path.join(data_dir, file_name)):
                    self.images_dir.append(file_name)
                    self.captions.append(img['captions'])

        self.vocab.build(self.captions)

        # Split test images
        if num_test_images:
            self.test_images_dir = self.images_dir[-num_test_images:]
            self.images_dir = self.images_dir[:-(num_test_images + 1)]
            self.test_captions = self.captions[-num_test_images:]
            self.captions = self.captions[:-(num_test_images + 1)]

    def __len__(self):
        return len(self.images_dir)

    def __getitem__(self, i):
        img_path = os.path.join(self.data_dir, self.images_dir[i])
        img = Image.open(img_path).convert("RGB")
        tensor_image = self.transform(img)

        tokenized_captions = []
        for caption in self.captions[i]:
            tokenized_captions.append(self.vocab.tokenize_caption(caption, self.max_caption_len))

        return tensor_image, tokenized_captions[randint(0, len(self.captions[i]) - 1)]


def get_data_loaders(dataset, batch_size, num_workers=1, train_ratio=0.9):
    train_len = int(len(dataset) * train_ratio)
    val_len = int(len(dataset) - train_len)
    train_set, val_set = random_split(dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader


class TestDataset(Dataset):
    def __init__(self, test_dir, transform=None):
        self.test_dir = test_dir
        self.transform = transform

        # Read file name of images from dir
        self.images = [img for img in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, img))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        file_name = self.images[i]
        image = Image.open(os.path.join(self.test_dir, file_name)).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, file_name
