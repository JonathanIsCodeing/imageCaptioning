from abc import ABC

import torch
import torchvision
from torch import nn


class Encoder(nn.Module, ABC):
    def __init__(self, embed_size=256):
        super(Encoder, self).__init__()

        resnet = torchvision.models.resnext101_32x8d(pretrained=True, progress=True)
        for param in resnet.parameters():
            param.requires_grad_(False)

        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        out = self.resnet(images)               # (batch_size, 2048, image_size/32, image_size/32)
        out = out.view(out.size(0), -1)         # (batch_size, 2048)
        out = self.bn(self.embed(out))          # (batch_size, embed_size)
        return out


class Decoder(nn.Module, ABC):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        embedded = self.embedding(captions[:, :-1])
        out = torch.cat((features.unsqueeze(dim=1), embedded), dim=1)
        out, _ = self.lstm(out)
        out = self.linear(out)

        return out      # (batch_size, caption_len, vocab_size)

    def sample(self, features, states=None, max_len=20):
        caption = []
        inputs = features.unsqueeze(1)
        for i in range(max_len):
            hidden, states = self.lstm(inputs, states)  # (batch_size, 1, hidden_size), _
            outputs = self.linear(hidden.squeeze(1))    # (batch_size, vocab_size)
            _, prediction = outputs.max(dim=1)          # (batch_size)
            inputs = self.embedding(prediction)         # (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                # (batch_size, 1, embed_size)
            caption.append(prediction.item())

        return caption
