import math
import os
import pickle
import sys

import numpy as np
import torch
from torchvision import transforms

from utils import arguments as args
from utils.dataset import ImageCaptionDataset, get_data_loaders
from utils.model import Encoder, Decoder
from utils.vocabulary import Vocabulary


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    vocab = Vocabulary()
    dataset = ImageCaptionDataset(args.json_path, args.data_dir, vocab, args.max_caption_len, transform=transform)
    train_loader, val_loader = get_data_loaders(dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    pickle.dump(vocab, open(args.vocab_path), "wb")

    encoder = Encoder(args.embed_size).to(device)
    decoder = Decoder(args.embed_size, args.hidden_size, len(vocab), args.num_layers).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.embed.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    num_train_steps = math.ceil(len(train_loader.dataset) / args.batch_size)
    num_val_steps = math.ceil(len(val_loader.dataset) / args.batch_size)

    print("Start training...")
    for epoch in range(args.num_epochs):
        encoder.train()
        decoder.train()

        sum_train_loss, sum_val_loss = 0, 0
        sum_train_plx, sum_val_plx = 0, 0
        train_len, val_len = 0, 0
        for i, (images, captions) in enumerate(train_loader):
            images, captions = images.to(device), captions.to(device)

            features = encoder(images)
            outputs = decoder(features, captions)   # outputs shape (batch_size, caption_len, vocab_size)
            loss = criterion(outputs.view(-1, len(vocab)), captions.view(-1))

            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            optimizer.step()

            sum_train_loss += loss.item()
            perplex = np.exp(loss.item())
            sum_train_plx += perplex
            train_len += 1

            stats = 'Train :: Epoch [%d/%d], Step [%d/%d] - Loss: %.4f, Perplexity: %5.4f' % (
                epoch + 1, args.num_epochs, i + 1, num_train_steps, loss.item(), perplex)
            print('\r' + stats, end="")
            sys.stdout.flush()
            if (i + 1) % 15 == 0:
                print('\r' + stats)

        if not os.path.isdir(args.out_dir):
            os.mkdir(args.out_dir)
        torch.save(decoder.state_dict(), os.path.join(args.out_dir, f'decoder_e{epoch + 1}.pkl'))
        torch.save(encoder.state_dict(), os.path.join(args.out_dir, f'encoder_e{epoch + 1}.pkl'))

        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            for i, (images, captions) in enumerate(val_loader):
                images, captions = images.to(device), captions.to(device)

                features = encoder(images)
                outputs = decoder(features, captions)
                loss = criterion(outputs.view(-1, len(vocab)), captions.view(-1))

                sum_val_loss += loss.item()
                perplex = np.exp(loss.item())
                sum_val_plx += perplex
                val_len += 1

                stats = 'Validation :: Epoch [%d/%d], Step [%d/%d] - Loss: %.4f, Perplexity: %5.4f' % (
                    epoch + 1, args.num_epochs, i + 1, num_val_steps, loss.item(), perplex)
                print('\r' + stats, end="")
                sys.stdout.flush()

        train_loss = sum_train_loss / train_len
        train_plx = sum_train_plx / train_len
        val_loss = sum_val_loss / val_len
        val_plx = sum_val_plx / val_len
        print('\nEpoch [%d/%d] :: Train - Loss: %.4f, Perplexity: %5.3f :: Validation - Loss: %.4f, Perplexity: %5.3f\n' % (
            epoch + 1, args.num_epochs, train_loss, train_plx, val_loss, val_plx))
