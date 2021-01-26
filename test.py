import json
import pickle

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from utils import arguments as args
from utils.dataset import TestDataset
from utils.model import Encoder, Decoder


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    with open(args.vocab_path, "rb") as file:
        vocab = pickle.load(file)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = TestDataset(args.test_dir, transform)
    test_loader = DataLoader(dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)

    encoder = Encoder(args.embed_size).eval().to(device)
    decoder = Decoder(args.embed_size, args.hidden_size, len(vocab), args.num_layers).eval().to(device)

    encoder.load_state_dict(torch.load(args.encoder_path, map_location=torch.device(device)))
    decoder.load_state_dict(torch.load(args.decoder_path, map_location=torch.device(device)))

    predictions_tokenized = []
    predictions_sentences = []
    file_names = []

    for i, (images, img_names) in enumerate(test_loader):
        images = images.to(device)
        file_names.extend(img_names)
        for img in images:
            img.unsqueeze_(0)
            feature = encoder(img)
            tokenized = decoder.sample(feature)

            words = []
            for word_id in tokenized:
                word = vocab.idx2word[word_id]
                words.append(word)
                if word == '<end>':
                    break
            sentence = ' '.join(words[1:-1])
            predictions_tokenized.append(tokenized)
            predictions_sentences.append(sentence)

    # Create Json
    data = {'images': []}
    for i in range(len(predictions_sentences)):
        data['images'].append({
            'file_name': file_names[i],
            'captions': predictions_sentences[i]
        })

    with open(args.result_file, 'w') as outfile:
        json.dump(data, outfile, indent=4)
