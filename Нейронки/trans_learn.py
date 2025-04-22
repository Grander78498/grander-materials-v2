import os
import librosa
from mutagen.flac import FLAC
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import math

# Конфигурация
class Config:
    music_dir = "D:\\Music\\complete"
    sr = 16000
    duration = 30
    n_mels = 128
    n_fft = 2048
    hop_length = 512
    batch_size = 32
    test_size = 0.2
    val_size = 0.1
    random_state = 78498
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    d_model = 128
    nhead = 4
    dim_feedforward = 128
    num_layers = 4
    dropout = 0.1

def load_data():
    df = pd.DataFrame(columns=['title', 'artist', 'genre', 'path'])

    for dir, dirname, files in os.walk(Config.music_dir):
        for filename in files:
            song_path = os.path.join(dir, filename)
            try:
                info = FLAC(song_path)
                genres = []
                for genre in info.get('genre', ['']):
                    genre = genre.lower().strip()
                    if genre == 'jazz':
                        continue
                    if genre == 'singer/songwriter':
                        genre = 'singer-songwriter'

                    replace_symbol = None
                    symbols = ';,/:'
                    for symbol in symbols:
                        if symbol in genre:
                            replace_symbol = symbol
                            break

                    if replace_symbol is not None:
                        splitted_genres = genre.split(replace_symbol)
                        genres.extend([x.replace('-', ' ').strip() for x in splitted_genres if 0 < len(x) < 30])
                        continue

                    if len(genre) > 30 or len(genre) == 0:
                        continue
                    genre.replace('-', ' ')
                    genres.append(genre)

                if len(genres) == 0:
                    continue

                df.loc[len(df)] = [
                    info.get('title', '')[0],
                    info.get('artist', '')[0],
                    genres[0],
                    song_path
                ]
            except Exception:
                pass

    return df

class AudioDataset(Dataset):
    def __init__(self, df, ohe):
        self.df = df
        self.ohe = ohe
        self.labels = ohe.transform(df.loc[:, ['genre']])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.df.iloc[idx]['path']
        audio = self.load_audio(file_path)
        spectrogram = self.create_mel_spectrogram(audio)
        label = self.labels[idx]

        return torch.FloatTensor(spectrogram).to(Config.device), torch.FloatTensor(label).to(Config.device)

    def load_audio(self, file_path):
        audio, _ = librosa.load(file_path, sr=Config.sr, duration=Config.duration)
        audio = np.pad(audio, [(0, Config.sr * Config.duration - audio.shape[0])])
        return audio

    def create_mel_spectrogram(self, audio):
        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio, sr=Config.sr, n_mels=Config.n_mels,
            n_fft=Config.n_fft, hop_length=Config.hop_length
        )
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        return mel_spectrogram_db

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)

class MusicGenreTransformer(nn.Module):
    def __init__(self, num_classes, input_dim=Config.n_mels):
        super(MusicGenreTransformer, self).__init__()

        self.input_proj = nn.Linear(input_dim, Config.d_model)

        self.pos_encoder = PositionalEncoding(Config.d_model, Config.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=Config.d_model,
            nhead=Config.nhead,
            dim_feedforward=Config.dim_feedforward,
            dropout=Config.dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, Config.num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(Config.d_model, Config.d_model // 2),
            nn.ReLU(),
            nn.Dropout(Config.dropout),
            nn.Linear(Config.d_model // 2, num_classes)
        )

    def forward(self, x):
        x = x.permute(2, 0, 1)

        x = self.input_proj(x)

        x = self.pos_encoder(x)

        x = self.transformer_encoder(x)

        x = x.mean(dim=0)

        x = self.classifier(x)

        return F.softmax(x, dim=1)

def train_model(model,
                train_loader,
                val_loader,
                criterion,
                optimizer,
                num_epochs=10):
    train_losses = []
    val_losses = []
    train_f1 = []
    val_f1 = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        for inputs, labels in tqdm(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = torch.abs(outputs - outputs.max(dim=0).values) < 1e-5
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        epoch_loss = running_loss / len(train_loader)
        epoch_f1 = f1_score(all_labels, all_preds, average='macro')
        train_losses.append(epoch_loss)
        train_f1.append(epoch_f1)

        val_loss, val_f1_score = evaluate_model(model, val_loader, criterion)
        val_losses.append(val_loss)
        val_f1.append(val_f1_score)

        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {epoch_loss:.4f} | Train F1: {epoch_f1:.4f}')
        print(f'Val Loss: {val_loss:.4f} | Val F1: {val_f1_score:.4f}')
        print('-' * 50)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_f1, label='Train F1')
    plt.plot(val_f1, label='Val F1')
    plt.title('Training and Validation F1 Score')
    plt.legend()

    plt.show()

    return model

def evaluate_model(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(loader):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            preds = torch.abs(outputs - outputs.max(dim=0).values) < 1e-5
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(loader)
    f1 = f1_score(all_labels, all_preds, average='macro')

    return avg_loss, f1

def main():
    df = load_data()
    print(f"Loaded {len(df)} audio files")

    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    labels = ohe.fit_transform(df.loc[:, ['genre']])
    num_classes = labels.shape[1]
    print(f"Number of classes: {num_classes}")
    print("Categories:", ohe.categories_)

    dataset = AudioDataset(df, ohe)

    test_size = int(len(dataset) * Config.test_size)
    val_size = int(len(dataset) * Config.val_size)
    train_size = len(dataset) - test_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(Config.random_state)
    )

    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=Config.batch_size)

    model = MusicGenreTransformer(num_classes).to(Config.device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("Starting training...")
    model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)

    test_loss, test_f1 = evaluate_model(model, test_loader, criterion)
    print(f'Test Loss: {test_loss:.4f} | Test F1: {test_f1:.4f}')

if __name__ == "__main__":
    main()
