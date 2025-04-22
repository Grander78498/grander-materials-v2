import os
import librosa
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
from datasets import load_dataset, Audio
from transformers import AutoFeatureExtractor
import warnings

# Игнорирование предупреждений
warnings.filterwarnings('ignore')

# Конфигурация
class Config:
    dataset_name = "marsyas/gtzan"  # Имя датасета в Hugging Face Hub
    sr = 22050  # Частота дискретизации
    duration = 30  # Длительность аудио
    n_mels = 128  # Количество mel-фильтров
    n_fft = 2048  # Размер FFT
    hop_length = 512  # Шаг для FFT
    batch_size = 32  # Размер батча
    test_size = 0.2  # Размер тестовой выборки
    val_size = 0.1  # Размер валидационной выборки
    random_state = 42  # Seed для воспроизводимости
    num_epochs = 30  # Количество эпох
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Использовать CUDA если доступно
    feature_extractor_name = "facebook/wav2vec2-base"  # Модель для экстракции фич (не используется напрямую)

# Загрузка и предобработка данных GTZAN через Hugging Face
def load_gtzan_data():
    dataset = load_dataset(Config.dataset_name, split="train")
    
    # Преобразуем аудио в нужный формат
    dataset = dataset.cast_column("audio", Audio(sampling_rate=Config.sr))
    
    # Создаем feature extractor (можно использовать для нормализации и т.д.)
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        Config.feature_extractor_name,
        sampling_rate=Config.sr,
        do_normalize=True,
    )
    
    return dataset, feature_extractor

# Класс Dataset для аудио
class AudioDataset(Dataset):
    def __init__(self, dataset, feature_extractor):
        self.dataset = dataset
        self.feature_extractor = feature_extractor
        self.genres = [
            'blues', 'classical', 'country', 'disco', 'hiphop', 
            'jazz', 'metal', 'pop', 'reggae', 'rock'
        ]
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        audio = item["audio"]["array"]
        genre = item["genre"]
        
        # Приводим к нужной длине (на всякий случай)
        if len(audio) > Config.sr * Config.duration:
            audio = audio[:Config.sr * Config.duration]
        else:
            audio = np.pad(audio, (0, max(0, Config.sr * Config.duration - len(audio))))
        
        # Создаем mel-спектрограмму
        spectrogram = self.create_mel_spectrogram(audio)
        
        # Преобразуем жанр в индекс
        label = genre
        
        return torch.FloatTensor(spectrogram), torch.tensor(label, dtype=torch.long)
    
    def create_mel_spectrogram(self, audio):
        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio, sr=Config.sr, n_mels=Config.n_mels, 
            n_fft=Config.n_fft, hop_length=Config.hop_length
        )
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        return mel_spectrogram_db

# Улучшенная CNN модель
class MusicGenreCNN(nn.Module):
    def __init__(self, num_classes):
        super(MusicGenreCNN, self).__init__()
        
        # Conv layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        # Автоматический расчет размера для полносвязного слоя
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 8 * 8, 512)  # Размер рассчитан для 128x128 спектрограмм после 4 пулингов
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = x.unsqueeze(1)  # [batch, 1, n_mels, time]
        
        # Conv layers with batch norm and ReLU
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        x = self.flatten(x)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Функция для обучения
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    model.to(Config.device)
    train_losses = []
    val_losses = []
    train_acc = []
    val_acc = []
    
    best_val_acc = 0.0
    best_model_weights = None
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            inputs, labels = inputs.to(Config.device), labels.to(Config.device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)
        epoch_f1 = f1_score(all_labels, all_preds, average='macro')
        train_losses.append(epoch_loss)
        train_acc.append(epoch_acc)
        
        # Валидация
        val_loss, val_accuracy, val_f1 = evaluate_model(model, val_loader, criterion)
        val_losses.append(val_loss)
        val_acc.append(val_accuracy)
        
        # Сохраняем лучшую модель
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_model_weights = model.state_dict().copy()
        
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f} | Train F1: {epoch_f1:.4f}')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f} | Val F1: {val_f1:.4f}')
        print('-' * 50)
    
    # Загружаем веса лучшей модели
    model.load_state_dict(best_model_weights)
    
    # Построение графиков
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='Train Accuracy')
    plt.plot(val_acc, label='Val Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.show()
    
    return model

# Функция для оценки модели
def evaluate_model(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(Config.device), labels.to(Config.device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = running_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    return avg_loss, accuracy, f1

def main():
    # Загрузка данных GTZAN через Hugging Face
    print("Loading dataset from Hugging Face Hub...")
    dataset, feature_extractor = load_gtzan_data()
    print(f"Loaded {len(dataset)} audio samples")
    
    # Создание датасета
    audio_dataset = AudioDataset(dataset, feature_extractor)
    
    # Разделение на train/val/test
    test_size = int(len(audio_dataset) * Config.test_size)
    val_size = int(len(audio_dataset) * Config.val_size)
    train_size = len(audio_dataset) - test_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        audio_dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(Config.random_state)
    )
    
    # Создание DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=Config.batch_size, 
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=Config.batch_size, 
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Инициализация модели
    num_classes = 10  # GTZAN имеет 10 жанров
    model = MusicGenreCNN(num_classes)
    model.to(Config.device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.5, 
        patience=3, 
        verbose=True
    )
    
    # Обучение модели
    print("\nStarting training...")
    print(f"Using device: {Config.device}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    model = train_model(
        model, 
        train_loader, 
        val_loader, 
        criterion, 
        optimizer, 
        num_epochs=Config.num_epochs
    )
    
    # Оценка на тестовом наборе
    test_loss, test_accuracy, test_f1 = evaluate_model(model, test_loader, criterion)
    print(f'\nTest Results:')
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Accuracy: {test_accuracy:.4f}')
    print(f'Test F1 Score: {test_f1:.4f}')

if __name__ == "__main__":
    main()