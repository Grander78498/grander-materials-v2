import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
from collections import defaultdict
import pretty_midi
from tqdm import tqdm

SEQ_LENGTH = 128
N_NOTES = 128
BATCH_SIZE = 32
LATENT_DIM = 100
LR = 0.0002
BETA1 = 0.5
N_EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = "classical_music_dataset"

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.main = nn.Sequential(
            nn.Linear(LATENT_DIM, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            
            nn.Linear(1024, SEQ_LENGTH * N_NOTES),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.main(x)
        x = x.view(-1, SEQ_LENGTH, N_NOTES)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.main = nn.Sequential(
            nn.Linear(SEQ_LENGTH * N_NOTES, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = x.view(-1, SEQ_LENGTH * N_NOTES)
        return self.main(x)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class MIDIDataset(Dataset):
    def __init__(self, root_dir, seq_length=128, fs=10):
        self.root_dir = root_dir
        self.seq_length = seq_length
        self.fs = fs
        self.data = []
        
        self._load_midi_files()
    
    def _load_midi_files(self):
        print("Loading MIDI files...")
        midi_files = []
        for dirpath, _, filenames in os.walk(self.root_dir):
            for f in filenames:
                if f.endswith('.mid'):
                    midi_files.append(os.path.join(dirpath, f))
        
        for midi_file in tqdm(midi_files[:100]):
            try:
                pm = pretty_midi.PrettyMIDI(midi_file)
                piano_roll = pm.get_piano_roll(fs=self.fs).T
                piano_roll = (piano_roll > 0).astype(float)
                
                for i in range(0, piano_roll.shape[0] - self.seq_length, self.seq_length):
                    seq = piano_roll[i:i + self.seq_length]
                    if seq.shape[0] == self.seq_length:
                        self.data.append(seq[:self.seq_length, :N_NOTES])  # Берем только первые N_NOTES нот
            except:
                continue
        
        self.data = np.stack(self.data)
        print(f"Loaded {len(self.data)} sequences")

    def _save_midi(self, piano_roll, filename, fs=10, program=0):
        midi = pretty_midi.PrettyMIDI()
        
        instrument = pretty_midi.Instrument(program=program)
        
        threshold = 0.5
        piano_roll_binary = piano_roll > threshold
        
        for note_num in range(piano_roll_binary.shape[1]):
            note_sequence = piano_roll_binary[:, note_num]
            
            diff = np.diff(note_sequence, prepend=0, append=0)
            
            note_starts = np.where(diff == 1)[0]
            note_ends = np.where(diff == -1)[0]
            
            for start, end in zip(note_starts, note_ends):
                start_time = start / fs
                end_time = end / fs
                
                note = pretty_midi.Note(
                    velocity=100,
                    pitch=note_num,
                    start=start_time,
                    end=end_time
                )
                instrument.notes.append(note)
        
        midi.instruments.append(instrument)
        
        midi.write(filename)
        print(f"Saved generated MIDI to {filename}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx])

generator = Generator().to(DEVICE)
discriminator = Discriminator().to(DEVICE)

generator.apply(weights_init)
discriminator.apply(weights_init)

criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=LR, betas=(BETA1, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=LR, betas=(BETA1, 0.999))

dataset = MIDIDataset(DATA_PATH, seq_length=SEQ_LENGTH)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

G_losses = []
D_losses = []

for epoch in range(N_EPOCHS):
    for i, real_music in enumerate(tqdm(dataloader)):
        real_music = real_music.to(DEVICE)
        batch_size = real_music.size(0)
        
        real_labels = torch.full((batch_size, 1), 0.9, device=DEVICE)
        fake_labels = torch.full((batch_size, 1), 0.0, device=DEVICE)
        
        discriminator.zero_grad()
        
        output = discriminator(real_music)
        errD_real = criterion(output, real_labels)
        errD_real.backward()
        
        noise = torch.randn(batch_size, LATENT_DIM, device=DEVICE)
        fake_music = generator(noise)
        output = discriminator(fake_music.detach())
        errD_fake = criterion(output, fake_labels)
        errD_fake.backward()
        
        errD = errD_real + errD_fake
        optimizer_D.step()
        
        generator.zero_grad()
        output = discriminator(fake_music)
        errG = criterion(output, real_labels)
        errG.backward()
        optimizer_G.step()
        
        G_losses.append(errG.item())
        D_losses.append(errD.item())
    
    print(f'Epoch [{epoch+1}/{N_EPOCHS}] Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f}')
    
    if (epoch + 1) % 5 == 0:
        with torch.no_grad():
            generator.eval()
            fake_samples = generator(torch.randn(1, LATENT_DIM, device=DEVICE)).cpu().numpy()
            generator.train()
            dataset._save_midi(fake_samples[0], f"generated_epoch_{epoch+1}.mid")

torch.save(generator.state_dict(), 'generator.pth')
torch.save(discriminator.state_dict(), 'discriminator.pth')

plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses, label="G")
plt.plot(D_losses, label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()
