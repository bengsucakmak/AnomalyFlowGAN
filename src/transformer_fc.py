# src/transformer_fc.py

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from utils import set_seed

# ——————————————————————————————
#  Ayarlar
# ——————————————————————————————
SEQ_LEN      = 200
NUM_CHANNELS = 4
ENC_LEN      = 150     # Girdi uzunluğu
PRED_LEN     = SEQ_LEN - ENC_LEN  # 50
D_MODEL      = 64      # Embedding boyutu
N_HEADS      = 4
N_LAYERS     = 3
DIM_FF       = 128
DROPOUT      = 0.1
BATCH_SIZE   = 256
EPOCHS       = 50
LR           = 1e-4
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ——————————————————————————————
#  Positional Encoding
# ——————————————————————————————
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=ENC_LEN):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :].unsqueeze(0)

# ——————————————————————————————
#  Dataset: Raw .npy dosyalarını segmentleyip, sentetikle birleştirir
# ——————————————————————————————
class TimeSeriesDataset(Dataset):
    def __init__(self, real_dir="data/processed", fake_file="generated_data.npy", max_segments=None):
        real_segments = []
        for fname in sorted(os.listdir(real_dir)):
            if not fname.endswith(".npy"):
                continue
            arr = np.load(os.path.join(real_dir, fname))  # (Ni, L, Ci)
            if arr.ndim != 3 or arr.shape[2] < NUM_CHANNELS:
                continue
            arr = arr[:, :, :NUM_CHANNELS]
            # kayan pencere
            for sample in arr:
                n_win = sample.shape[0] // SEQ_LEN
                for w in range(n_win):
                    seg = sample[w*SEQ_LEN:(w+1)*SEQ_LEN]
                    if seg.shape == (SEQ_LEN, NUM_CHANNELS) and np.isfinite(seg).all():
                        real_segments.append(seg)
        if not real_segments:
            raise RuntimeError("Gerçek segment bulunamadı!")
        total = len(real_segments)
        if max_segments and total > max_segments:
            idx = np.random.choice(total, max_segments, replace=False)
            real_segments = [real_segments[i] for i in idx]
        real = np.stack(real_segments)  # (N_real, 200, 4)

        # Sentetik
        fake = np.load(fake_file)       # (N_fake,200,4)

        # Combine and normalize
        data = np.vstack([real, fake])
        self.mean = data.mean(axis=(0,1), keepdims=True)
        self.std  = data.std(axis=(0,1), keepdims=True) + 1e-6
        data = (data - self.mean) / self.std

        # Split into input/output
        self.X = data[:, :ENC_LEN, :]   # (N,150,4)
        self.Y = data[:, ENC_LEN:, :]   # (N,50,4)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.Y[idx], dtype=torch.float32)
        )

# ——————————————————————————————
#  Transformer Forecast Modeli
# ——————————————————————————————
class TransformerForecast(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(NUM_CHANNELS, D_MODEL)
        self.pos_enc    = PositionalEncoding(D_MODEL, max_len=ENC_LEN)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=N_HEADS,
            dim_feedforward=DIM_FF, dropout=DROPOUT
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=N_LAYERS)
        self.decoder = nn.Sequential(
            nn.Linear(D_MODEL, DIM_FF),
            nn.ReLU(),
            nn.Linear(DIM_FF, NUM_CHANNELS * PRED_LEN)
        )

    def forward(self, x):
        x = self.input_proj(x)            # (B,ENC_LEN,D_MODEL)
        x = self.pos_enc(x)
        x = x.transpose(0,1)              # (ENC_LEN,B,D_MODEL)
        enc = self.encoder(x)             # (ENC_LEN,B,D_MODEL)
        enc = enc.transpose(0,1)          # (B,ENC_LEN,D_MODEL)
        rep = enc.mean(dim=1)             # (B,D_MODEL)
        out = self.decoder(rep)           # (B, NUM_CHANNELS*PRED_LEN)
        return out.view(-1, PRED_LEN, NUM_CHANNELS)

# ——————————————————————————————
#  Eğitim Fonksiyonu
# ——————————————————————————————
def train_transformer(real_dir="data/processed",
                      fake_file="generated_data.npy",
                      max_segments=100000):
    set_seed()
    ds = TimeSeriesDataset(real_dir, fake_file, max_segments)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    model = TransformerForecast().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    losses = []

    print("🚀 Transformer forecast eğitimi başlıyor...")
    for epoch in range(1, EPOCHS+1):
        total = 0.0
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        avg = total / len(loader)
        losses.append(avg)
        print(f"  Epoch {epoch}/{EPOCHS}  MSE: {avg:.6f}")

    torch.save(model.state_dict(), "transformer_fc.pth")
    print("✅ Transformer modeli kaydedildi: transformer_fc.pth")

    plt.figure(figsize=(8,4))
    plt.plot(losses, label="MSE")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Transformer Forecast Loss")
    plt.grid()
    plt.savefig("transformer_fc_loss.png")
    plt.show()

if __name__ == "__main__":
    train_transformer(
        real_dir="data/processed",
        fake_file="generated_data.npy",
        max_segments=100000
    )
