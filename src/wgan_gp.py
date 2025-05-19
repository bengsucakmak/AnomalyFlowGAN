# src/wgan_gp.py

import os
import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import argparse
from config import SEQ_LEN, BATCH_SIZE
from utils import set_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Z_DIM = 100
HIDDEN_DIM = 128
EPOCHS = 50
LAMBDA_GP = 10
N_CRITIC = 5
NUM_CHANNELS = 4

def load_segmented_data(path, win_len=200, step=200, max_segments=None):
    files = sorted([f for f in os.listdir(path) if f.endswith('.npy')])
    segments = []

    for f in files:
        arr = np.load(os.path.join(path, f))
        if arr.ndim != 3 or arr.shape[2] < NUM_CHANNELS:
            continue
        arr = arr[:, :, :NUM_CHANNELS]
        for sample in arr:
            for i in range(0, sample.shape[0] - win_len + 1, step):
                segments.append(sample[i:i+win_len])

    total = len(segments)
    if max_segments is not None and total > max_segments:
        segments = segments[:max_segments]
    print(f"✅ Segment sayısı: {len(segments)} (toplam raw: {total}), segment shape: {segments[0].shape}")
    return torch.tensor(np.stack(segments), dtype=torch.float32)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(Z_DIM, HIDDEN_DIM),
            nn.ReLU(True),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM * 2),
            nn.ReLU(True),
            nn.Linear(HIDDEN_DIM * 2, SEQ_LEN * NUM_CHANNELS),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z).view(-1, SEQ_LEN, NUM_CHANNELS)

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(SEQ_LEN * NUM_CHANNELS, HIDDEN_DIM * 4),
            nn.LeakyReLU(0.2, True),
            nn.Linear(HIDDEN_DIM * 4, HIDDEN_DIM * 2),
            nn.LeakyReLU(0.2, True),
            nn.Linear(HIDDEN_DIM * 2, 1)
        )

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))

def compute_gradient_penalty(critic, real, fake):
    alpha = torch.rand(real.size(0), 1, 1, device=device).expand_as(real)
    interpolates = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interpolates = critic(interpolates)
    grads = autograd.grad(
        outputs=d_interpolates, inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True, retain_graph=True
    )[0]
    grads = grads.view(grads.size(0), -1)
    gp = ((grads.norm(2, dim=1) - 1) ** 2).mean()
    return gp

def train_wgan_gp(data_path, max_segments):
    set_seed()
    data = load_segmented_data(data_path, win_len=SEQ_LEN, step=SEQ_LEN, max_segments=max_segments)
    loader = DataLoader(TensorDataset(data), batch_size=BATCH_SIZE, shuffle=True)

    G = Generator().to(device)
    D = Critic().to(device)
    opt_G = optim.Adam(G.parameters(), lr=1e-4, betas=(0.0, 0.9))
    opt_D = optim.Adam(D.parameters(), lr=1e-4, betas=(0.0, 0.9))

    G_losses, D_losses = [], []

    print("🚀 WGAN-GP eğitimi başlıyor...")
    for epoch in range(1, EPOCHS+1):
        for real_batch, in loader:
            real = real_batch.to(device)
            b_size = real.size(0)

            # Critic eğitim
            for _ in range(N_CRITIC):
                z = torch.randn(b_size, Z_DIM, device=device)
                fake = G(z).detach()
                loss_D = -D(real).mean() + D(fake).mean()
                loss_D += LAMBDA_GP * compute_gradient_penalty(D, real, fake)

                opt_D.zero_grad()
                loss_D.backward()
                opt_D.step()

            # Generator eğitim
            z = torch.randn(b_size, Z_DIM, device=device)
            loss_G = -D(G(z)).mean()
            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

        D_losses.append(loss_D.item())
        G_losses.append(loss_G.item())
        print(f"📈 Epoch {epoch}/{EPOCHS}  D_loss: {loss_D.item():.4f}  G_loss: {loss_G.item():.4f}")

    torch.save(G.state_dict(), "generator.pth")
    torch.save(D.state_dict(), "critic.pth")
    print("✅ Eğitim tamamlandı. 'generator.pth' ve 'critic.pth' kaydedildi.")

    # Loss grafiği
    plt.figure(figsize=(10,5))
    plt.plot(D_losses, label="Critic Loss")
    plt.plot(G_losses, label="Generator Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("WGAN-GP Kayıp Grafiği")
    plt.legend(); plt.grid()
    plt.savefig("loss_plot.png")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed_dir', type=str, required=True,
                        help=".npy segment dosyalarını içeren klasör")
    parser.add_argument('--max_segments', type=int, default=100000,
                        help="Eğitimde kullanılacak maksimum segment sayısı")
    args = parser.parse_args()
    train_wgan_gp(args.processed_dir, args.max_segments)
