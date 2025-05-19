# src/save_generated_npy.py

import os
import torch
import torch.nn as nn
import numpy as np
import argparse

# Ayarlar
SEQ_LEN      = 200
NUM_CHANNELS = 4
Z_DIM        = 100
HIDDEN_DIM   = 128
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generator mimarisi (WGAN-GP ile aynı)
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

def save_generated(generator_path="generator.pth",
                   out_file="generated_data.npy",
                   num_samples=100000,
                   batch_size=1000):
    # Modeli yükle
    G = Generator().to(DEVICE)
    G.load_state_dict(torch.load(generator_path, map_location=DEVICE))
    G.eval()

    all_samples = []
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            n = min(batch_size, num_samples - i)
            z = torch.randn(n, Z_DIM, device=DEVICE)
            fake = G(z).cpu().numpy()  # (n,200,4)
            all_samples.append(fake)

    data = np.vstack(all_samples)  # (num_samples,200,4)
    np.save(out_file, data)
    print(f"✅ {num_samples} adet sentetik segment '{out_file}' olarak kaydedildi.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator_path", type=str, default="generator.pth",
                        help="WGAN-GP ile eğitilmiş generator ağırlıkları")
    parser.add_argument("--out_file", type=str, default="generated_data.npy",
                        help="Kaydedilecek .npy dosyasının adı")
    parser.add_argument("--num_samples", type=int, default=100000,
                        help="Üretilecek sentetik segment sayısı")
    parser.add_argument("--batch_size", type=int, default=1000,
                        help="Tek seferde üretilecek batch büyüklüğü")
    args = parser.parse_args()

    save_generated(
        generator_path=args.generator_path,
        out_file=args.out_file,
        num_samples=args.num_samples,
        batch_size=args.batch_size
    )
