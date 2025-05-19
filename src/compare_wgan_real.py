import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# --- Ayarlar ---
SEQ_LEN      = 200
NUM_CHANNELS = 4
Z_DIM        = 100
HIDDEN_DIM   = 128
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_SAMPLES    = 5    # Karşılaştırmak istediğin örnek sayısı

# --- Generator (WGAN ile aynı mimari) ---
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

# --- Real Örnekleri Yükleme ---
def load_real_segments(processed_dir="data/processed", n=N_SAMPLES):
    segments = []
    for fname in sorted(os.listdir(processed_dir)):
        if not fname.endswith('.npy'):
            continue
        arr = np.load(os.path.join(processed_dir, fname))  # shape (Ni, 20480, Ci)
        if arr.ndim != 3 or arr.shape[2] < NUM_CHANNELS:
            continue
        arr = arr[:, :, :NUM_CHANNELS]
        for sample in arr:
            # ilk pencereyi al
            if sample.shape[0] >= SEQ_LEN:
                segments.append(sample[:SEQ_LEN])
            if len(segments) >= n:
                return np.stack(segments)
    raise RuntimeError(f"{processed_dir} içinde yeterli real segment yok")

# --- Fake Örnekleri Üretme ---
def generate_fake(generator_path="generator.pth", n=N_SAMPLES):
    G = Generator().to(DEVICE)
    G.load_state_dict(torch.load(generator_path, map_location=DEVICE))
    G.eval()
    z = torch.randn(n, Z_DIM, device=DEVICE)
    with torch.no_grad():
        fake = G(z).cpu().numpy()  # (n, 200, 4)
    return fake

# --- Karşılaştırma ve Görselleştirme ---
def compare_and_plot(real, fake, save_path="wgan_real_vs_fake.png"):
    n = real.shape[0]
    plt.figure(figsize=(16, 4 * n))
    for i in range(n):
        for c in range(NUM_CHANNELS):
            # Real
            ax = plt.subplot(n, NUM_CHANNELS*2, i*(NUM_CHANNELS*2) + c*2 + 1)
            ax.plot(real[i,:,c])
            ax.set_title(f"Real {i+1} - Ch {c+1}")
            ax.grid(True)
            # Fake
            ax = plt.subplot(n, NUM_CHANNELS*2, i*(NUM_CHANNELS*2) + c*2 + 2)
            ax.plot(fake[i,:,c])
            ax.set_title(f"Fake {i+1} - Ch {c+1}")
            ax.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"✅ Karşılaştırma görseli '{save_path}' olarak kaydedildi.")

if __name__ == "__main__":
    real_samples = load_real_segments("data/processed", n=N_SAMPLES)
    fake_samples = generate_fake("generator.pth", n=N_SAMPLES)
    compare_and_plot(real_samples, fake_samples)
