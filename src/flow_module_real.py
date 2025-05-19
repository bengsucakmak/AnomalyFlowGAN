# src/flow_module_real.py

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from utils import set_seed
import argparse

# ——————————————————————————————
#  Ayarlar
# ——————————————————————————————
SEQ_LEN = 200
NUM_CHANNELS = 4
INPUT_DIM = SEQ_LEN * NUM_CHANNELS  # 800
HIDDEN_DIM = 512
NUM_COUPLING_LAYERS = 6
EPOCHS = 50
BATCH_SIZE = 256
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ——————————————————————————————
#  MLP Blok
# ——————————————————————————————
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, x):
        return self.net(x)

# ——————————————————————————————
#  Affine Coupling Katmanı
# ——————————————————————————————
class CouplingLayer(nn.Module):
    def __init__(self, dim, mask):
        super().__init__()
        self.mask = mask
        self.scale_net = MLP(dim, HIDDEN_DIM, dim)
        self.translate_net = MLP(dim, HIDDEN_DIM, dim)

    def forward(self, x):
        x_masked = x * self.mask
        s = torch.tanh(self.scale_net(x_masked)) * (1 - self.mask)
        t = self.translate_net(x_masked) * (1 - self.mask)
        z = x_masked + (1 - self.mask) * (x * torch.exp(s) + t)
        log_det = s.sum(dim=1)
        return z, log_det

    def inverse(self, z):
        z_masked = z * self.mask
        s = torch.tanh(self.scale_net(z_masked)) * (1 - self.mask)
        t = self.translate_net(z_masked) * (1 - self.mask)
        x = z_masked + (1 - self.mask) * ((z - t) * torch.exp(-s))
        return x

# ——————————————————————————————
#  RealNVP Modeli
# ——————————————————————————————
class RealNVP(nn.Module):
    def __init__(self, dim, n_coupling_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(n_coupling_layers):
            mask = self._create_mask(i % 2)
            self.layers.append(CouplingLayer(dim, mask))
        self.base_dist = torch.distributions.MultivariateNormal(
            torch.zeros(dim, device=DEVICE),
            torch.eye(dim, device=DEVICE)
        )

    def _create_mask(self, even=True):
        mask = torch.zeros(INPUT_DIM, device=DEVICE)
        mask[::2] = 1 if even else 0
        mask[1::2] = 0 if even else 1
        return mask

    def forward(self, x):
        log_det = 0
        for layer in self.layers:
            x, ldj = layer(x)
            log_det += ldj
        return x, log_det

    def log_prob(self, x):
        z, log_det = self.forward(x)
        return self.base_dist.log_prob(z) + log_det

# ——————————————————————————————
#  Veri yükleme & segmentleme & örnekleme
# ——————————————————————————————
def load_and_segment(processed_dir, max_segments=None):
    print("📂 Segmentlere ayırma başlıyor...")
    segments = []
    for fname in sorted(os.listdir(processed_dir)):
        if not fname.endswith(".npy"):
            continue
        arr = np.load(os.path.join(processed_dir, fname))  # (Ni, 20480, Ci)
        print(f"  • {fname}: {arr.shape}")
        if arr.ndim != 3:
            continue
        # Kanal sayısını kesin 4'e indir
        if arr.shape[2] >= NUM_CHANNELS:
            arr = arr[:, :, :NUM_CHANNELS]
        else:
            continue

        # Kayan pencere: yalnızca tam SEQ_LEN uzun segmentleri al
        total_len = arr.shape[1]
        n_windows = total_len // SEQ_LEN
        for sample in arr:
            for w in range(n_windows):
                start = w * SEQ_LEN
                seg = sample[start:start+SEQ_LEN]  # (SEQ_LEN, 4)
                if seg.shape == (SEQ_LEN, NUM_CHANNELS) and np.isfinite(seg).all():
                    segments.append(seg)

    if not segments:
        raise RuntimeError("Hiç segment bulunamadı!")
    N = len(segments)
    print(f"▶️ Toplam raw segment: {N}")

    # Rastgele örnekleme
    if max_segments and N > max_segments:
        idx = np.random.choice(N, max_segments, replace=False)
        segments = [segments[i] for i in idx]
        print(f"▶️ {max_segments} segment rastgele seçildi.")

    data = np.stack(segments)  # (N, SEQ_LEN, NUM_CHANNELS)
    print(f"✅ Kullanılacak segment sayısı: {data.shape[0]}, shape: {data.shape[1:]}")
    x = data.reshape(data.shape[0], -1)  # (N, SEQ_LEN*NUM_CHANNELS)
    return torch.tensor(x, dtype=torch.float32)

# ——————————————————————————————
#  Eğitim
# ——————————————————————————————
def train_flow_real(processed_dir, max_segments):
    set_seed()
    x = load_and_segment(processed_dir, max_segments).to(DEVICE)
    loader = torch.utils.data.DataLoader(x, batch_size=BATCH_SIZE, shuffle=True)

    model = RealNVP(dim=INPUT_DIM, n_coupling_layers=NUM_COUPLING_LAYERS).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    losses = []
    print("🚀 RealNVP (gerçek veri) eğitimi başlıyor...")
    for epoch in range(1, EPOCHS+1):
        total = 0
        for batch in loader:
            batch = batch.to(DEVICE)
            loss = -model.log_prob(batch).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        avg = total / len(loader)
        losses.append(avg)
        print(f"  Epoch {epoch}/{EPOCHS}  Loss: {avg:.4f}")

    # Modeli kaydet
    torch.save(model.state_dict(), "realnvp_real.pth")
    print("✅ Model kaydedildi: realnvp_real.pth")

    # Loss grafiği
    plt.figure(figsize=(10,4))
    plt.plot(losses, label="NegLogLik")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Real Data RealNVP Loss")
    plt.grid()
    plt.savefig("flow_real_loss.png")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processed_dir", type=str, default="data/processed",
        help="Normal kabul edilen .npy dosyalarının bulunduğu klasör"
    )
    parser.add_argument(
        "--max_segments", type=int, default=100000,
        help="Eğitimde kullanılacak maksimum segment sayısı"
    )
    args = parser.parse_args()
    train_flow_real(args.processed_dir, args.max_segments)
