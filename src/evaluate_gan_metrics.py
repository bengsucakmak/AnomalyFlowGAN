# src/evaluate_gan_metrics.py

import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import rbf_kernel

# Ayarlar
SEQ_LEN      = 200
NUM_CHANNELS = 4
Z_DIM        = 100
HIDDEN_DIM   = 128
N_SAMPLES    = 5000   # Gerçek ve sahte veri sayısı
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Generator Tanımı ---
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

# --- Gerçek Segmentleri Yükle ---
def load_real(path="data/processed", n=N_SAMPLES):
    segs = []
    for fname in sorted(os.listdir(path)):
        if not fname.endswith(".npy"):
            continue
        arr = np.load(os.path.join(path, fname))
        if arr.ndim != 3 or arr.shape[2] < NUM_CHANNELS:
            continue
        arr = arr[:, :, :NUM_CHANNELS]
        for sample in arr:
            if sample.shape[0] >= SEQ_LEN:
                segs.append(sample[:SEQ_LEN])
            if len(segs) >= n:
                return np.stack(segs)
    return np.stack(segs)

# --- Sahte Segmentleri Üret ---
def load_fake(model_path="generator.pth", n=N_SAMPLES):
    G = Generator().to(DEVICE)
    G.load_state_dict(torch.load(model_path, map_location=DEVICE))
    G.eval()
    zs = torch.randn(n, Z_DIM, device=DEVICE)
    with torch.no_grad():
        fake = G(zs).cpu().numpy()
    return fake

# --- MMD Hesaplama ---
def compute_mmd(x, y, gamma=None):
    if gamma is None:
        gamma = 1 / x.shape[1]
    Kxx = rbf_kernel(x, x, gamma=gamma)
    Kyy = rbf_kernel(y, y, gamma=gamma)
    Kxy = rbf_kernel(x, y, gamma=gamma)
    return Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()

def main():
    # 1) Verileri yükle
    real = load_real()
    fake = load_fake()
    print(f"[+] Real örnek: {real.shape}, Fake örnek: {fake.shape}")

    # 2) Düzleştir
    real_flat = real.reshape(real.shape[0], -1)
    fake_flat = fake.reshape(fake.shape[0], -1)

    # 3) MMD Hesapla
    mmd = compute_mmd(real_flat, fake_flat)
    print(f"\n🔹 MMD (RBF kernel): {mmd:.6f}")

    # 4) İki-örnek testi: LogisticRegression AUC/PR
    X = np.vstack([real_flat, fake_flat])
    y = np.hstack([np.zeros(len(real_flat)), np.ones(len(fake_flat))])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    probs = clf.predict_proba(X_test)[:,1]
    roc_auc = roc_auc_score(y_test, probs)
    pr_auc  = average_precision_score(y_test, probs)
    print(f"🔹 Classification AUC: ROC = {roc_auc:.4f}, PR = {pr_auc:.4f}")

    # 5) Kanal bazında Mean & Std farklar
    print("\nChannel | real_mean | fake_mean | mean_diff | real_std | fake_std | std_diff")
    for c in range(NUM_CHANNELS):
        rm = real[:,:,c].mean()
        fm = fake[:,:,c].mean()
        rs = real[:,:,c].std()
        fs = fake[:,:,c].std()
        print(f"  Ch{c+1}   {rm:8.4f}    {fm:8.4f}    {abs(rm-fm):8.4f}    {rs:8.4f}    {fs:8.4f}    {abs(rs-fs):8.4f}")

if __name__ == "__main__":
    main()
