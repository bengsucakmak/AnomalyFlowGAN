import os
import numpy as np

RAW_ROOT = 'data/raw'
PROC_ROOT = 'data/processed'
os.makedirs(PROC_ROOT, exist_ok=True)

# Set adı → kanal sayısı eşlemesi
CHANNEL_MAP = {
    '1st_test': 8,
    '2nd_test': 4,
    '3rd_test': 4,
}

def load_ascii_file(path, channels):
    """
    Her satırı whitespace ile ayırarak float olarak oku.
    Dosyada satır sayısı = 20480, sütun sayısı = channels.
    """
    data = np.loadtxt(path)
    assert data.shape == (20480, channels), \
        f"{path} shape mismatch: {data.shape}"
    return data  # shape: (20480, channels)

for set_name, ch_count in CHANNEL_MAP.items():
    folder = os.path.join(RAW_ROOT, set_name)
    all_segments = []
    # Dosyaları kronolojik sırada işle
    files = sorted(os.listdir(folder))
    for fname in files:
        path = os.path.join(folder, fname)
        seg = load_ascii_file(path, ch_count)
        all_segments.append(seg)
    arr = np.stack(all_segments)  # shape: (N_files, 20480, channels)
    out_path = os.path.join(PROC_ROOT, f'{set_name}_arr.npy')
    np.save(out_path, arr)
    print(f"{set_name}: {arr.shape} → kaydedildi → {out_path}")

