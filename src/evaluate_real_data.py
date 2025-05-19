# src/filter_sensor_real_anomalies.py

import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

SEQ_LEN      = 200
NUM_CHANNELS = 4

def load_and_segment(processed_dir, max_segments=None):
    """
    data/processed içindeki .npy dosyaları (Ni,20480,Ci) üzerinden
    200’lük kaydırmalı segmentler üretir ve
    rastgele max_segments ile sınırlarsa onu uygular.
    """
    segments = []
    for fname in sorted(os.listdir(processed_dir)):
        if not fname.endswith(".npy"):
            continue
        arr = np.load(os.path.join(processed_dir, fname))
        if arr.ndim != 3 or arr.shape[2] < NUM_CHANNELS:
            continue
        arr = arr[:, :, :NUM_CHANNELS]
        n_win = arr.shape[1] // SEQ_LEN
        for sample in arr:
            for w in range(n_win):
                seg = sample[w*SEQ_LEN:(w+1)*SEQ_LEN]
                if seg.shape == (SEQ_LEN, NUM_CHANNELS) and np.isfinite(seg).all():
                    segments.append(seg)
    if not segments:
        raise RuntimeError("Hiç segment bulunamadı!")
    total = len(segments)

    # Rastgele örnekleme
    if max_segments and total > max_segments:
        idx = np.random.choice(total, max_segments, replace=False)
        segments = [segments[i] for i in idx]
        print(f"▶️ {max_segments} segment rastgele seçildi (raw: {total})")
    else:
        print(f"▶️ Toplam segment: {total}")

    data = np.stack(segments)  # (N,200,4)
    return data

def main(processed_dir, score_csv, percentile, max_segments):
    # 1) Segmentleri yükle (aynı seed eklemek istersen burada yapabilirsiniz)
    print("📂 Gerçek segmentler yükleniyor ve pencereleniyor…")
    segments = load_and_segment(processed_dir, max_segments)  # (N,200,4)
    N = segments.shape[0]

    # 2) Skorları yükle
    scores = np.loadtxt(score_csv, delimiter=",")
    assert scores.shape[0] == N, (
        f"Segment sayısı ({N}) ile skor sayısı ({scores.shape[0]}) eşleşmiyor!"
    )

    # 3) Eşik değeri belirle
    thr = np.percentile(scores, percentile)
    print(f"⚡ {percentile}. percentile eşik değeri: {thr:.2f}")

    # 4) Histogram
    plt.figure(figsize=(8,4))
    plt.hist(scores, bins=100, alpha=0.8)
    plt.axvline(x=thr, color='r', linestyle='--', label=f"{percentile}th perc. = {thr:.1f}")
    plt.title("Gerçek Verilerin Anomaly Score Dağılımı")
    plt.xlabel("Score (-log_prob)")
    plt.ylabel("Sayı")
    plt.legend()
    plt.tight_layout()
    plt.savefig("sensor_score_histogram.png")
    plt.show()
    print("   Histogram kaydedildi: sensor_score_histogram.png")

    # 5) Anomali/Normal ayır
    anomaly_idx = scores > thr
    normals_idx = ~anomaly_idx
    anomalies = segments[anomaly_idx]
    normals   = segments[normals_idx]
    print(f"✅ Anomali segment sayısı: {anomalies.shape[0]}")
    print(f"✅ Normal segment sayısı: {normals.shape[0]}")

    # 6) Kaydet
    np.save("sensor_anomalies.npy", anomalies)
    np.save("sensor_normals.npy", normals)
    print("   sensor_anomalies.npy ve sensor_normals.npy oluşturuldu.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", required=True,
                        help="data/processed klasörü yolu")
    parser.add_argument("--score_csv", default="sensor_real_scores.csv",
                        help="evaluate_real_data.py çıktısı skor CSV dosyası")
    parser.add_argument("--percentile", type=float, default=95.0,
                        help="Anomali eşikleme için percentil (örn. 95)")
    parser.add_argument("--max_segments", type=int, default=100000,
                        help="Skor hesaplamada kullanılan segment sayısıyla eşleşmeli")
    args = parser.parse_args()

    main(args.processed_dir, args.score_csv, args.percentile, args.max_segments)

