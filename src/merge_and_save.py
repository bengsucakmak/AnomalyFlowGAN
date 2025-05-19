import os
import numpy as np
import argparse

def merge_npy(files, out_file, num_channels=4):
    """
    Her .npy'yi num_channels kanala kırpar ve birleştirir.
    """
    arrays = []
    for f in files:
        print(f"📥 Yükleniyor: {f}")
        arr = np.load(f)
        if arr.ndim == 3 and arr.shape[2] >= num_channels:
            arr = arr[:, :, :num_channels]
        else:
            raise ValueError(f"Array {f} kanal sayısı uyumsuz: {arr.shape}")
        arrays.append(arr)
    merged = np.concatenate(arrays, axis=0)
    print(f"✅ Birleştirildi: shape {merged.shape}")
    np.save(out_file, merged)
    print(f"✅ Kaydedildi: {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True,
                        help="Birleştirilecek .npy dosyalarının klasörü")
    parser.add_argument("--pattern", default="_test_arr.npy",
                        help="Dosya adı desen filtresi")
    parser.add_argument("--out_file", default="sensor_windows.npy",
                        help="Çıktı dosyası adı")
    args = parser.parse_args()

    files = sorted(
        os.path.join(args.input_dir, f)
        for f in os.listdir(args.input_dir)
        if f.endswith(args.pattern)
    )
    merge_npy(files, args.out_file)

