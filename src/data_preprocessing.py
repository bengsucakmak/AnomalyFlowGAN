import os
import argparse
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import scipy.io as sio

def load_mat(file_path: str) -> np.ndarray:
    mat = sio.loadmat(file_path)
    for key, val in mat.items():
        if isinstance(val, np.ndarray):
            return val
    raise ValueError(f"No numpy array found in {file_path}")

def load_series(file_path: str) -> np.ndarray:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".mat":
        return load_mat(file_path)
    elif ext in [".csv", ".txt"]:
        return np.loadtxt(file_path, delimiter=',')
    else:
        raise ValueError(f"Unsupported extension: {ext}")

def preprocess_and_segment(input_dir: str, output_dir: str, seq_len: int = 200):
    os.makedirs(output_dir, exist_ok=True)
    scaler = MinMaxScaler()
    for fname in os.listdir(input_dir):
        path = os.path.join(input_dir, fname)
        try:
            series = load_series(path)
        except ValueError:
            continue
        if series.ndim == 2 and series.shape[1] > 1:
            series = series[:, 0]
        series = scaler.fit_transform(series.reshape(-1, 1)).flatten()
        n_segments = (len(series) - seq_len) // seq_len
        for i in range(n_segments):
            seg = series[i * seq_len:(i + 1) * seq_len]
            out_path = os.path.join(
                output_dir,
                f"{os.path.splitext(fname)[0]}_{i}.npz"
            )
            np.savez_compressed(out_path, data=seg.astype(np.float32))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seq_len", type=int, default=200)
    args = parser.parse_args()
    preprocess_and_segment(args.input_dir, args.output_dir, args.seq_len)
