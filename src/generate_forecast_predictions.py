import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformer_fc import TransformerForecast
from utils import set_seed

BATCH_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def segment_time_series(data, window_size=200, step=200):
    segments = []
    for sample in data:
        for start in range(0, sample.shape[0] - window_size + 1, step):
            seg = sample[start:start + window_size]
            segments.append(seg)
    return np.array(segments)

class MultiNpyDataset(Dataset):
    def __init__(self, real_files, fake_file, window_size=200, step=200):
        real_segments = []
        for f in real_files:
            arr = np.load(f)
            if arr.ndim == 3 and arr.shape[2] > 4:
                arr = arr[:, :, :4]
            segs = segment_time_series(arr, window_size, step)
            real_segments.append(segs)
        real = np.vstack(real_segments)

        fake = np.load(fake_file)

        data = np.vstack([real, fake])
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx, :-50, :]  # geçmiş 150 zaman adımı
        y = self.data[idx, -50:, :]  # gelecek 50 zaman adımı
        return x, y

def generate_predictions(real_files, fake_file, model_path="transformer_fc.pth"):
    set_seed()
    dataset = MultiNpyDataset(real_files, fake_file)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = TransformerForecast().to(DEVICE)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    all_preds = []
    all_trues = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            pred = model(x).cpu().numpy()
            all_preds.append(pred)
            all_trues.append(y.numpy())

    preds = np.vstack(all_preds)
    trues = np.vstack(all_trues)

    print(f"✅ Tahminler üretildi: {preds.shape}, Gerçekler: {trues.shape}")

    np.save("forecast_pred.npy", preds)
    np.save("forecast_true.npy", trues)
    print("✅ 'forecast_pred.npy' ve 'forecast_true.npy' dosyaları kaydedildi.")

if __name__ == "__main__":
    real_files = [
        "data/processed/1st_test_arr.npy",
        "data/processed/2nd_test_arr.npy",
        "data/processed/3rd_test_arr.npy"
    ]
    fake_file = "generated_data.npy"  # Dosyanın gerçek konumuna göre değiştir

    generate_predictions(real_files, fake_file)
