import torch
import torch.nn as nn
import numpy as np


class PPG_CAE(nn.Module):
    def __init__(self, input_len=800, latent_dim=64):
        super(PPG_CAE, self).__init__()

        self.input_len = input_len
        self.latent_dim = latent_dim

        # =========================================================
        # ENCODER
        # =========================================================

        self.encoder = nn.Sequential(

            # -------------------------------------------------
            # BLOCK 1
            # (B,1,800) -> (B,16,400)
            # -------------------------------------------------
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ELU(),

            nn.Conv1d(16, 16, kernel_size=3, padding=1),
            nn.ELU(),

            nn.MaxPool1d(kernel_size=2, stride=2),

            # -------------------------------------------------
            # BLOCK 2
            # (B,16,400) -> (B,32,200)
            # -------------------------------------------------
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ELU(),

            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ELU(),

            nn.MaxPool1d(kernel_size=2, stride=2),

            # -------------------------------------------------
            # BLOCK 3
            # (B,32,200) -> (B,64,100)
            # -------------------------------------------------
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ELU(),

            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ELU(),

            nn.MaxPool1d(kernel_size=2, stride=2),

            # -------------------------------------------------
            # BLOCK 4
            # (B,64,100) -> (B,128,50)
            # -------------------------------------------------
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ELU(),

            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ELU(),

            nn.MaxPool1d(kernel_size=2, stride=2),

            # -------------------------------------------------
            # BLOCK 5
            # (B,128,50) -> (B,256,25)
            # -------------------------------------------------
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ELU(),

            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ELU(),

            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        # After 5 pools:
        # 800 -> 400 -> 200 -> 100 -> 50 -> 25

        self.feature_length = 25
        self.feature_channels = 256

        self.flattened_dim = self.feature_channels * self.feature_length

        # -----------------------------------------------------
        # LATENT SPACE
        # -----------------------------------------------------
        self.flatten = nn.Flatten()

        self.encoder_dense = nn.Linear(
            self.flattened_dim,
            latent_dim
        )

        # =========================================================
        # DECODER
        # =========================================================

        self.decoder_dense = nn.Linear(
            latent_dim,
            self.flattened_dim
        )

        self.decoder = nn.Sequential(

            # -------------------------------------------------
            # BLOCK 5
            # (B,256,25) -> (B,128,50)
            # -------------------------------------------------
            nn.Upsample(scale_factor=2, mode='nearest'),

            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ELU(),

            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.ELU(),

            # -------------------------------------------------
            # BLOCK 4
            # (B,128,50) -> (B,64,100)
            # -------------------------------------------------
            nn.Upsample(scale_factor=2, mode='nearest'),

            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ELU(),

            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.ELU(),

            # -------------------------------------------------
            # BLOCK 3
            # (B,64,100) -> (B,32,200)
            # -------------------------------------------------
            nn.Upsample(scale_factor=2, mode='nearest'),

            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ELU(),

            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ELU(),

            # -------------------------------------------------
            # BLOCK 2
            # (B,32,200) -> (B,16,400)
            # -------------------------------------------------
            nn.Upsample(scale_factor=2, mode='nearest'),

            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ELU(),

            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.ELU(),

            # -------------------------------------------------
            # BLOCK 1
            # (B,16,400) -> (B,16,800)
            # -------------------------------------------------
            nn.Upsample(scale_factor=2, mode='nearest'),

            nn.Conv1d(16, 16, kernel_size=3, padding=1),
            nn.ELU(),

            nn.Conv1d(16, 16, kernel_size=3, padding=1),
            nn.ELU(),

            # -------------------------------------------------
            # OUTPUT LAYER
            # -------------------------------------------------
            nn.Conv1d(16, 1, kernel_size=1)
        )

    def forward(self, x):

        # =====================================================
        # ENCODER
        # =====================================================
        x = self.encoder(x)

        x = self.flatten(x)

        latent = self.encoder_dense(x)

        # =====================================================
        # DECODER
        # =====================================================
        x = self.decoder_dense(latent)

        x = x.view(
            -1,
            self.feature_channels,
            self.feature_length
        )

        reconstructed = self.decoder(x)

        return reconstructed, latent


# ============================================================
# MODEL CREATION
# ============================================================

def create_model():
    return PPG_CAE(
        input_len=800,
        latent_dim=64
    )


# ============================================================
# LOSS
# ============================================================

def reconstruction_loss(input_signal, reconstructed_signal):

    criterion = nn.MSELoss()

    return criterion(
        reconstructed_signal,
        input_signal
    )


# ============================================================
# ANOMALY SCORE
# ============================================================

def calculate_reconstruction_error(
        input_signal,
        reconstructed_signal
):

    if isinstance(input_signal, torch.Tensor):

        mse = torch.mean(
            (input_signal - reconstructed_signal) ** 2,
            dim=(1, 2)
        )

        return mse

    else:

        mse = np.mean(
            (input_signal - reconstructed_signal) ** 2,
            axis=(1, 2)
        )

        return mse


# ============================================================
# THRESHOLD DETECTION
# ============================================================

def detect_anomaly(
        reconstruction_errors,
        threshold
):

    return reconstruction_errors > threshold


# ============================================================
# TRAINING THRESHOLD
# ============================================================

def compute_threshold(train_errors, n_std=3):

    mean = np.mean(train_errors)
    std = np.std(train_errors)

    threshold = mean + n_std * std

    return threshold


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("PPG CONVOLUTIONAL AUTOENCODER TEST")
    print("=" * 60)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = create_model().to(device)

    batch_size = 8

    dummy_input = torch.randn(
        batch_size,
        1,
        800
    ).to(device)

    with torch.no_grad():

        reconstructed, latent = model(dummy_input)

    print(f"Input Shape:          {dummy_input.shape}")
    print(f"Latent Shape:         {latent.shape}")
    print(f"Reconstructed Shape:  {reconstructed.shape}")

    errors = calculate_reconstruction_error(
        dummy_input,
        reconstructed
    )

    print(f"Reconstruction Error Shape: {errors.shape}")

    threshold = compute_threshold(
        errors.cpu().numpy(),
        n_std=3
    )

    anomalies = detect_anomaly(
        errors,
        threshold
    )

    print(f"Threshold: {threshold:.6f}")
    print(f"Anomalies: {anomalies}")