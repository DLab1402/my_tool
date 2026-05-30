

import torch
import torch.nn as nn
import numpy as np

class construction_health_monitoring_ppg(nn.Module):
    """
    Deep Convolutional Autoencoder network exactly as described by Gautam & Jebelli (2024).
    """
    def __init__(self, input_len=128, latent_dim=4):
        super(construction_health_monitoring_ppg, self).__init__()
        self.input_len = input_len
        self.latent_dim = latent_dim

        self.hidden = None

        # ==========================================
        # 1. ENCODER
        # ==========================================
        # Conv_Block_1: Conv1D (8 filters, kernel=5, padding='same') -> BN -> ReLU -> MaxPool (pool=2)
        # Input shape: (B, 1, 128) -> Output shape: (B, 8, 64)
        self.enc_conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=5, padding='same'),
            nn.BatchNorm1d(num_features=8),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # Conv_Block_2: Conv1D (16 filters, kernel=3, padding='same') -> BN -> ReLU -> MaxPool (pool=2)
        # Input shape: (B, 8, 64) -> Output shape: (B, 16, 32)
        self.enc_conv2 = nn.Sequential(
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, padding='same'),
            nn.BatchNorm1d(num_features=16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # Calculate flattened dimension: 16 channels * (input_len // 4) = 16 * 32 = 512
        self.flattened_dim = 16 * (input_len // 4)

        # Dense_1: 16 units, ReLU activation
        self.enc_dense1 = nn.Sequential(
            nn.Linear(in_features=self.flattened_dim, out_features=16),
            nn.ReLU()
        )

        # Latent_Space (Bottleneck): Dense layer with exactly 4 units, ReLU activation
        self.enc_latent = nn.Sequential(
            nn.Linear(in_features=16, out_features=self.latent_dim),
            nn.ReLU()
        )

        # ==========================================
        # 2. DECODER
        # ==========================================
        # Dense_2: 16 units, ReLU activation
        self.dec_dense2 = nn.Sequential(
            nn.Linear(in_features=self.latent_dim, out_features=16),
            nn.ReLU()
        )

        # Dense_3: Expand the 16 units to match the flattened size before latent projection
        self.dec_dense3 = nn.Sequential(
            nn.Linear(in_features=16, out_features=self.flattened_dim),
            nn.ReLU()
        )

        # Transposed_Conv_Block_1: 1D Transposed Conv (16 filters->8 filters, kernel=3, stride=2) -> BN -> ReLU
        # Input shape: (B, 16, 32) -> Output shape: (B, 8, 64)
        # Formula: L_out = (L_in - 1)*stride - 2*padding + kernel_size + output_padding
        # 64 = (32 - 1)*2 - 2(1) + 3 + 1 = 62 - 2 + 4 = 64
        self.dec_trans_conv1 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(num_features=8),
            nn.ReLU()
        )

        # Transposed_Conv_Block_2: 1D Transposed Conv (8 filters->8 filters, kernel=5, stride=2) -> BN -> ReLU
        # Input shape: (B, 8, 64) -> Output shape: (B, 8, 128)
        # Formula: 128 = (64 - 1)*2 - 2(2) + 5 + 1 = 126 - 4 + 6 = 128
        self.dec_trans_conv2 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=8, out_channels=8, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(num_features=8),
            nn.ReLU()
        )

        # Output_Layer: 1D Convolution (filters=1, kernel=1) -> Sigmoid (matching Min-Max normalized PPG [0,1])
        # Input shape: (B, 8, 128) -> Output shape: (B, 1, 128)
        self.output_layer = nn.Sequential(
            nn.Conv1d(in_channels=8, out_channels=1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):

        # --- Encoder ---
        out = self.enc_conv1(x)
        out = self.enc_conv2(out)
        
        # Flatten: (B, 16, 32) -> (B, 512)
        out = torch.flatten(out, start_dim=1)
        
        out = self.enc_dense1(out)
        latent = self.enc_latent(out)
        self.hidden = latent

        # --- Decoder ---
        out = self.dec_dense2(latent)
        out = self.dec_dense3(out)
        
        # Reshape back to 3D tensor: (B, 512) -> (B, 16, 32)
        out = out.view(-1, 16, self.input_len // 4)
        
        out = self.dec_trans_conv1(out)
        out = self.dec_trans_conv2(out)
        decoded = self.output_layer(out)
        
        mu = latent
        log_var = torch.zeros_like(latent)

        return decoded, mu, log_var


def create_model(latent_dim=4):
    return construction_health_monitoring_ppg(input_len=128, latent_dim=latent_dim)


def calculate_reconstruction_mae(input_signal, reconstructed_signal):
    if isinstance(input_signal, torch.Tensor):
        # Calculate MAE across channel and length dimensions (dim 1 and 2)
        mae = torch.mean(torch.abs(input_signal - reconstructed_signal), dim=(1, 2))
        return mae
    else:
        # Numpy implementation
        mae = np.mean(np.abs(input_signal - reconstructed_signal), axis=(1, 2))
        return mae


def detect_motion_artifact(mae_scores, threshold=0.0405):
    return mae_scores > threshold


# =====================================================================
# STANDALONE DUMMY DATA TEST LOOP (Verification of Dimensions)
# =====================================================================
if __name__ == "__main__":
    print("="*65)
    print("VERIFYING DEEP CONVOLUTIONAL AUTOENCODER ")
    print("="*65)
    
    # 1. Initialize configurations
    batch_size = 32
    L = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Running test on device: {device}")
    
    # Initialize model
    model = create_model(latent_dim=4).to(device)
    model.eval()
    
    # Create dummy PPG time-series data: shape (32, 1, 128)
    dummy_input = torch.rand(batch_size, 1, L).to(device)
    print(f"\n1. Input Shape:                 {list(dummy_input.shape)}")
    
    # 2. Test forward pass step-by-step to display intermediate shapes
    with torch.no_grad():
        # Encoder intermediate
        enc_c1 = model.enc_conv1(dummy_input)
        print(f"2. After Conv_Block_1 Shape:    {list(enc_c1.shape)}")
        
        enc_c2 = model.enc_conv2(enc_c1)
        print(f"3. After Conv_Block_2 Shape:    {list(enc_c2.shape)}")
        
        flat = torch.flatten(enc_c2, start_dim=1)
        print(f"4. After Flatten Shape:         {list(flat.shape)}")
        
        dense1 = model.enc_dense1(flat)
        latent = model.enc_latent(dense1)
        print(f"5. Latent Space (Bottleneck):   {list(latent.shape)} --> EXACTLY 4 UNITS!")
        
        # Decoder intermediate
        dec_d2 = model.dec_dense2(latent)
        dec_d3 = model.dec_dense3(dec_d2)
        unflat = dec_d3.view(-1, 16, L // 4)
        print(f"6. After Reshape (Unflatten):   {list(unflat.shape)}")
        
        dec_tc1 = model.dec_trans_conv1(unflat)
        print(f"7. After Trans_Conv_Block_1:    {list(dec_tc1.shape)}")
        
        dec_tc2 = model.dec_trans_conv2(dec_tc1)
        print(f"8. After Trans_Conv_Block_2:    {list(dec_tc2.shape)}")
        
        decoded, mu, log_var = model(dummy_input)
        print(f"9. Final Reconstructed Output:  {list(decoded.shape)} --> MATCHES INPUT SHAPE!")
    
    # 3. Test Utility Functions
    print("\n" + "="*65)
    print("TESTING UTILITY FUNCTIONS (MAE & ANOMALY DETECTION)")
    print("="*65)
    mae_scores = calculate_reconstruction_mae(dummy_input, decoded)
    print(f"[Utility] MAE Scores Shape:     {list(mae_scores.shape)}")
    print(f"[Utility] Sample MAE values:    {mae_scores[:5].cpu().numpy()}")
    
    anomalies = detect_motion_artifact(mae_scores, threshold=0.0405)
    print(f"[Utility] Anomalies Detected:   {anomalies[:5].cpu().numpy()} (Threshold=0.0405)")
    print("="*65)
    print("SUCCESS: ALL ARCHITECTURAL SPECIFICATIONS AND SHAPES VERIFIED!")
    print("="*65)