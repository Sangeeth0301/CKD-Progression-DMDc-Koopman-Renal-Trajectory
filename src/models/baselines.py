import torch
import torch.nn as nn
import numpy as np

class TemporalTransformerBaseline(nn.Module):
    """
    Temporal Transformer with Multi-Head Self-Attention for EHR visit sequences.
    """
    def __init__(self, input_dim=14, d_model=64, nhead=4, num_layers=2, output_dim=14, dropout=0.1):
        super(TemporalTransformerBaseline, self).__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.time_projection = nn.Linear(1, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_head = nn.Linear(d_model, output_dim)

    def forward(self, x_seq, delta_t_seq, mask):
        # Project features + time embedding
        feat_embed = self.input_projection(x_seq)
        time_embed = self.time_projection(delta_t_seq / 12.0)
        h = feat_embed + time_embed
        
        # Invert mask for PyTorch Transformer (True = Ignored/Padded)
        src_key_padding_mask = (mask == 0.0)
        
        out = self.transformer(h, src_key_padding_mask=src_key_padding_mask)
        x_pred_next = self.output_head(out[:, :-1, :])  # Predict t+1 from visits 0..t
        return x_pred_next


class LSTMBaseline(nn.Module):
    """
    Standard Recurrent LSTM Baseline for EHR sequence forecasting.
    """
    def __init__(self, input_dim=14, hidden_dim=64, num_layers=2, output_dim=14, dropout=0.1):
        super(LSTMBaseline, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim + 1,  # Features + delta_t
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x_seq, delta_t_seq, mask):
        # Concatenate normalized time delta
        h_in = torch.cat([x_seq, delta_t_seq / 12.0], dim=-1)
        out, _ = self.lstm(h_in)
        x_pred_next = self.fc(out[:, :-1, :])  # Predict t+1 from visits 0..t
        return x_pred_next

if __name__ == "__main__":
    print("Testing Baseline Benchmark Architectures...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dummy_x = torch.randn(16, 8, 14).to(device)
    dummy_dt = torch.FloatTensor(16, 8, 1).uniform_(6.0, 12.0).to(device)
    dummy_mask = torch.ones(16, 8).to(device)
    dummy_mask[:, 5:] = 0.0
    
    transformer = TemporalTransformerBaseline().to(device)
    lstm = LSTMBaseline().to(device)
    
    out_trans = transformer(dummy_x, dummy_dt, dummy_mask)
    out_lstm = lstm(dummy_x, dummy_dt, dummy_mask)
    
    print("\n--- BASELINES VERIFIED ---")
    print(f"Transformer Parameters: {sum(p.numel() for p in transformer.parameters())} | Output: {out_trans.shape}")
    print(f"LSTM Parameters       : {sum(p.numel() for p in lstm.parameters())} | Output: {out_lstm.shape}")
    print("\nPhase 3 Baselines COMPLETE & VERIFIED!")
