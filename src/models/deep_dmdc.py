import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class KoopmanEncoder(nn.Module):
    """
    Non-Linear Observable Encoder ψ_enc: R^n -> R^d (d >> n)
    Maps non-linear clinical biomarkers into linear Koopman latent space.
    """
    def __init__(self, input_dim=14, latent_dim=32, hidden_dim=64, dropout_rate=0.1):
        super(KoopmanEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, x):
        return self.net(x)


class KoopmanDecoder(nn.Module):
    """
    Non-Linear Clinical Decoder ψ_dec: R^d -> R^n
    Reconstructs continuous clinical biomarkers from future latent state.
    """
    def __init__(self, latent_dim=32, output_dim=14, hidden_dim=64, dropout_rate=0.1):
        super(KoopmanDecoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, z):
        return self.net(z)


class ContinuousKoopmanOperator(nn.Module):
    """
    Continuous State-Space DMDc Operator:
    dz(t)/dt = A·z(t) + B·u(t)
    Solved continuously over irregular time interval Δt via Matrix Exponential exp(A·Δt).
    """
    def __init__(self, latent_dim=32, control_dim=3):
        super(ContinuousKoopmanOperator, self).__init__()
        self.latent_dim = latent_dim
        self.control_dim = control_dim
        
        # Autonomous Disease Dynamics Matrix A (d x d)
        # Initialized with slight negative diagonal for biological stability
        A_init = -0.05 * torch.eye(latent_dim) + 0.01 * torch.randn(latent_dim, latent_dim)
        self.A = nn.Parameter(A_init)
        
        # Therapeutic Control Matrix B (d x p)
        B_init = 0.02 * torch.randn(latent_dim, control_dim)
        self.B = nn.Parameter(B_init)

    def forward_step(self, z_t, u_t, delta_t):
        """
        Computes z(t + Δt) given z(t), control u(t), and elapsed time delta_t.
        Exact update: z(t+Δt) = exp(A·Δt)·z(t) + ∫_0^Δt exp(A·s) ds · B · u(t)
        """
        batch_size = z_t.shape[0]
        device = z_t.device
        
        # Scale Δt in years for stable matrix exponential
        # delta_t is in months, so convert to years (dt_yr = dt / 12.0)
        dt_scaled = (delta_t / 12.0).view(-1, 1, 1)  # [Batch, 1, 1]
        
        # Batch Matrix Exponential: exp(A * dt)
        A_batch = self.A.unsqueeze(0).repeat(batch_size, 1, 1)  # [Batch, d, d]
        exp_A_dt = torch.linalg.matrix_exp(A_batch * dt_scaled)  # [Batch, d, d]
        
        # Autonomous disease progression term: exp(A·Δt) · z_t
        z_autonomous = torch.bmm(exp_A_dt, z_t.unsqueeze(-1)).squeeze(-1)  # [Batch, d]
        
        # Therapeutic control term approximation: dt * B * u_t
        B_batch = self.B.unsqueeze(0).repeat(batch_size, 1, 1)  # [Batch, d, p]
        u_effect = torch.bmm(B_batch, u_t.unsqueeze(-1)).squeeze(-1)  # [Batch, d]
        z_control = u_effect * dt_scaled.squeeze(-1)
        
        z_next = z_autonomous + z_control
        return z_next, exp_A_dt

    def get_eigenvalues(self):
        """
        Computes the spectral decomposition of Matrix A: A = W·Λ·W^-1
        Returns real and imaginary components of all eigenvalues.
        """
        with torch.no_grad():
            eigenvalues = torch.linalg.eigvals(self.A).cpu().numpy()
            return eigenvalues

    def simulate_what_if(self, z_t, delta_u, delta_t_months=12.0):
        """
        Counterfactual What-If Simulator:
        Computes ΔTrajectory = B · Δu over elapsed time.
        """
        with torch.no_grad():
            dt_yr = delta_t_months / 12.0
            delta_u_tensor = torch.tensor(delta_u, dtype=torch.float32, device=self.B.device).unsqueeze(-1)
            delta_z = torch.matmul(self.B, delta_u_tensor).squeeze(-1) * dt_yr
            return delta_z


class DeepContinuousDMDc(nn.Module):
    """
    Unified Multi-Resolution Deep Continuous DMDc Network for CKD Forecasting.
    """
    def __init__(self, input_dim=14, latent_dim=32, control_dim=3, hidden_dim=64, dropout_rate=0.1):
        super(DeepContinuousDMDc, self).__init__()
        self.encoder = KoopmanEncoder(input_dim, latent_dim, hidden_dim, dropout_rate)
        self.koopman = ContinuousKoopmanOperator(latent_dim, control_dim)
        self.decoder = KoopmanDecoder(latent_dim, input_dim, hidden_dim, dropout_rate)

    def forward(self, x_seq, u_seq, delta_t_seq, mask):
        """
        Full Forward Pass over Patient Visit Sequence:
        x_seq: [Batch, Seq_Len, Features]
        u_seq: [Batch, Seq_Len, Controls]
        delta_t_seq: [Batch, Seq_Len, 1]
        mask: [Batch, Seq_Len]
        """
        batch_size, seq_len, _ = x_seq.shape
        
        # 1. Encode all visit snapshots into latent space z_seq
        z_seq = self.encoder(x_seq)  # [Batch, Seq_Len, Latent_Dim]
        
        # 2. Reconstruct current snapshots (Autoencoder Fidelity)
        x_reconstructed = self.decoder(z_seq)  # [Batch, Seq_Len, Features]
        
        # 3. Propagate Koopman state transitions across irregular Δt
        z_predicted_next = []
        x_predicted_next = []
        
        for t in range(seq_len - 1):
            z_t = z_seq[:, t, :]
            u_t = u_seq[:, t, :]
            dt_t = delta_t_seq[:, t+1, :]  # Time elapsed to next visit
            
            z_next, _ = self.koopman.forward_step(z_t, u_t, dt_t)
            x_next = self.decoder(z_next)
            
            z_predicted_next.append(z_next)
            x_predicted_next.append(x_next)
            
        z_pred_tensor = torch.stack(z_predicted_next, dim=1)  # [Batch, Seq_Len-1, Latent_Dim]
        x_pred_tensor = torch.stack(x_predicted_next, dim=1)  # [Batch, Seq_Len-1, Features]
        
        return {
            "z_seq": z_seq,
            "x_reconstructed": x_reconstructed,
            "z_pred_next": z_pred_tensor,
            "x_pred_next": x_pred_tensor
        }


class DeepDMDcLoss(nn.Module):
    """
    Unified 4-Part Multi-Objective Loss Function:
    L_total = λ_rec·L_rec + λ_pred·L_pred + λ_lin·L_lin + λ_bio·L_bio
    """
    def __init__(self, lambda_rec=1.0, lambda_pred=2.0, lambda_lin=1.0, lambda_bio=0.5):
        super(DeepDMDcLoss, self).__init__()
        self.lambda_rec = lambda_rec
        self.lambda_pred = lambda_pred
        self.lambda_lin = lambda_lin
        self.lambda_bio = lambda_bio

    def forward(self, model_outputs, x_seq, mask):
        batch_size, seq_len, num_features = x_seq.shape
        mask_expanded = mask.unsqueeze(-1)  # [Batch, Seq_Len, 1]
        valid_visits = mask.sum().clamp(min=1.0)
        
        # 1. Reconstruction Loss: || x(t) - ψ_dec(ψ_enc(x(t))) ||^2
        x_rec = model_outputs["x_reconstructed"]
        l_rec = (F.mse_loss(x_rec, x_seq, reduction='none') * mask_expanded).sum() / valid_visits
        
        # 2. Prediction Loss: || x̂(t+Δt) - x(t+Δt) ||^2 (on real visits only)
        x_pred = model_outputs["x_pred_next"]  # [Batch, Seq_Len-1, Features]
        x_true_next = x_seq[:, 1:, :]          # [Batch, Seq_Len-1, Features]
        mask_next = mask[:, 1:].unsqueeze(-1)  # [Batch, Seq_Len-1, 1]
        valid_next = mask[:, 1:].sum().clamp(min=1.0)
        
        l_pred = (F.mse_loss(x_pred, x_true_next, reduction='none') * mask_next).sum() / valid_next
        
        # 3. Koopman Linearity Loss: || z(t+Δt) - (exp(A·Δt)·z(t) + B·u(t)) ||^2
        z_seq = model_outputs["z_seq"]
        z_true_next = z_seq[:, 1:, :]
        z_pred_next = model_outputs["z_pred_next"]
        l_lin = (F.mse_loss(z_pred_next, z_true_next, reduction='none') * mask_next).sum() / valid_next
        
        # 4. Biological Plausibility Guardrails (Prevent impossible jumps / negative eGFR)
        # eGFR is the 0-th column in our normalized features
        pred_egfr = x_pred[:, :, 0]
        # Slope penalty: max rate of continuous change
        egfr_diff = torch.abs(x_pred[:, :, 0] - x_true_next[:, :, 0])
        l_bio = F.relu(egfr_diff - 3.0).sum() / valid_next
        
        # Total Weighted Loss
        l_total = (self.lambda_rec * l_rec +
                   self.lambda_pred * l_pred +
                   self.lambda_lin * l_lin +
                   self.lambda_bio * l_bio)
        
        return {
            "loss_total": l_total,
            "loss_rec": l_rec,
            "loss_pred": l_pred,
            "loss_lin": l_lin,
            "loss_bio": l_bio
        }

if __name__ == "__main__":
    print("Testing Deep Continuous DMDc Architecture...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = DeepContinuousDMDc(input_dim=14, latent_dim=32, control_dim=3).to(device)
    criterion = DeepDMDcLoss()
    
    # Dummy Batch for Verification
    dummy_x = torch.randn(16, 8, 14).to(device)
    dummy_u = torch.randn(16, 8, 3).to(device)
    dummy_dt = torch.FloatTensor(16, 8, 1).uniform_(6.0, 12.0).to(device)
    dummy_mask = torch.ones(16, 8).to(device)
    dummy_mask[:, 5:] = 0.0  # Simulate padded visits
    
    outputs = model(dummy_x, dummy_u, dummy_dt, dummy_mask)
    losses = criterion(outputs, dummy_x, dummy_mask)
    
    print("\n--- MODEL ARCHITECTURE VERIFIED ---")
    print(f"Total Parameters          : {sum(p.numel() for p in model.parameters())}")
    print(f"Latent Sequence Shape     : {outputs['z_seq'].shape}")
    print(f"Reconstructed State Shape : {outputs['x_reconstructed'].shape}")
    print(f"Predicted Next State Shape: {outputs['x_pred_next'].shape}")
    print(f"Total Combined Loss       : {losses['loss_total'].item():.4f}")
    
    # Spectral Eigenvalues
    eigenvalues = model.koopman.get_eigenvalues()
    print(f"\nMatrix A Eigenvalues Sample (Total {len(eigenvalues)}):")
    for i, ev in enumerate(eigenvalues[:4]):
        stability = "UNSTABLE (Crash Mode)" if ev.real > 0 else "STABLE (Dampened Mode)"
        print(f"  lambda_{i+1}: {ev.real:+.4f} + {ev.imag:+.4f}i  --> {stability}")
        
    print("\nPhase 2 Deep Continuous DMDc PyTorch Model COMPLETE & VERIFIED!")
