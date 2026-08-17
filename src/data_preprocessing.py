import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class CKDLongitudinalDataset(Dataset):
    def __init__(self, sequences, targets, controls, delta_t, masks, phenotypes):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.controls = torch.tensor(controls, dtype=torch.float32)
        self.delta_t = torch.tensor(delta_t, dtype=torch.float32)
        self.masks = torch.tensor(masks, dtype=torch.float32)
        self.phenotypes = torch.tensor(phenotypes, dtype=torch.long)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {
            "x_seq": self.sequences[idx],       # [seq_len, num_state_features]
            "target_seq": self.targets[idx],   # [seq_len, num_targets] (eGFR, Creatinine, UACR)
            "u_seq": self.controls[idx],       # [seq_len, num_controls] (ACEi, SGLT2i, Diuretic)
            "delta_t": self.delta_t[idx],      # [seq_len, 1] (Irregular visit delta)
            "mask": self.masks[idx],           # [seq_len] (1 for real visits, 0 for padded)
            "phenotype": self.phenotypes[idx]  # Ground truth mode (0: Stable, 1: Moderate, 2: Rapid)
        }

class CKDDataPipeline:
    def __init__(self, data_path=r"c:\Users\sange\sem 5_Math\dataset\cric_longitudinal_trajectory_dataset.csv", max_seq_len=8):
        self.data_path = data_path
        self.max_seq_len = max_seq_len
        self.state_scaler = StandardScaler()
        self.control_scaler = StandardScaler()
        
        # Clinical Feature Definitions
        self.target_cols = ['egfr', 'serum_creatinine', 'uacr']
        self.vital_lab_cols = ['sbp', 'dbp', 'hba1c', 'hemoglobin', 'potassium', 'bmi', 'age_at_visit']
        self.static_cols = ['sex_male', 'race_black', 'diabetes_baseline', 'cvd_baseline']
        self.control_cols = ['acei_arb_active', 'sglt2i_active', 'diuretic_active']
        
        # All state features x(t)
        self.all_state_cols = self.target_cols + self.vital_lab_cols + self.static_cols
        self.phenotype_map = {'Stable': 0, 'Moderate': 1, 'Rapid': 2}

    def load_and_preprocess(self):
        print(f"Loading longitudinal data from: {self.data_path}")
        df = pd.read_csv(self.data_path)
        
        # Ensure correct ordering
        df = df.sort_values(by=['patient_id', 'visit_index']).reset_index(drop=True)
        
        # Patient-Level Train / Validation / Test Split (Prevents data leakage)
        unique_patients = df['patient_id'].unique()
        train_pats, test_pats = train_test_split(unique_patients, test_size=0.30, random_state=42)
        val_pats, test_pats = train_test_split(test_pats, test_size=0.50, random_state=42)
        
        df_train = df[df['patient_id'].isin(train_pats)].copy()
        df_val = df[df['patient_id'].isin(val_pats)].copy()
        df_test = df[df['patient_id'].isin(test_pats)].copy()
        
        # Fit Scalers on Train Set ONLY
        self.state_scaler.fit(df_train[self.all_state_cols])
        
        # Transform Splits
        for d in [df_train, df_val, df_test]:
            d[self.all_state_cols] = self.state_scaler.transform(d[self.all_state_cols])
            
        print(f"Dataset Split: {len(train_pats)} Train | {len(val_pats)} Val | {len(test_pats)} Test Patients")
        
        # Build 3D Tensor Sequences [Batch, Seq_Len, Features]
        train_data = self._build_sequences(df_train, train_pats)
        val_data = self._build_sequences(df_val, val_pats)
        test_data = self._build_sequences(df_test, test_pats)
        
        return train_data, val_data, test_data

    def _build_sequences(self, df_subset, patient_ids):
        sequences = []
        targets = []
        controls = []
        delta_ts = []
        masks = []
        phenotypes = []
        
        num_state_features = len(self.all_state_cols)
        num_targets = len(self.target_cols)
        num_controls = len(self.control_cols)

        for pat_id in patient_ids:
            pat_df = df_subset[df_subset['patient_id'] == pat_id]
            seq_len = min(len(pat_df), self.max_seq_len)
            
            # Arrays padded with zeros
            x_pad = np.zeros((self.max_seq_len, num_state_features), dtype=np.float32)
            t_pad = np.zeros((self.max_seq_len, num_targets), dtype=np.float32)
            u_pad = np.zeros((self.max_seq_len, num_controls), dtype=np.float32)
            dt_pad = np.zeros((self.max_seq_len, 1), dtype=np.float32)
            m_pad = np.zeros((self.max_seq_len,), dtype=np.float32)
            
            # Fill with actual patient visits
            x_pad[:seq_len] = pat_df[self.all_state_cols].values[:seq_len]
            t_pad[:seq_len] = pat_df[self.target_cols].values[:seq_len]
            u_pad[:seq_len] = pat_df[self.control_cols].values[:seq_len]
            dt_pad[:seq_len] = pat_df[['delta_t_months']].values[:seq_len]
            m_pad[:seq_len] = 1.0  # 1 for valid visit
            
            raw_phenotype = pat_df['progression_phenotype'].iloc[0]
            phenotype_code = self.phenotype_map.get(raw_phenotype, 0)
            
            sequences.append(x_pad)
            targets.append(t_pad)
            controls.append(u_pad)
            delta_ts.append(dt_pad)
            masks.append(m_pad)
            phenotypes.append(phenotype_code)
            
        return {
            "sequences": np.array(sequences),
            "targets": np.array(targets),
            "controls": np.array(controls),
            "delta_t": np.array(delta_ts),
            "masks": np.array(masks),
            "phenotypes": np.array(phenotypes)
        }

    def get_dataloaders(self, batch_size=32):
        train_dict, val_dict, test_dict = self.load_and_preprocess()
        
        train_ds = CKDLongitudinalDataset(**train_dict)
        val_ds = CKDLongitudinalDataset(**val_dict)
        test_ds = CKDLongitudinalDataset(**test_dict)
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        
        # Save Scaler for later model inference & dashboard
        scaler_dir = r"c:\Users\sange\sem 5_Math\models"
        os.makedirs(scaler_dir, exist_ok=True)
        scaler_path = os.path.join(scaler_dir, "state_scaler.pkl")
        with open(scaler_path, "wb") as f:
            pickle.dump(self.state_scaler, f)
        print(f"Saved state scaler to {scaler_path}")
        
        return train_loader, val_loader, test_loader

if __name__ == "__main__":
    pipeline = CKDDataPipeline()
    train_loader, val_loader, test_loader = pipeline.get_dataloaders(batch_size=32)
    
    # Inspect a single batch
    sample_batch = next(iter(train_loader))
    print("\n--- SAMPLE BATCH TENSOR SHAPES ---")
    print(f"State Sequences x(t) [Batch, Seq_Len, Features] : {sample_batch['x_seq'].shape}")
    print(f"Target States [Batch, Seq_Len, Targets]          : {sample_batch['target_seq'].shape}")
    print(f"Control Interventions u(t) [Batch, Seq_Len, Meds]: {sample_batch['u_seq'].shape}")
    print(f"Time Delta dt [Batch, Seq_Len, 1]                : {sample_batch['delta_t'].shape}")
    print(f"Sequence Mask [Batch, Seq_Len]                   : {sample_batch['mask'].shape}")
    print(f"Phenotype Labels [Batch]                         : {sample_batch['phenotype'].shape}")
    print("\nPhase 1 Data Preprocessing & PyTorch Loader COMPLETE & VERIFIED!")
