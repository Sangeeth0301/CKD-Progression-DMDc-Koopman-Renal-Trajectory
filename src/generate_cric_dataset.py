import os
import numpy as np
import pandas as pd

def generate_synthetic_cric_dataset(
    num_patients=1000,
    min_visits=3,
    max_visits=10,
    random_seed=42,
    output_csv_path="data/raw/cric_longitudinal.csv"
):
    np.random.seed(random_seed)
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    records = []

    for pat_idx in range(1, num_patients + 1):
        patient_id = f"CRIC_PAT_{pat_idx:04d}"
        
        # 1. Static Demographics & Baseline Characteristics
        age_base = int(np.random.normal(58, 11))
        age_base = max(25, min(85, age_base))
        sex = np.random.choice([1, 0], p=[0.55, 0.45])  # 1: Male, 0: Female
        race_black = np.random.choice([1, 0], p=[0.40, 0.60])
        has_diabetes = np.random.choice([1, 0], p=[0.48, 0.52])
        has_cvd = np.random.choice([1, 0], p=[0.32, 0.68])
        
        # Patient Progression Phenotype (Latent True Eigenvalue Mode)
        # 0: Stable / Slow (60%), 1: Moderate (25%), 2: Rapid Collapse Progressor (15%)
        progression_phenotype = np.random.choice(
            ['Stable', 'Moderate', 'Rapid'],
            p=[0.60, 0.25, 0.15]
        )
        
        if progression_phenotype == 'Stable':
            base_decline_rate = np.random.uniform(0.5, 1.8)   # mL/min/year
        elif progression_phenotype == 'Moderate':
            base_decline_rate = np.random.uniform(2.0, 4.5)   # mL/min/year
        else: # Rapid Progressor (Severe)
            base_decline_rate = np.random.uniform(5.5, 12.0)  # mL/min/year

        # Baseline Target Biomarkers (eGFR Stage 2 to 4)
        base_egfr = np.random.normal(48, 14)
        base_egfr = max(18.0, min(75.0, base_egfr))
        
        # Initial Serum Creatinine (inversely related to eGFR via simplified CKD-EPI)
        base_creatinine = 1.0 + (60.0 - base_egfr) * 0.035 + np.random.normal(0, 0.15)
        base_creatinine = max(0.7, base_creatinine)
        
        # Initial Urine Albumin-to-Creatinine Ratio (UACR in mg/g)
        if has_diabetes:
            base_uacr = np.random.exponential(350) + 50
        else:
            base_uacr = np.random.exponential(120) + 15
            
        base_sbp = np.random.normal(136, 14)
        base_dbp = np.random.normal(82, 9)
        base_hba1c = np.random.normal(7.4, 1.2) if has_diabetes else np.random.normal(5.5, 0.4)
        base_hemoglobin = np.random.normal(12.8, 1.4)
        base_potassium = np.random.normal(4.3, 0.35)
        base_bmi = np.random.normal(29.5, 5.0)

        # Generate Longitudinal Follow-Up Visits (Irregular time delta Δt)
        num_visits = np.random.randint(min_visits, max_visits + 1)
        current_months = 0.0
        current_egfr = base_egfr
        current_creatinine = base_creatinine
        current_uacr = base_uacr
        current_sbp = base_sbp
        current_dbp = base_dbp
        
        # Active Medications (Controls u_t)
        active_acei_arb = np.random.choice([1, 0], p=[0.70, 0.30])
        active_sglt2i = np.random.choice([1, 0], p=[0.35, 0.65])
        active_diuretic = np.random.choice([1, 0], p=[0.60, 0.40])
        
        for visit_num in range(num_visits):
            if visit_num == 0:
                delta_t_months = 0.0
            else:
                # Irregular follow-up: roughly 6 to 14 months apart
                delta_t_months = np.random.uniform(5.5, 13.5)
                current_months += delta_t_months
                
                # Time in years for rate calculation
                delta_years = delta_t_months / 12.0
                
                # Therapeutic slowdown effect (Matrix B·u)
                therapeutic_protection = 0.0
                if active_acei_arb:
                    therapeutic_protection += 0.8  # Saves 0.8 mL/min/year
                if active_sglt2i:
                    therapeutic_protection += 1.4  # Saves 1.4 mL/min/year
                if current_sbp < 130:
                    therapeutic_protection += 0.5  # SBP control bonus
                    
                net_decline_per_year = max(0.1, base_decline_rate - therapeutic_protection)
                
                # Update eGFR with biological noise
                egfr_drop = net_decline_per_year * delta_years + np.random.normal(0, 0.8)
                current_egfr = max(5.0, current_egfr - egfr_drop)
                
                # Update Creatinine (inverse relationship)
                current_creatinine = max(0.6, current_creatinine + (egfr_drop * 0.038) + np.random.normal(0, 0.08))
                
                # Update UACR
                uacr_change = (egfr_drop * 15.0) - (25.0 if active_sglt2i else 0.0) + np.random.normal(0, 20)
                current_uacr = max(5.0, current_uacr + uacr_change)
                
                # Blood Pressure Dynamics (Multi-resolution fast fluctuation)
                current_sbp = max(95, min(190, current_sbp + np.random.normal(0, 6.0)))
                current_dbp = max(55, min(115, current_dbp + np.random.normal(0, 4.0)))

            # Check if patient reached ESRD (eGFR < 15)
            reached_esrd = 1 if current_egfr < 15.0 else 0

            records.append({
                "patient_id": patient_id,
                "visit_index": visit_num,
                "elapsed_months": round(current_months, 2),
                "delta_t_months": round(delta_t_months, 2),
                # Targets x(t)
                "egfr": round(current_egfr, 2),
                "serum_creatinine": round(current_creatinine, 2),
                "uacr": round(current_uacr, 2),
                # Vitals & Labs
                "sbp": round(current_sbp, 1),
                "dbp": round(current_dbp, 1),
                "hba1c": round(base_hba1c + np.random.normal(0, 0.2), 2),
                "hemoglobin": round(max(7.5, base_hemoglobin - (visit_num * 0.1) + np.random.normal(0, 0.2)), 2),
                "potassium": round(max(3.2, min(6.0, base_potassium + np.random.normal(0, 0.15))), 2),
                "bmi": round(base_bmi + np.random.normal(0, 0.3), 1),
                # Interventions / Controls u(t)
                "acei_arb_active": active_acei_arb,
                "sglt2i_active": active_sglt2i,
                "diuretic_active": active_diuretic,
                # Static Demographics & Context
                "age_at_visit": int(age_base + (current_months / 12.0)),
                "sex_male": sex,
                "race_black": race_black,
                "diabetes_baseline": has_diabetes,
                "cvd_baseline": has_cvd,
                # Clinical Ground Truth Labels
                "progression_phenotype": progression_phenotype,
                "reached_esrd": reached_esrd
            })

            # If patient is on terminal ESRD, stop follow-ups
            if reached_esrd == 1:
                break

    df = pd.DataFrame(records)
    df.to_csv(output_csv_path, index=False)
    print(f"Dataset successfully created!")
    print(f"Location: {output_csv_path}")
    print(f"Total Patients: {num_patients}")
    print(f"Total Longitudinal Visits: {len(df)}")
    print(f"Columns ({len(df.columns)}): {list(df.columns)}")
    print(f"File Size: {os.path.getsize(output_csv_path) / 1024:.2f} KB")

if __name__ == "__main__":
    generate_synthetic_cric_dataset(
        num_patients=1200,
        min_visits=3,
        max_visits=8,
        output_csv_path=r"c:\Users\sange\sem 5_Math\data\raw\cric_longitudinal.csv"
    )
