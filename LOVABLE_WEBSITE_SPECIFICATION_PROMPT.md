# 🩺 NephroKoopman AI — Master Website Architecture & Lovable.dev Prompt Specification

> **Copy and paste this complete specification directly into [Lovable.dev](https://lovable.dev) (or v0.dev / Cursor) to generate the complete, modern, production-grade Next.js / React + Tailwind CSS clinician web application.**

---

```markdown
# PROMPT FOR LOVABLE.DEV / V0 / FRONTEND BUILDER:

Build a world-class, ultra-modern, clinical-grade Decision Support Web Application called "NephroKoopman AI: Continuous CKD Trajectory & 'What-If' Simulation Platform".

## 1. Product Vision & Role
The application serves Nephrologists and Clinical Care Teams. It takes multi-visit longitudinal EHR lab markers and uses Deep Continuous Dynamical Systems (Deep Koopman Operator / DMDc) to:
1. Predict continuous eGFR kidney decline trajectories over 3, 6, 12, and 24 months with 95% conformal prediction uncertainty bounds.
2. Enable interactive "What-If" counterfactual simulations (e.g. testing the clinical benefit of adding SGLT2 inhibitors or lowering blood pressure).
3. Provide transparent mathematical explainability via continuous spectral eigenvalue modes and SHAP feature attributions.

---

## 2. Design System & Aesthetics
- **Theme:** Sleek, high-contrast dark medical aesthetic (Dark Slate `#0B0F19` background, `#111827` card surfaces, `#1F2937` borders).
- **Accent Colors:**
  - `Electric Cyan / Medical Blue` (`#06B6D4` / `#3B82F6`): Continuous AI Trajectories & active states.
  - `Emerald Green` (`#10B981`): Proactive Intervention, Preserved Renal Function, Stable Stage.
  - `Crimson Red / Coral` (`#EF4444`): Dialysis / ESRD threshold (eGFR < 15), Rapid Progression Warning.
  - `Amber Yellow` (`#F59E0B`): Moderate Decline / Stage 3b alerts.
- **Typography:** Inter / Plus Jakarta Sans for crisp clinical readability.
- **Components & Library:** Lucide React icons, Radix UI / Shadcn UI components, Recharts / Tremor interactive charts, Framer Motion for smooth tab and card transitions.

---

## 3. Information Architecture & Layout Flow

### 🔝 A. Header & Top Bar
- **Brand Logo:** Glowing kidney icon + "NephroKoopman AI" with "Clinical Decision Support Engine v2.4 (CRIC Validated)".
- **Hospital / Clinic Context:** Badge showing "Apollo Nephrology Network · Live Patient Cohort".
- **Global Triage Filter / Quick Switcher:** Dropdown to quickly switch between 3 patient archetypes:
  1. *Patient #1042* — Rapid Progressor (Diabetic Nephropathy, eGFR 36.5, High UACR)
  2. *Patient #2085* — Moderate Decline (Hypertensive CKD, eGFR 48.0)
  3. *Patient #3019* — Stable Trajectory (Controlled Stage 3a, eGFR 56.0)
- **Top Actions:** "Export Clinical PDF Summary", "Reset to Default", "KDIGO 2024 Guidelines Toggle".

---

### 📋 B. Left Sidebar — Patient Lab & Biometrics Control Center (Collapsible)
- **Patient Demographics:** Age (Years), Sex (Male/Female), Race (Black/Non-Black), Baseline Comorbidities (Diabetes Mellitus, CVD).
- **Current Lab Panel (Interactive Numeric Inputs & Sliders):**
  - `eGFR` (mL/min/1.73m²): Slider + Input [5 to 120]
  - `Serum Creatinine` (mg/dL): [0.5 to 10.0]
  - `UACR (Urine Albumin-to-Creatinine Ratio)` (mg/g): [5 to 3000]
  - `Systolic BP / Diastolic BP` (mmHg): [80/50 to 220/130]
  - `HbA1c` (%): [4.0 to 15.0]
  - `Hemoglobin` (g/dL): [6.0 to 18.0]
  - `Serum Potassium (K+)` (mEq/L): [2.5 to 7.0]
  - `BMI` (kg/m²): [15.0 to 50.0]
- **Current Baseline Medications (Checkboxes):**
  - [x] ACEi / ARB (Renin-Angiotensin System Inhibitor)
  - [ ] SGLT2 Inhibitor (Dapagliflozin / Empagliflozin)
  - [x] Loop / Thiazide Diuretic

---

### 📊 C. Main Dashboard Area — 3 Primary Clinical Views

#### 1. Hero Clinical Triage Banner (Top Cards):
- **Card 1: Current Renal Function:** eGFR value + KDIGO Stage (e.g. "Stage G3b: Moderate-to-Severe Decline").
- **Card 2: Albuminuria Risk:** UACR value + KDIGO Category (A1: Normal, A2: Microalbuminuria, A3: Severe Nephropathy).
- **Card 3: Progression Phenotype Badge:**
  - 🔴 `RAPID PROGRESSOR` (> 5 mL/min/year loss)
  - 🟡 `MODERATE DECLINE` (2 - 5 mL/min/year loss)
  - 🟢 `STABLE TRAJECTORY` (< 1 mL/min/year loss)
- **Card 4: Dialysis Countdown Window:** Estimated time to reaching ESRD (eGFR ≤ 15 mL/min) under standard care vs. proactive care.

---

#### 2. Interactive Tabbed Analysis Center:

##### 📑 Tab 1: 📈 Multi-Horizon Trajectory & 95% Conformal Uncertainty
- **Interactive Time-Series Chart (Recharts / Plotly):**
  - Shows continuous curve from Month 0 to Month 24.
  - **Blue Solid Curve:** Deep DMDc Predicted continuous decline trajectory.
  - **Shaded Blue Area:** 95% Conformal Confidence Envelope (statistically guaranteed safety bounds).
  - **Red Dotted Horizontal Line:** ESRD Dialysis Threshold at `eGFR = 15 mL/min`.
- **Forecast Milestone Table:**
  - Columns: Horizon (3 Mo, 6 Mo, 12 Mo, 24 Mo) | Mean eGFR | 95% Confidence Band | Est. Serum Creatinine | KDIGO Risk Stage.

---

##### 📑 Tab 2: 🧪 'What-If' Counterfactual Treatment Simulator
- **Interactive Intervention Controls:**
  - Switch: `Initiate SGLT2 Inhibitor (Dapagliflozin 10mg)`
  - Switch: `Add Max-Tolerated ACEi / ARB (Losartan/Enalapril)`
  - Slider: `Target Systolic Blood Pressure Reduction` (-10 mmHg to -30 mmHg)
  - Switch: `Add Non-Steroidal MRA (Finerenone)`
- **Dual-Trajectory Comparison Chart:**
  - **Red Dashed Line:** "Current Standard of Care" (Continuing current rapid decline).
  - **Neon Green Solid Line:** "Proactive Multi-Target Intervention" (Preserved trajectory).
- **Clinical Outcome Impact Cards:**
  - 🛡️ **eGFR Preserved:** `+4.55 mL/min/1.73m²` saved at 24 months.
  - ⏳ **Dialysis Postponement:** Postpones end-stage renal failure by an estimated **3.2 years**.
  - 📉 **MAKE Risk Reduction:** 38% reduction in Major Adverse Kidney Events.

---

##### 📑 Tab 3: 🧬 Koopman Spectral Diagnostics & Mathematical Explainability
- **Eigenvalue Spectrum in the Complex Plane:**
  - Plot showing the 32 continuous eigenvalues $\lambda_i = \sigma_i + i\omega_i$.
  - Stability threshold vertical line at $\text{Re}(\lambda) = 0$.
  - Categorization of dominant dynamic modes (Strongly Damped, Accelerated Decline, Oscillatory).
- **Feature Attribution Ranking (SHAP Waterfalls):**
  - Bar chart of relative risk drivers: UACR Proteinuria (34%), Systolic BP (28%), HbA1c Glycemia (19%), Age (11%), Hemoglobin (8%).
- **AI Architecture Benchmark Table:**
  - Compares **Deep Continuous DMDc (Ours)** vs **Temporal Transformer** vs **Sequential LSTM** across 12-Mo MAE, RMSE, $R^2$, and Rapid Progressor AUC-ROC.
  - Explanatory callout: Why Koopman dynamical systems provide continuous-time interpolation and treatment control matrices that black-box transformers cannot provide.

---

##### 📑 Tab 4: 📄 Clinical Summary & One-Click PDF Report
- Formatted medical discharge / consult note with:
  - Patient ID & Visit Date.
  - Current Staging & Trajectory Projection.
  - Recommended Pharmacotherapy Changes (ACEi + SGLT2i initiation).
  - Physician Sign-off Box & Print/Export PDF Button.

---

## 4. Mock Data & State Management Structure (React / TypeScript)

```typescript
export interface PatientProfile {
  id: string;
  name: string;
  age: number;
  sex: 'Male' | 'Female';
  race: 'Black' | 'Non-Black';
  egfr: number;
  creatinine: number;
  uacr: number;
  sbp: number;
  dbp: number;
  hba1c: number;
  hemoglobin: number;
  potassium: number;
  bmi: number;
  hasDiabetes: boolean;
  hasCVD: boolean;
  medications: {
    acei_arb: boolean;
    sglt2i: boolean;
    diuretic: boolean;
  };
}

export interface TrajectoryPoint {
  month: number;
  standardEgfr: number;
  intervenedEgfr: number;
  lowerBound95: number;
  upperBound95: number;
}
```

Implement dynamic calculations so adjusting sliders or toggling medications immediately updates the trajectory charts, metric cards, and clinical benefit text.
```
