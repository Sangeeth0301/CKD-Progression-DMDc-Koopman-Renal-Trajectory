# Explainable AI-Based Chronic Kidney Disease (CKD) Progression Forecasting via Multi-Resolution Deep Continuous Koopman Operator Theory (mr-DeepDMDc)

---

## 1. Clinical Problem Statement & Global Healthcare Burden

### 1.1 The Real-World Crisis
Chronic Kidney Disease (CKD) affects over **850 million people globally**, including more than **100 million patients in India alone**. 
* **The Asymptomatic Silent Phase:** In early and intermediate stages (Stages 1 through 3b), patients experience zero symptoms while nephrons undergo progressive, irreversible sclerotic destruction.
* **The Tipping Point to Dialysis:** By the time overt symptoms appear, kidney function has often dropped below 15 mL/min/1.73m² (Stage 5 / End-Stage Renal Disease - ESRD). At this point, conservative medical management is exhausted, leaving patients with only two grueling, expensive, and life-limiting options: **lifelong dialysis** or **kidney transplantation**.
* **Socioeconomic & Mortality Burden:** In developing nations like India, fewer than 10–15% of ESRD patients can afford or access sustained maintenance dialysis, leading to catastrophic out-of-pocket healthcare expenses and high avoidable mortality.

### 1.2 The Doctor's Real Clinical Dilemma: "When Will It Happen?"
In routine clinical practice, nephrologists already know *if* a patient has CKD (diagnosis is simple via blood creatinine). What doctors urgently need to answer are:
1. **The Timeline:** *"Will this specific patient's kidney function crash within 6 months, or remain stable for the next 5 years?"*
2. **The Window of Intervention:** *"When is the critical window before irreversible nephron failure occurs?"*
3. **The 'What-If' Dilemma:** *"If I aggressively lower this patient's Systolic BP from 150 to 125 mmHg, or start an SGLT2 inhibitor today, how many years of dialysis-free life do we save?"*

### 1.3 Why Existing AI Models Fail
* **Static Single-Snapshot Blindness:** 90% of published AI models perform basic binary classification (`CKD: Yes/No`) on static single-visit datasets. They are completely blind to historical rate of change (velocity and acceleration of decline).
* **Black-Box Opacity:** Standard Deep Neural Networks provide unexplainable predictions without mathematical guarantees or biological proof.
* **No Actionable Interventions:** Standard models lack control-theoretic mechanisms, making counterfactual "What-If" treatment simulations impossible.

---

## 2. Dynamic Mode Decomposition (DMD): Which Variant and Why?

Standard Dynamic Mode Decomposition (DMD) was originally formulated for linear fluid dynamics. Because human disease progression is non-linear, multi-scale, and actively altered by medical treatments, standard DMD is insufficient.

| DMD Variant | Mathematical Form | Why Standard Fails & Why We Need It | Role in Our Project |
| :--- | :--- | :--- | :--- |
| **Standard Exact DMD** | $x_{t+1} = A \cdot x_t$ | Assumes raw clinical data is already linear. Cannot handle complex biology or medications. | Insufficient on its own (baseline only). |
| **Extended DMD (EDMD)** | $z_{t+1} = A \cdot z_t$ ($z=\Phi(x)$) | Uses fixed polynomial dictionaries. In biology, manually guessing the right non-linear equations is impossible. | Replaced by Deep Learning (Autoencoder) to learn $\Phi(x)$ automatically. |
| **DMD with Control (DMDc)** | $\frac{dz}{dt} = A \cdot z + B \cdot u$ | Crucial because patients receive medications $u(t)$. Separates natural disease decline ($A$) from drug response ($B$). | **Core Engine for the 'What-If' Counterfactual Treatment Simulator.** |
| **Multi-Resolution DMD (mrDMD)** | $z(t) = z_{\text{slow}}(t) + z_{\text{fast}}(t)$ | Decomposes multi-scale time dynamics: separates **fast biomarker noise** (BP spikes, temporary glucose shifts) from **slow chronic decline** (years-long eGFR drop). | Decomposes temporal scales during preprocessing. |
| **Our Model: `mr-DeepDMDc`** | $\frac{dz}{dt} = A \cdot \psi_{\text{enc}}(x) + B \cdot u$ | Combines Neural Autoencoder (Deep Learning) + Multi-Scale Temporal Split (mrDMD) + Treatment Matrix (DMDc). | **THE PRIMARY FLAGSHIP NOVEL MODEL of this project.** |

---

## 3. Base Papers Benchmark

| Dimension | Base Paper Paradigm 1: Classical ML / Flat EHR | Base Paper Paradigm 2: Standard Deep EHR (LSTM / RETAIN) | **Our Proposed Architecture: Multi-Resolution Deep DMDc + XAI** |
| :--- | :--- | :--- | :--- |
| **Representative Works** | XGBoost, Random Forest, Logistic Reg. | Choi et al. (RETAIN), Li et al. (BEHRT) | **Multi-Resolution Deep Continuous DMDc** |
| **Prediction Type** | Static binary classification (`CKD: Yes/No`) | Discrete next-step event risk | **Continuous Multi-Horizon Trajectory (3, 6, 12 mo)** |
| **Irregular Visits ($\Delta t$)** | Ignored (assumes uniform intervals) | Crude binning / standard positional enc. | **Continuous Koopman Generator ($\exp(A\Delta t)$) handles arbitrary $\Delta t$** |
| **What-If Simulation** | Impossible (no causal/control mechanism) | Infeasible (requires synthetic retraining) | **Explicit Control Matrix $B \cdot u$ enables real-time counterfactuals** |
| **Multi-Scale Dynamics** | None | None | **mrDMD separates fast BP spikes from slow eGFR loss** |
| **Interpretability** | Post-hoc SHAP only | Attention weights (often noisy / non-causal) | **Dual-Layer: Spectral Eigenmodes ($A=W\Lambda W^{-1}$) + SHAP** |
| **Uncertainty & Safety** | None (blind overconfident point estimates) | Poorly calibrated softmax scores | **Bayesian Monte Carlo Dropout + 95% Conformal Safety Bands** |

---

## 4. End-to-End System Architecture

### 4.1 Text-Based Flow Architecture Diagram

```
+---------------------------------------------------------------------------------------+
| 1. CRIC DATASET INGESTION: Multi-Visit History [x_1, ..., x_t], Interventions u(t), Δt |
+-------------------------------------------+-------------------------------------------+
                                            |
                                            v
+---------------------------------------------------------------------------------------+
| 2. mrDMD MULTI-RESOLUTION PREPROCESSING: Decompose Fast (BP/Glucose) vs Slow (eGFR)   |
+-------------------------------------------+-------------------------------------------+
                                            |
                                            v
+---------------------------------------------------------------------------------------+
| 3. DEEP AUTOENCODER ENCODER: ψ_enc maps non-linear state x(t) into Latent z(t) ∈ R^d  |
+-------------------------------------------+-------------------------------------------+
                                            |
                                            v
+---------------------------------------------------------------------------------------+
| 4. CONTINUOUS DMDc STATE-SPACE OPERATOR: dz/dt = A·z(t) + B·u(t)                      |
|    Exact Matrix Exponential: z(t+Δt) = exp(A·Δt)·z(t) + A⁻¹(exp(A·Δt) - I)·B·u(t)     |
+-------------------------------------------+-------------------------------------------+
                                            |
                                            v
+---------------------------------------------------------------------------------------+
| 5. NON-LINEAR DECODER: x̂(t+Δt) = ψ_dec(z(t+Δt)) --> Multi-Horizon eGFR (3, 6, 12 mo)  |
+-------------------------------------------+-------------------------------------------+
                                            |
                                            v
+---------------------------------------------------------------------------------------+
| 6. CLINICAL AI POST-PROCESSING:                                                       |
|    • What-If Simulator: ΔTraj = B·Δu         • Spectral Modes: A = W·Λ·W⁻¹ (Crash/Safe) |
|    • 95% Conformal Prediction Bounds         • SHAP Feature Attribution Waterfalls     |
+-------------------------------------------+-------------------------------------------+
                                            |
                                            v
+---------------------------------------------------------------------------------------+
| 7. INTERACTIVE CLINICIAN DASHBOARD: Trajectory Viewer, Medication Sliders, Risk Badge |
+---------------------------------------------------------------------------------------+
```

---

## 5. Step-by-Step Methodological Approach

1. **Step 1 (Cohort Ingestion & Time-Delta Alignment):** Load CRIC longitudinal records, extracting target states $x(t)$ (eGFR, Creatinine, UACR), intervention controls $u(t)$ (BP targets, ACEi/ARB, SGLT2i), and irregular follow-up intervals $\Delta t$.
2. **Step 2 (Multi-Resolution mrDMD Time-Scale Decomposition):** Apply Multi-Resolution DMD to split acute physiological fluctuations (BP spikes, temporary glucose changes) from slow-scale chronic renal deterioration (long-term eGFR decline).
3. **Step 3 (Deep Koopman Autoencoder Construction):** Build non-linear Encoder $\psi_{\text{enc}}: \mathbb{R}^n \to \mathbb{R}^d$ and Decoder $\psi_{\text{dec}}: \mathbb{R}^d \to \mathbb{R}^n$ to project clinical measurements into a linear latent space $z(t)$.
4. **Step 4 (Continuous State-Space Operator & Training):** Solve $\frac{dz}{dt} = A \cdot z + B \cdot u$ via matrix exponential integration $\exp(A\Delta t)$ to naturally forecast across irregular visit gaps. Train end-to-end to minimize the unified 4-part loss function.
5. **Step 5 (Explainability & Safety Bounds):** Decompose Matrix $A$ into eigenvalues to classify Rapid vs. Stable progressors based on Lyapunov stability, compute 95% Conformal Prediction bands, and calculate SHAP feature attributions.
6. **Step 6 (Interactive Clinician Decision Dashboard):** Deploy a Streamlit web application with interactive "What-If" sliders ($\Delta \text{Trajectory} = B \cdot \Delta u$), live trajectory curves, and risk badges.

---

## 6. Core Mathematical Formulations (The 4 Key Equations)

### Main Equation 1: Continuous Koopman State-Space System & Exact Solution
$$\frac{dz(t)}{dt} = A \cdot z(t) + B \cdot u(t)$$
$$z(t + \Delta t) = e^{A \cdot \Delta t} \cdot z(t) + A^{-1} \cdot (e^{A \cdot \Delta t} - I) \cdot B \cdot u(t)$$
$$\hat{x}(t + \Delta t) = \psi_{\text{dec}}(z(t + \Delta t))$$

* **Matrix & Vector Dictionary:**
  * **$x(t) \in \mathbb{R}^n$:** Measurable clinical features (eGFR, Serum Creatinine, UACR, Systolic BP, Diastolic BP, Hemoglobin, Potassium).
  * **$z(t) \in \mathbb{R}^d$:** Latent biological nephron capacity, filtered of laboratory measurement error.
  * **Matrix $A \in \mathbb{R}^{d \times d}$:** Autonomous Renal Decline Operator (natural disease decline speed without intervention).
  * **Matrix $B \in \mathbb{R}^{d \times p}$:** Therapeutic Response Operator (quantifies how much medications slow down decline).
  * **Vector $u(t) \in \mathbb{R}^p$:** Clinical interventions (target SBP reduction, ACEi/ARB dosage, SGLT2 inhibitor presence).

### Main Equation 2: Spectral Eigenmode Stability (Crash vs. Stable)
$$A = W \cdot \Lambda \cdot W^{-1}, \quad \Lambda = \text{diag}(\lambda_1, \lambda_2, \dots, \lambda_d)$$
* **$\text{Re}(\lambda_j) > 0$ (Rapid Collapse Mode):** Unstable exponential kidney failure (flags high-risk rapid progressors).
* **$\text{Re}(\lambda_j) \approx 0$ (Chronic Stable Mode):** Predictable age-associated decline.
* **$\text{Re}(\lambda_j) < 0$ (Stabilizing Response Mode):** Medical therapy actively protecting the kidney.

### Main Equation 3: Unified 4-Part Multi-Objective Loss Function
$$\mathcal{L}_{\text{total}} = \lambda_1 \mathcal{L}_{\text{reconstruction}} + \lambda_2 \mathcal{L}_{\text{prediction}} + \lambda_3 \mathcal{L}_{\text{koopman\_linear}} + \lambda_4 \mathcal{L}_{\text{bio\_guardrail}}$$
* **$\mathcal{L}_{\text{reconstruction}}$:** $\frac{1}{N}\sum \|x_i(t) - \psi_{\text{dec}}(\psi_{\text{enc}}(x_i(t)))\|_2^2$
* **$\mathcal{L}_{\text{prediction}}$:** $\frac{1}{N}\sum \sum_{k \in \{3,6,12\}} \|\hat{x}_i(t+k) - x_i(t+k)\|_2^2$
* **$\mathcal{L}_{\text{koopman\_linear}}$:** $\frac{1}{N}\sum \|z_i(t+\Delta t) - [e^{A\Delta t}z_i(t) + B_{\text{eff}}u_i(t)]\|_2^2$
* **$\mathcal{L}_{\text{bio\_guardrail}}$:** $\text{ReLU}(-\hat{\text{eGFR}}) + \text{ReLU}(-\hat{\text{Creatinine}}) + \lambda_s \text{ReLU}(|\frac{d\hat{\text{eGFR}}}{dt}| - \text{MaxSlope})$

### Main Equation 4: Non-Parametric Conformal 95% Confidence Guarantee
$$P\left(y_{\text{true}} \in [\hat{x} - \hat{q}, \, \hat{x} + \hat{q}]\right) \ge 0.95$$
* Guarantees that the true future eGFR will fall inside the shaded confidence band at least 95% of the time.

---

## 7. CRIC Dataset Specifications & Schema

| Feature Category | Specific Variables | Clinical Unit / Format | Role in Model |
| :--- | :--- | :--- | :--- |
| **Target States $x(t)$** | eGFR (CKD-EPI formula)<br>Serum Creatinine<br>Urine Protein (UACR) | mL/min/1.73m²<br>mg/dL<br>mg/g | Target state vector to forecast across 3, 6, 12 months |
| **Hemodynamic Vitals** | Systolic BP (SBP)<br>Diastolic BP (DBP)<br>Body Mass Index (BMI) | mmHg<br>mmHg<br>kg/m² | Dynamic physiological observables impacting renal perfusion |
| **Metabolic & Lab Markers** | HbA1c / Blood Glucose<br>Serum Potassium & Bicarbonate<br>Serum Hemoglobin & Albumin | % / mg/dL<br>mEq/L<br>g/dL | Biomarkers tracking diabetic nephropathy, acidosis, and anemia |
| **Clinical Controls $u(t)$** | RAAS Blockade (ACEi/ARB)<br>SGLT2 Inhibitor Therapy<br>Target SBP Reduction | Binary (0/1)<br>Binary (0/1)<br>Continuous $\Delta$mmHg | Exogenous control inputs for What-If counterfactual simulator |
| **Demographics & Prior** | Age, Biological Sex, Race<br>Baseline Diabetes & CVD History | Years, Binary<br>Binary indicators | Static contextual conditioning for latent encoder |

---

## 8. Technical Stack & Clinician Dashboard

* **Backend & Math:** Python 3.10+, PyTorch / PyTorch Lightning, SciPy (`scipy.linalg.expm`), PyDMD, SHAP.
* **Baselines for Comparison:** Scikit-Learn, XGBoost, LightGBM, Temporal Transformer, LSTM.
* **Safety & Calibration:** MAPIE / Conformal Prediction.
* **Frontend Web Dashboard:** **Streamlit + Plotly** featuring:
  1. **Trajectory Viewer:** Past visits + 3/6/12-month predicted curves with shaded 95% confidence bands.
  2. **"What-If" Simulator:** Interactive sliders for Blood Pressure and Medications that alter future trajectories in real time ($\Delta \text{Trajectory} = B \cdot \Delta u$).
  3. **Explainability Panel:** SHAP waterfall plot and spectral stability classification badge.
