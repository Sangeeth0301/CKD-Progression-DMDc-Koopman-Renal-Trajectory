import os
import sys
import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# Add src/ to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from explainability import ClinicalExplainabilityEngine
from pdf_generator import generate_clinical_consult_pdf

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="NephroKoopman AI | Continuous CKD Progression & What-If Platform",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded" if st.session_state.get("authenticated", False) else "collapsed"
)

# -----------------------------------------------------------------------------
# EXACT 1-TO-1 LOVABLE DARK MEDICAL DESIGN SYSTEM (CSS)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Global Base */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
    
    .stApp {
        background-color: #06090E !important;
        color: #F8FAFC !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        -webkit-font-smoothing: antialiased;
    }
    
    /* Hide Default Header / Footer */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }
    footer {
        visibility: hidden !important;
    }
    
    /* Top Bar */
    .top-bar-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 14px 0px 22px 0px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
        margin-bottom: 30px;
    }
    .brand-logo-box {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .brand-icon {
        width: 34px;
        height: 34px;
        background: rgba(6, 182, 212, 0.12);
        border: 1px solid rgba(6, 182, 212, 0.35);
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #06B6D4;
        font-size: 1.1rem;
        font-weight: bold;
    }
    .brand-title {
        font-size: 0.98rem;
        font-weight: 700;
        color: #FFFFFF;
        line-height: 1.2;
        letter-spacing: -0.02em;
    }
    .brand-subtitle {
        font-size: 0.76rem;
        color: #94A3B8;
    }
    .restricted-badge {
        display: flex;
        align-items: center;
        gap: 6px;
        font-size: 0.76rem;
        color: #94A3B8;
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.07);
        padding: 5px 12px;
        border-radius: 20px;
    }
    
    /* Hero Pill Tag */
    .hero-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #06B6D4;
        background: rgba(6, 182, 212, 0.08);
        border: 1px solid rgba(6, 182, 212, 0.25);
        padding: 4px 12px;
        border-radius: 20px;
        margin-bottom: 20px;
    }
    
    /* Hero Headline & Subtitle */
    .hero-headline {
        font-size: 3.1rem;
        font-weight: 700;
        line-height: 1.1;
        color: #FFFFFF;
        margin-bottom: 20px;
        letter-spacing: -0.04em;
    }
    .hero-body {
        font-size: 1.02rem;
        color: #94A3B8;
        line-height: 1.65;
        margin-bottom: 35px;
        max-width: 600px;
    }
    
    /* 4 Feature Cards (Exact Grid) */
    .feature-card {
        background: #0B111A;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px 20px;
        height: 100%;
        transition: all 0.2s ease-in-out;
    }
    .feature-card:hover {
        border-color: rgba(6, 182, 212, 0.35);
    }
    .feature-icon-teal {
        color: #06B6D4;
        font-size: 1.25rem;
        margin-bottom: 8px;
    }
    .feature-title {
        font-size: 0.92rem;
        font-weight: 600;
        color: #FFFFFF;
        margin-bottom: 4px;
        letter-spacing: -0.01em;
    }
    .feature-text {
        font-size: 0.82rem;
        color: #94A3B8;
        line-height: 1.5;
    }
    
    /* Bottom Stats Row */
    .stats-container {
        display: flex;
        gap: 40px;
        margin-top: 35px;
        padding-top: 25px;
        border-top: 1px solid rgba(255, 255, 255, 0.07);
    }
    .stat-val {
        font-size: 1.9rem;
        font-weight: 800;
        color: #FFFFFF;
        line-height: 1.1;
    }
    .stat-lbl {
        font-size: 0.82rem;
        color: #64748B;
        margin-top: 4px;
    }
    
    /* Clinician Sign-In Panel */
    .signin-panel {
        background: #0B111A;
        border: 1px solid rgba(255, 255, 255, 0.09);
        border-radius: 16px;
        padding: 32px 28px;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
    }
    .signin-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 6px;
    }
    .signin-sub {
        font-size: 0.85rem;
        color: #94A3B8;
        line-height: 1.5;
        margin-bottom: 25px;
    }
    .disclaimer-box {
        font-size: 0.78rem;
        color: #64748B;
        line-height: 1.45;
        margin-top: 20px;
        border-top: 1px solid rgba(255, 255, 255, 0.06);
        padding-top: 15px;
    }
    
    /* Primary Action Button */
    div[data-testid="stButton"] > button:first-child {
        background: #06B6D4 !important;
        color: #06090E !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        padding: 12px 20px !important;
        width: 100% !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stButton"] > button:first-child:hover {
        background: #22D3EE !important;
        box-shadow: 0 0 20px rgba(6, 182, 212, 0.45) !important;
        color: #000000 !important;
    }
    
    /* Text Inputs in Dark Theme */
    div[data-baseweb="input"] {
        background-color: #06090E !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 8px !important;
    }
    div[data-baseweb="input"] input {
        color: #FFFFFF !important;
    }
    
    /* Dashboard Glass Cards */
    .metric-card-kdigo {
        background: #0B111A;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 18px;
    }
    
    /* Hospital Consult PDF Export Pill Button */
    div[data-testid="stDownloadButton"] > button {
        background-color: #06B6D4 !important;
        color: #06090E !important;
        border: none !important;
        border-radius: 24px !important;
        font-weight: 700 !important;
        font-size: 0.92rem !important;
        padding: 10px 22px !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 8px !important;
        box-shadow: 0 4px 14px rgba(6, 182, 212, 0.3) !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stDownloadButton"] > button:hover {
        background-color: #22D3EE !important;
        box-shadow: 0 6px 20px rgba(6, 182, 212, 0.5) !important;
        transform: translateY(-1px);
        color: #000000 !important;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# ENGINE INITIALIZATION
# -----------------------------------------------------------------------------
@st.cache_resource
def get_engine():
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "deep_dmdc_best.pth"))
    scaler_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "state_scaler.pkl"))
    return ClinicalExplainabilityEngine(model_path=model_path, scaler_path=scaler_path)

try:
    engine = get_engine()
except Exception as e:
    st.error(f"Error loading model weights: {e}")
    st.stop()

# Authentication State
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "clinician_name" not in st.session_state:
    st.session_state.clinician_name = "Dr. A. Koopman"

# =============================================================================
# VIEW 1: EXACT LOVABLE INFO LANDING PAGE & CLINICIAN SIGN-IN
# =============================================================================
if not st.session_state.authenticated:
    # 1. Top Bar
    st.markdown("""
    <div class='top-bar-container'>
        <div class='brand-logo-box'>
            <div class='brand-icon'>〽</div>
            <div>
                <div class='brand-title'>NephroKoopman AI</div>
                <div class='brand-subtitle'>Continuous CKD progression & what-if decision platform</div>
            </div>
        </div>
        <div class='restricted-badge'>
            🔒 Restricted clinical preview
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 2. Two-Column Layout (Left: Hero & Features | Right: Sign-in Box)
    col_left, col_space, col_right = st.columns([1.55, 0.08, 1.0])
    
    with col_left:
        st.markdown("""
        <div class='hero-pill'>
            🩺 NEPHROLOGY DECISION SUPPORT
        </div>
        <div class='hero-headline'>
            See kidney decline before it happens — and test the therapy that changes it.
        </div>
        <div class='hero-body'>
            NephroKoopman AI fuses Koopman/DMDc dynamical modelling with conformal prediction 
            to turn routine labs into a continuous, uncertainty-aware CKD trajectory — plus a 
            counterfactual simulator for every major renoprotective lever.
        </div>
        """, unsafe_allow_html=True)
        
        # 2x2 Feature Grid
        f_row1_c1, f_row1_c2 = st.columns(2)
        with f_row1_c1:
            st.markdown("""
            <div class='feature-card'>
                <div class='feature-icon-teal'>📈</div>
                <div class='feature-title'>Continuous trajectory</div>
                <div class='feature-text'>24-month eGFR forecasts with 95% conformal uncertainty envelopes and dialysis countdown.</div>
            </div>
            """, unsafe_allow_html=True)
        with f_row1_c2:
            st.markdown("""
            <div class='feature-card'>
                <div class='feature-icon-teal'>🧪</div>
                <div class='feature-title'>Counterfactual therapy</div>
                <div class='feature-text'>Simulate SGLT2i, ACEi maximisation, finerenone and BP targets — see nephron-years preserved.</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        
        f_row2_c1, f_row2_c2 = st.columns(2)
        with f_row2_c1:
            st.markdown("""
            <div class='feature-card'>
                <div class='feature-icon-teal'>🧬</div>
                <div class='feature-title'>Koopman explainability</div>
                <div class='feature-text'>Spectral modes and SHAP attributions expose exactly why the model expects decline.</div>
            </div>
            """, unsafe_allow_html=True)
        with f_row2_c2:
            st.markdown("""
            <div class='feature-card'>
                <div class='feature-icon-teal'>🛡️</div>
                <div class='feature-title'>Audit-ready notes</div>
                <div class='feature-text'>One-click consult summaries formatted for the chart, with model version and assumptions.</div>
            </div>
            """, unsafe_allow_html=True)
            
        # Stats Row
        st.markdown("""
        <div class='stats-container'>
            <div>
                <div class='stat-val'>95%</div>
                <div class='stat-lbl'>Conformal coverage target</div>
            </div>
            <div>
                <div class='stat-val'>24 mo</div>
                <div class='stat-lbl'>Forecast horizon</div>
            </div>
            <div>
                <div class='stat-val'>6</div>
                <div class='stat-lbl'>Modifiable therapy levers</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_right:
        st.markdown("""
        <div class='signin-panel'>
            <div class='signin-title'>Clinician sign-in</div>
            <div class='signin-sub'>Demonstration workspace with synthetic patients — no protected health information is stored.</div>
        """, unsafe_allow_html=True)
        
        c_name = st.text_input("Clinician name or NPI", value="Dr. A. Koopman")
        c_code = st.text_input("Access code", value="••••••", type="password")
        
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        
        if st.button("Enter platform", key="enter_btn"):
            st.session_state.authenticated = True
            st.session_state.clinician_name = c_name
            st.rerun()
            
        st.markdown("""
            <div class='disclaimer-box'>
                Research prototype. Outputs are not regulatory-approved diagnostics and must not replace clinical judgement.
            </div>
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# VIEW 2: FULL DECISION SUPPORT DASHBOARD (GATED AT /dashboard)
# =============================================================================
else:
    # -------------------------------------------------------------------------
    # SIDEBAR CONTROLS
    # -------------------------------------------------------------------------
    col_sb1, col_sb2 = st.sidebar.columns([2, 1])
    col_sb1.markdown("### 📋 Patient EHR")
    if col_sb2.button("Sign Out", key="signout_btn"):
        st.session_state.authenticated = False
        st.rerun()
    
    preset = st.sidebar.selectbox(
        "Select Clinical Case:",
        ["Patient #1042 (Rapid Progressor - Diabetic Nephropathy)",
         "Patient #2085 (Moderate Decline - Hypertensive CKD)",
         "Patient #3019 (Stable CKD Stage 3a - Controlled)"]
    )
    
    if "1042" in preset:
        default_egfr, default_creat, default_uacr = 36.5, 2.3, 580.0
        default_sbp, default_dbp = 152.0, 94.0
        default_hba1c, default_hgb, default_k, default_bmi, default_age = 8.6, 10.8, 4.9, 31.2, 63
        default_dm, default_cvd = 1, 1
        default_acei, default_sglt2i, default_diur = 0, 0, 1
        badge_bg = "rgba(239, 68, 68, 0.1)"
        badge_border = "rgba(239, 68, 68, 0.35)"
        badge_dot = "#EF4444"
        badge_title = "Rapid Progressor"
        badge_color = "#FCA5A5"
        badge_sub = "Diabetic Nephropathy · Rapid Decline (-5.8 mL/min/yr)"
    elif "2085" in preset:
        default_egfr, default_creat, default_uacr = 48.0, 1.6, 220.0
        default_sbp, default_dbp = 142.0, 88.0
        default_hba1c, default_hgb, default_k, default_bmi, default_age = 7.1, 12.1, 4.4, 27.5, 59
        default_dm, default_cvd = 0, 1
        default_acei, default_sglt2i, default_diur = 1, 0, 0
        badge_bg = "rgba(245, 158, 11, 0.1)"
        badge_border = "rgba(245, 158, 11, 0.35)"
        badge_dot = "#F59E0B"
        badge_title = "Moderate Decline"
        badge_color = "#FDE68A"
        badge_sub = "Hypertensive CKD · Expected Slope (-3.2 mL/min/yr)"
    else:
        default_egfr, default_creat, default_uacr = 56.0, 1.2, 85.0
        default_sbp, default_dbp = 126.0, 78.0
        default_hba1c, default_hgb, default_k, default_bmi, default_age = 6.4, 13.5, 4.2, 24.8, 52
        default_dm, default_cvd = 0, 0
        default_acei, default_sglt2i, default_diur = 1, 1, 0
        badge_bg = "rgba(16, 185, 129, 0.1)"
        badge_border = "rgba(16, 185, 129, 0.35)"
        badge_dot = "#10B981"
        badge_title = "Stable Trajectory"
        badge_color = "#A7F3D0"
        badge_sub = "Stage 3a Controlled · Slow Slope (-0.8 mL/min/yr)"
    
    st.sidebar.markdown(f"""
    <div style='background: {badge_bg}; border: 1px solid {badge_border}; border-radius: 8px; padding: 9px 12px; margin: 10px 0 16px 0;'>
        <div style='font-size: 0.68rem; color: #94A3B8; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em;'>Clinical Phenotype</div>
        <div style='display: flex; align-items: center; gap: 7px; margin-top: 3px;'>
            <span style='display: inline-block; width: 7px; height: 7px; border-radius: 50%; background-color: {badge_dot}; box-shadow: 0 0 8px {badge_dot};'></span>
            <span style='font-size: 0.85rem; font-weight: 700; color: {badge_color};'>{badge_title}</span>
        </div>
        <div style='font-size: 0.72rem; color: #94A3B8; margin-top: 2px; line-height: 1.3;'>{badge_sub}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("#### 🧪 Laboratory Measurements")
    col_s1, col_s2 = st.sidebar.columns(2)
    egfr_in = col_s1.number_input("eGFR (mL/min)", 5.0, 120.0, float(default_egfr), 0.5)
    creat_in = col_s2.number_input("Creatinine (mg/dL)", 0.5, 10.0, float(default_creat), 0.1)
    uacr_in = col_s1.number_input("UACR (mg/g)", 5.0, 3000.0, float(default_uacr), 10.0)
    hba1c_in = col_s2.number_input("HbA1c (%)", 4.0, 15.0, float(default_hba1c), 0.1)
    sbp_in = col_s1.number_input("Systolic BP (mmHg)", 80.0, 220.0, float(default_sbp), 1.0)
    dbp_in = col_s2.number_input("Diastolic BP (mmHg)", 50.0, 130.0, float(default_dbp), 1.0)
    hgb_in = col_s1.number_input("Hemoglobin (g/dL)", 6.0, 18.0, float(default_hgb), 0.1)
    k_in = col_s2.number_input("Potassium (mEq/L)", 2.5, 7.0, float(default_k), 0.1)
    bmi_in = col_s1.number_input("BMI (kg/m²)", 15.0, 50.0, float(default_bmi), 0.1)
    age_in = col_s2.number_input("Age (Years)", 18, 95, int(default_age), 1)
    
    st.sidebar.markdown("#### 💊 Baseline Pharmacotherapy")
    b_acei = st.sidebar.checkbox("Current ACEi / ARB", value=bool(default_acei))
    b_sglt2i = st.sidebar.checkbox("Current SGLT2 Inhibitor", value=bool(default_sglt2i))
    b_diur = st.sidebar.checkbox("Current Diuretic", value=bool(default_diur))
    
    st.sidebar.markdown("---")
    if st.sidebar.button("🚪 Sign Out to Landing Page"):
        st.session_state.authenticated = False
        st.rerun()

    current_state = [
        egfr_in, creat_in, uacr_in, sbp_in, dbp_in, hba1c_in, 
        hgb_in, k_in, bmi_in, float(age_in),
        1, 0, default_dm, default_cvd
    ]
    baseline_meds = [float(b_acei), float(b_sglt2i), float(b_diur)]

    # Precompute Trajectory & Uncertainty
    unc_df = engine.predict_with_conformal_uncertainty(current_state, baseline_meds, time_horizons_months=[3, 6, 12, 24])
    m0_row = pd.DataFrame([{
        "month": 0,
        "mean_egfr": egfr_in,
        "lower_bound_95": egfr_in,
        "upper_bound_95": egfr_in,
        "confidence_margin": 0.0
    }])
    plot_df = pd.concat([m0_row, unc_df], ignore_index=True)

    if egfr_in >= 90: kdigo_stage, kdigo_sub = "Stage G1", "Normal / High"
    elif egfr_in >= 60: kdigo_stage, kdigo_sub = "Stage G2", "Mild Decline"
    elif egfr_in >= 45: kdigo_stage, kdigo_sub = "Stage G3a", "Mild-to-Moderate"
    elif egfr_in >= 30: kdigo_stage, kdigo_sub = "Stage G3b", "Moderate-to-Severe"
    elif egfr_in >= 15: kdigo_stage, kdigo_sub = "Stage G4", "Severe Reduction"
    else: kdigo_stage, kdigo_sub = "Stage G5", "Kidney Failure (ESRD)"
    
    if uacr_in < 30: alb_cat = "A1 (Normal)"
    elif uacr_in <= 300: alb_cat = "A2 (Microalbuminuria)"
    else: alb_cat = "A3 (Severely Increased)"
    
    annual_loss = 5.8 if ("1042" in preset or egfr_in < 40) else (3.2 if egfr_in < 55 else 0.8)
    months_to_dialysis = max(0, int(((egfr_in - 15.0) / max(0.1, annual_loss)) * 12)) if egfr_in > 15 else 0

    # Assemble Patient Details for Hospital PDF Export
    patient_pdf_data = {
        'patient_id': preset.split(' (')[0].replace('Patient ', ''),
        'age': age_in,
        'sbp': int(sbp_in),
        'dbp': int(dbp_in),
        'egfr': egfr_in,
        'creatinine': creat_in,
        'uacr': int(uacr_in),
        'hba1c': hba1c_in,
        'bmi': bmi_in,
        'kdigo_stage': f"{kdigo_stage} ({kdigo_sub})",
        'alb_cat': alb_cat,
        'annual_decline': annual_loss,
        'months_to_dialysis': months_to_dialysis,
        'saved_egfr': 4.55,
        'meds_list': [
            ("ACEi / ARB RAS Blocker", "Ramipril 10mg Daily" if b_acei else "Losartan 50mg (Targeted)", "Reduces intraglomerular pressure & lowers urine protein leak", "Active" if b_acei else "Recommended"),
            ("SGLT2 Inhibitor", "Dapagliflozin 10mg Once Daily" if b_sglt2i else "Dapagliflozin 10mg (Renoprotection)", "Slows kidney functional decline & provides cardiac protection", "Active" if b_sglt2i else "Recommended"),
            ("Diuretic", "Furosemide 20mg Morning" if b_diur else "Not Prescribed", "Manages fluid overload and maintains blood pressure control", "Active" if b_diur else "Inactive")
        ]
    }
    
    try:
        consult_pdf_bytes = generate_clinical_consult_pdf(patient_pdf_data, plot_df, st.session_state.clinician_name)
    except Exception as e:
        consult_pdf_bytes = b""

    # Authenticated Top Bar with Cyan Pill PDF Export Button
    col_th1, col_th2 = st.columns([3.2, 1.3])
    with col_th1:
        st.markdown(f"""
        <div class='brand-logo-box' style='padding: 6px 0;'>
            <div class='brand-icon'>〽</div>
            <div>
                <div class='brand-title'>NephroKoopman AI <span style='font-size:0.75rem; font-weight:bold; color:#06B6D4; background:rgba(6,182,212,0.1); padding:2px 8px; border-radius:12px; margin-left:8px;'>WORKSPACE ACTIVE</span></div>
                <div class='brand-subtitle'>Session: <b>{st.session_state.clinician_name}</b> · Patient: <b>{preset.split(' (')[0]}</b></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_th2:
        st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)
        if consult_pdf_bytes:
            st.download_button(
                label="📄 Export Consult PDF",
                data=consult_pdf_bytes,
                file_name=f"Hospital_Nephrology_Consult_{preset.split(' (')[0].replace('#','').replace(' ','_')}.pdf",
                mime="application/pdf",
                key="top_bar_pdf_btn"
            )

    st.markdown("<div style='border-bottom: 1px solid rgba(255, 255, 255, 0.06); margin-bottom: 25px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # MAIN STAGE: KDIGO HERO TRIAGE CARDS
    # -------------------------------------------------------------------------
    if egfr_in >= 90: kdigo_stage, kdigo_sub = "Stage G1", "Normal / High"
    elif egfr_in >= 60: kdigo_stage, kdigo_sub = "Stage G2", "Mild Decline"
    elif egfr_in >= 45: kdigo_stage, kdigo_sub = "Stage G3a", "Mild-to-Moderate"
    elif egfr_in >= 30: kdigo_stage, kdigo_sub = "Stage G3b", "Moderate-to-Severe"
    elif egfr_in >= 15: kdigo_stage, kdigo_sub = "Stage G4", "Severe Reduction"
    else: kdigo_stage, kdigo_sub = "Stage G5", "Kidney Failure (ESRD)"
    
    if uacr_in < 30: alb_cat = "A1 (Normal)"
    elif uacr_in <= 300: alb_cat = "A2 (Microalbuminuria)"
    else: alb_cat = "A3 (Severely Increased)"
    
    annual_loss = 5.8 if ("1042" in preset or egfr_in < 40) else (3.2 if egfr_in < 55 else 0.8)
    months_to_dialysis = max(0, int(((egfr_in - 15.0) / max(0.1, annual_loss)) * 12)) if egfr_in > 15 else 0

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class='metric-card-kdigo'>
            <div style='color: #94A3B8; font-size: 0.8rem; font-weight: 700; text-transform: uppercase;'>Renal Filtration</div>
            <div style='font-size: 1.85rem; font-weight: 800; color: #06B6D4; margin: 4px 0;'>{egfr_in:.1f} <span style='font-size: 0.85rem; font-weight: 400; color: #94A3B8;'>mL/min/1.73m²</span></div>
            <div style='color: #38BDF8; font-weight: 600; font-size: 0.85rem;'>{kdigo_stage} ({kdigo_sub})</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class='metric-card-kdigo'>
            <div style='color: #94A3B8; font-size: 0.8rem; font-weight: 700; text-transform: uppercase;'>Albuminuria Matrix</div>
            <div style='font-size: 1.85rem; font-weight: 800; color: #F8FAFC; margin: 4px 0;'>{uacr_in:.0f} <span style='font-size: 0.85rem; font-weight: 400; color: #94A3B8;'>mg/g</span></div>
            <div style='color: {"#EF4444" if "A3" in alb_cat else "#F59E0B"}; font-weight: 600; font-size: 0.85rem;'>{alb_cat}</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        if egfr_in < 45 or uacr_in > 300:
            badge_str = "<span style='color:#EF4444; font-weight:bold;'>🔴 RAPID PROGRESSOR</span>"
        elif egfr_in < 60:
            badge_str = "<span style='color:#F59E0B; font-weight:bold;'>🟡 MODERATE DECLINE</span>"
        else:
            badge_str = "<span style='color:#10B981; font-weight:bold;'>🟢 STABLE TRAJECTORY</span>"
        st.markdown(f"""
        <div class='metric-card-kdigo'>
            <div style='color: #94A3B8; font-size: 0.8rem; font-weight: 700; text-transform: uppercase;'>Progression Phenotype</div>
            <div style='font-size: 1.15rem; margin: 8px 0;'>{badge_str}</div>
            <div style='color: #94A3B8; font-size: 0.82rem;'>Est. Slope: <b>-{annual_loss:.1f} mL/min/yr</b></div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class='metric-card-kdigo'>
            <div style='color: #94A3B8; font-size: 0.8rem; font-weight: 700; text-transform: uppercase;'>Dialysis Countdown</div>
            <div style='font-size: 1.85rem; font-weight: 800; color: {"#EF4444" if months_to_dialysis < 24 else "#10B981"}; margin: 4px 0;'>
                {f"{months_to_dialysis} Mo" if months_to_dialysis > 0 else "Active ESRD"}
            </div>
            <div style='color: #94A3B8; font-size: 0.82rem;'>To critical eGFR ≤ 15 threshold</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 4 INTERACTIVE TABS
    # -------------------------------------------------------------------------
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Continuous Trajectory & Uncertainty", 
        "🧪 Counterfactual 'What-If' Simulator", 
        "🧬 Koopman Spectral Diagnostics",
        "📄 Clinical Consult Note & EHR Export"
    ])

    # Tab 1: Trajectory
    with tab1:
        st.markdown("#### 📈 Continuous Multi-Horizon Trajectory (3, 6, 12, 24 Months)")
        unc_df = engine.predict_with_conformal_uncertainty(current_state, baseline_meds, time_horizons_months=[3, 6, 12, 24])
        
        m0_row = pd.DataFrame([{
            "month": 0,
            "mean_egfr": egfr_in,
            "lower_bound_95": egfr_in,
            "upper_bound_95": egfr_in,
            "confidence_margin": 0.0
        }])
        plot_df = pd.concat([m0_row, unc_df], ignore_index=True)
        
        fig1 = go.Figure()
        
        # 95% Confidence Band
        fig1.add_trace(go.Scatter(
            x=list(plot_df["month"]) + list(plot_df["month"])[::-1],
            y=list(plot_df["upper_bound_95"]) + list(plot_df["lower_bound_95"])[::-1],
            fill='toself',
            fillcolor='rgba(6, 182, 212, 0.15)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            name='95% Conformal Confidence Band'
        ))
        
        # Predicted Trajectory
        fig1.add_trace(go.Scatter(
            x=plot_df["month"],
            y=plot_df["mean_egfr"],
            mode='lines+markers',
            name='Deep Continuous DMDc Trajectory',
            line=dict(color='#06B6D4', width=3.5),
            marker=dict(size=9, color='#38BDF8', line=dict(color='#06090E', width=2))
        ))
        
        fig1.add_hline(y=15, line_dash="dot", line_color="#EF4444", annotation_text="ESRD Dialysis Line (15 mL/min)", annotation_position="bottom right")
        
        fig1.update_layout(
            xaxis_title="Time Horizon (Months)",
            yaxis_title="Estimated GFR (mL/min/1.73m²)",
            template="plotly_dark",
            paper_bgcolor="#0B111A",
            plot_bgcolor="#0B111A",
            height=440,
            hovermode="x unified",
            margin=dict(l=40, r=40, t=30, b=40)
        )
        st.plotly_chart(fig1, width='stretch')
        
        st.dataframe(unc_df.style.format({
            "mean_egfr": "{:.1f} mL/min",
            "lower_bound_95": "{:.1f} mL/min",
            "upper_bound_95": "{:.1f} mL/min",
            "confidence_margin": "±{:.1f} mL/min"
        }), width='stretch')

    # Tab 2: What-If
    with tab2:
        st.markdown("#### 🧪 Pharmacological 'What-If' Counterfactual Simulator")
        w_c1, w_c2, w_c3 = st.columns(3)
        opt_acei = w_c1.checkbox("Maximize ACEi / ARB Titration", value=True)
        opt_sglt2i = w_c2.checkbox("Initiate SGLT2 Inhibitor (Dapagliflozin)", value=True)
        opt_diur = w_c3.checkbox("Optimize Diuretic Control", value=bool(default_diur))
        
        intervened_meds = [float(opt_acei), float(opt_sglt2i), float(opt_diur)]
        sim_df = engine.simulate_what_if_intervention(current_state, baseline_meds, intervened_meds, [3, 6, 12, 24])
        
        sim_m0 = pd.DataFrame([{"month": 0, "pred_egfr": egfr_in, "intervened_egfr": egfr_in, "egfr_saved": 0.0}])
        sim_plot = pd.concat([sim_m0, sim_df], ignore_index=True)
        
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=sim_plot["month"], y=sim_plot["pred_egfr"],
            mode='lines+markers', name='Standard Care (Current Decline)',
            line=dict(color='#EF4444', width=3, dash='dash'), marker=dict(size=7)
        ))
        fig2.add_trace(go.Scatter(
            x=sim_plot["month"], y=sim_plot["intervened_egfr"],
            mode='lines+markers', name='Proactive Multi-Target Regimen (ACEi + SGLT2i)',
            line=dict(color='#10B981', width=3.5), marker=dict(size=9, color='#34D399')
        ))
        fig2.add_hline(y=15, line_dash="dot", line_color="#94A3B8", annotation_text="ESRD Dialysis Line (15 mL/min)")
        
        fig2.update_layout(
            xaxis_title="Elapsed Months", yaxis_title="eGFR (mL/min/1.73m²)",
            template="plotly_dark", paper_bgcolor="#0B111A", plot_bgcolor="#0B111A",
            height=440, hovermode="x unified", margin=dict(l=40, r=40, t=30, b=40)
        )
        st.plotly_chart(fig2, width='stretch')
        
        saved_24m = sim_df[sim_df["month"] == 24]["egfr_saved"].values[0]
        postponed_years = max(0.5, saved_24m / max(0.5, annual_loss))
        
        st.markdown(f"""
        <div class='metric-card-kdigo' style='border-left: 4px solid #10B981;'>
            <div style='color: #10B981; font-weight: 700; font-size: 1.05rem;'>🛡️ Projected Clinical Gain</div>
            <div style='color: #F8FAFC; font-size: 0.92rem; margin-top: 5px; line-height: 1.5;'>
                Initiating combination SGLT2i + ACEi/ARB therapy is projected to preserve 
                <b style='color: #10B981;'>+{saved_24m:.2f} mL/min/1.73m²</b> of renal filtration over 24 months, 
                postponing dialysis necessity by <b style='color: #10B981;'>~{postponed_years:.1f} years</b>.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Tab 3: Spectral
    with tab3:
        st.markdown("#### 🧬 Koopman Continuous Spectral Modes & Benchmark")
        modes_df = engine.analyze_spectral_modes()
        
        col_e1, col_e2 = st.columns([1.2, 1])
        with col_e1:
            fig_spec = px.scatter(
                modes_df, x="real_lambda", y="imag_lambda",
                color="clinical_interpretation", hover_data=["mode_id", "half_life_months"],
                title="Continuous Koopman Eigenvalue Spectrum (Complex Plane)",
                labels={"real_lambda": "Real Part Re(λ) [Growth / Damping]", "imag_lambda": "Imaginary Part Im(λ)"}
            )
            fig_spec.add_vline(x=0, line_dash="dash", line_color="#EF4444", annotation_text="Stability Line")
            fig_spec.update_layout(template="plotly_dark", height=400, paper_bgcolor="#0B111A", plot_bgcolor="#0B111A")
            st.plotly_chart(fig_spec, width='stretch')
        with col_e2:
            st.markdown("##### 🏆 Comparative Benchmark Results")
            bench_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "benchmark_results.csv"))
            if os.path.exists(bench_path):
                st.dataframe(pd.read_csv(bench_path, index_col=0), width='stretch')

    # Tab 4: Consult Note & PDF Export
    with tab4:
        st.markdown("#### 📄 Hospital Clinical Consultation & Patient Trajectory Report")
        
        col_p1, col_p2 = st.columns([1.5, 1])
        with col_p1:
            st.markdown(f"""
            <div class='metric-card-kdigo' style='border-left: 4px solid #06B6D4;'>
                <div style='color: #06B6D4; font-weight: 700; font-size: 1.05rem;'>🏥 Official Hospital Consult PDF Report</div>
                <div style='color: #94A3B8; font-size: 0.85rem; margin-top: 6px; line-height: 1.55;'>
                    A print-ready, professional white-background consultation document formatted for hospital records and patient discharge. 
                    Includes:
                    <ul style='margin-top: 6px; color: #E2E8F0; padding-left: 20px;'>
                        <li><b>Patient Details:</b> Age, Blood Pressure, BMI, eGFR, KDIGO Stage, UACR.</li>
                        <li><b>Prescribed Tablets:</b> Active medication regimen (ACEi/ARB, SGLT2i, Diuretics) with clinical purposes.</li>
                        <li><b>High-Res Graph:</b> 24-Month continuous eGFR trajectory with 95% conformal safety margins.</li>
                        <li><b>Simple English Summary:</b> Plain-language explanation of kidney health, dialysis countdown, and therapeutic gains.</li>
                        <li><b>Attending Doctor Verification:</b> Certified by {st.session_state.clinician_name}.</li>
                        <li><b>Personal Note:</b> Thoughtful patient care thank-you message at the bottom.</li>
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            if consult_pdf_bytes:
                st.download_button(
                    label="📄 Export Consult PDF (Hospital Print-Ready)",
                    data=consult_pdf_bytes,
                    file_name=f"Hospital_Nephrology_Consult_{preset.split(' (')[0].replace('#','').replace(' ','_')}.pdf",
                    mime="application/pdf",
                    key="tab4_pdf_download_btn"
                )
        
        with col_p2:
            st.markdown("##### 📝 EHR Raw Consult Text")
            note_text = f"""========================================================================================
NEPHROLOGY CLINICAL CONSULTATION NOTE
Evaluator: {st.session_state.clinician_name} | Protocol: Deep Continuous DMDc v2.4
========================================================================================
PATIENT SUMMARY:
- Case Profile: {preset}
- Age: {age_in} Years | Baseline eGFR: {egfr_in:.1f} mL/min ({kdigo_stage}, {kdigo_sub})
- Proteinuria: UACR {uacr_in:.0f} mg/g ({alb_cat}) | BP: {int(sbp_in)}/{int(dbp_in)} mmHg | HbA1c: {hba1c_in:.1f}%

AI TRAJECTORY & PROJECTION:
- Natural Trajectory (24 Mo): {unc_df[unc_df['month']==24]['mean_egfr'].values[0]:.1f} mL/min (95% CI: [{unc_df[unc_df['month']==24]['lower_bound_95'].values[0]:.1f}, {unc_df[unc_df['month']==24]['upper_bound_95'].values[0]:.1f}])
- Estimated Time to Dialysis (Standard Care): {months_to_dialysis} Months

RECOMMENDED INTERVENTIONS:
1. Initiate SGLT2 Inhibitor (Dapagliflozin 10mg daily) + Maximize RAS Blockade.
2. Target Blood Pressure < 120-130 mmHg per KDIGO 2024 guidelines.
3. Projected Gain: +{saved_24m:.2f} mL/min filtration saved | Dialysis delayed by ~{postponed_years:.1f} years.

Physician Signature: {st.session_state.clinician_name}  Date: {datetime.date.today()}
========================================================================================"""
            st.text_area("EHR Clipboard Preview:", value=note_text.strip(), height=200)
            st.download_button("📥 Export EHR Text (.txt)", data=note_text.strip(), file_name=f"CKD_Consult_{preset.split(' ')[1]}.txt", key="tab4_txt_download_btn")

# -----------------------------------------------------------------------------
# GLOBAL FOOTER
# -----------------------------------------------------------------------------
st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)
st.markdown("<center style='color:#64748B; font-size:0.8rem;'>NephroKoopman AI · Continuous Koopman DMDc Clinical Intelligence · Research Prototype</center>", unsafe_allow_html=True)
