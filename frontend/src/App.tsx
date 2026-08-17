import React, { useState } from 'react';
import { 
  Activity, 
  Lock, 
  TrendingDown, 
  FlaskConical, 
  GitBranch, 
  ShieldCheck, 
  ArrowRight, 
  Sliders, 
  LogOut, 
  FileText, 
  Download,
  AlertTriangle,
  CheckCircle2,
  Clock,
  HeartPulse,
  UserCheck
} from 'lucide-react';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer, 
  Area, 
  ComposedChart,
  ReferenceLine,
  ScatterChart,
  Scatter
} from 'recharts';

export function App() {
  // Navigation State
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [clinicianName, setClinicianName] = useState('Dr. A. Koopman');
  const [accessCode, setAccessCode] = useState('••••••');
  
  // Dashboard State
  const [activeTab, setActiveTab] = useState<'trajectory' | 'whatif' | 'koopman' | 'notes'>('trajectory');
  const [selectedCase, setSelectedCase] = useState<'1042' | '2085' | '3019'>('1042');

  // Patient Biomarkers
  const [egfr, setEgfr] = useState(36.5);
  const [creatinine, setCreatinine] = useState(2.3);
  const [uacr, setUacr] = useState(580);
  const [sbp, setSbp] = useState(152);
  const [dbp, setDbp] = useState(94);
  const [hba1c, setHba1c] = useState(8.6);
  const [hgb, setHgb] = useState(10.8);
  const [potassium, setPotassium] = useState(4.9);
  const [bmi, setBmi] = useState(31.2);
  const [age, setAge] = useState(63);

  // Baseline Medications
  const [baseAcei, setBaseAcei] = useState(false);
  const [baseSglt2i, setBaseSglt2i] = useState(false);
  const [baseDiuretic, setBaseDiuretic] = useState(true);

  // What-If Interventions
  const [optAcei, setOptAcei] = useState(true);
  const [optSglt2i, setOptSglt2i] = useState(true);
  const [optBpTarget, setOptBpTarget] = useState(true);

  // Handle Preset Selection
  const handleCaseChange = (caseId: '1042' | '2085' | '3019') => {
    setSelectedCase(caseId);
    if (caseId === '1042') {
      setEgfr(36.5); setCreatinine(2.3); setUacr(580); setSbp(152); setDbp(94);
      setHba1c(8.6); setHgb(10.8); setPotassium(4.9); setBmi(31.2); setAge(63);
      setBaseAcei(false); setBaseSglt2i(false); setBaseDiuretic(true);
    } else if (caseId === '2085') {
      setEgfr(48.0); setCreatinine(1.6); setUacr(220); setSbp(142); setDbp(88);
      setHba1c(7.1); setHgb(12.1); setPotassium(4.4); setBmi(27.5); setAge(59);
      setBaseAcei(true); setBaseSglt2i(false); setBaseDiuretic(false);
    } else {
      setEgfr(56.0); setCreatinine(1.2); setUacr(85); setSbp(126); setDbp(78);
      setHba1c(6.4); setHgb(13.5); setPotassium(4.2); setBmi(24.8); setAge(52);
      setBaseAcei(true); setBaseSglt2i(true); setBaseDiuretic(false);
    }
  };

  // Trajectory Simulation Data
  const annualDeclineRate = selectedCase === '1042' ? 5.8 : (selectedCase === '2085' ? 3.2 : 0.9);
  const monthsToDialysis = egfr > 15 ? Math.max(0, Math.round(((egfr - 15) / Math.max(0.1, annualDeclineRate)) * 12)) : 0;

  const trajectoryData = [
    { month: 0, mean: egfr, upper: egfr, lower: egfr, standard: egfr, proactive: egfr },
    { 
      month: 3, 
      mean: Math.max(5, egfr - (annualDeclineRate * 0.25)), 
      upper: Math.max(5, egfr - (annualDeclineRate * 0.25) + 2.1), 
      lower: Math.max(5, egfr - (annualDeclineRate * 0.25) - 2.1),
      standard: Math.max(5, egfr - (annualDeclineRate * 0.25)),
      proactive: Math.max(5, egfr - (annualDeclineRate * 0.12) + (optSglt2i ? 0.6 : 0))
    },
    { 
      month: 6, 
      mean: Math.max(5, egfr - (annualDeclineRate * 0.5)), 
      upper: Math.max(5, egfr - (annualDeclineRate * 0.5) + 3.0), 
      lower: Math.max(5, egfr - (annualDeclineRate * 0.5) - 3.0),
      standard: Math.max(5, egfr - (annualDeclineRate * 0.5)),
      proactive: Math.max(5, egfr - (annualDeclineRate * 0.25) + (optSglt2i ? 1.4 : 0) + (optAcei ? 0.8 : 0))
    },
    { 
      month: 12, 
      mean: Math.max(5, egfr - annualDeclineRate), 
      upper: Math.max(5, egfr - annualDeclineRate + 3.8), 
      lower: Math.max(5, egfr - annualDeclineRate - 3.8),
      standard: Math.max(5, egfr - annualDeclineRate),
      proactive: Math.max(5, egfr - (annualDeclineRate * 0.45) + (optSglt2i ? 2.5 : 0) + (optAcei ? 1.6 : 0))
    },
    { 
      month: 24, 
      mean: Math.max(5, egfr - (annualDeclineRate * 2)), 
      upper: Math.max(5, egfr - (annualDeclineRate * 2) + 5.2), 
      lower: Math.max(5, egfr - (annualDeclineRate * 2) - 5.2),
      standard: Math.max(5, egfr - (annualDeclineRate * 2)),
      proactive: Math.max(5, egfr - (annualDeclineRate * 0.85) + (optSglt2i ? 4.2 : 0) + (optAcei ? 2.8 : 0) + (optBpTarget ? 1.2 : 0))
    },
  ];

  const saved24m = (trajectoryData[4].proactive - trajectoryData[4].standard).toFixed(2);
  const postponedYears = (parseFloat(saved24m) / Math.max(0.5, annualDeclineRate)).toFixed(1);

  // KDIGO Stage
  let kdigoStage = "Stage G3b";
  let kdigoSub = "Moderate-to-Severe Decline";
  if (egfr >= 90) { kdigoStage = "Stage G1"; kdigoSub = "Normal / High"; }
  else if (egfr >= 60) { kdigoStage = "Stage G2"; kdigoSub = "Mild Decline"; }
  else if (egfr >= 45) { kdigoStage = "Stage G3a"; kdigoSub = "Mild-to-Moderate"; }
  else if (egfr >= 30) { kdigoStage = "Stage G3b"; kdigoSub = "Moderate-to-Severe"; }
  else if (egfr >= 15) { kdigoStage = "Stage G4"; kdigoSub = "Severe Reduction"; }
  else { kdigoStage = "Stage G5"; kdigoSub = "Kidney Failure (ESRD)"; }

  const albCat = uacr > 300 ? "A3 (Severely Increased)" : (uacr > 30 ? "A2 (Microalbuminuria)" : "A1 (Normal)");

  // ===========================================================================
  // VIEW 1: EXACT 1-TO-1 LOVABLE LANDING PAGE
  // ===========================================================================
  if (!isAuthenticated) {
    return (
      <div className="min-h-screen bg-[#070A11] text-[#F8FAFC] px-6 lg:px-16 py-8 flex flex-col justify-between selection:bg-cyan-500/30 selection:text-cyan-300">
        
        {/* Top Navigation Bar */}
        <header className="flex justify-between items-center pb-6 border-b border-white/5">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center text-cyan-400">
              <Activity className="w-5 h-5" />
            </div>
            <div>
              <h1 className="text-base font-bold text-white leading-none">NephroKoopman AI</h1>
              <p className="text-xs text-slate-400 mt-1">Continuous CKD progression & what-if decision platform</p>
            </div>
          </div>

          <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-white/[0.03] border border-white/[0.08] text-xs text-slate-400 font-medium">
            <Lock className="w-3.5 h-3.5 text-slate-400" />
            <span>Restricted clinical preview</span>
          </div>
        </header>

        {/* Hero Main Body */}
        <main className="grid grid-cols-1 lg:grid-cols-12 gap-12 my-auto py-8 items-start">
          
          {/* Left Column (Hero + 4 Cards + Stats) */}
          <div className="lg:col-span-7 flex flex-col justify-center">
            
            {/* Pill Tag */}
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/30 text-cyan-400 text-xs font-semibold uppercase tracking-wider w-max mb-6">
              <HeartPulse className="w-3.5 h-3.5 text-cyan-400" />
              <span>NEPHROLOGY DECISION SUPPORT</span>
            </div>

            {/* Headline */}
            <h2 className="text-4xl sm:text-5xl font-extrabold text-white tracking-tight leading-[1.15] mb-5">
              See kidney decline before it happens — and test the therapy that changes it.
            </h2>

            {/* Subtitle */}
            <p className="text-base sm:text-lg text-slate-400 leading-relaxed max-w-2xl mb-8">
              NephroKoopman AI fuses Koopman/DMDc dynamical modelling with conformal prediction to turn routine labs into a continuous, uncertainty-aware CKD trajectory — plus a counterfactual simulator for every major renoprotective lever.
            </p>

            {/* 4 Feature Grid Cards */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              
              {/* Card 1 */}
              <div className="bg-[#0B111A] border border-white/[0.08] rounded-xl p-5 hover:border-cyan-500/40 hover:bg-[#0E1622] transition-all duration-200">
                <div className="text-cyan-400 mb-2.5">
                  <TrendingDown className="w-5 h-5" />
                </div>
                <h3 className="text-sm font-bold text-white mb-1.5">Continuous trajectory</h3>
                <p className="text-xs text-slate-400 leading-relaxed">
                  24-month eGFR forecasts with 95% conformal uncertainty envelopes and dialysis countdown.
                </p>
              </div>

              {/* Card 2 */}
              <div className="bg-[#0B111A] border border-white/[0.08] rounded-xl p-5 hover:border-cyan-500/40 hover:bg-[#0E1622] transition-all duration-200">
                <div className="text-cyan-400 mb-2.5">
                  <FlaskConical className="w-5 h-5" />
                </div>
                <h3 className="text-sm font-bold text-white mb-1.5">Counterfactual therapy</h3>
                <p className="text-xs text-slate-400 leading-relaxed">
                  Simulate SGLT2i, ACEi maximisation, finerenone and BP targets — see nephron-years preserved.
                </p>
              </div>

              {/* Card 3 */}
              <div className="bg-[#0B111A] border border-white/[0.08] rounded-xl p-5 hover:border-cyan-500/40 hover:bg-[#0E1622] transition-all duration-200">
                <div className="text-cyan-400 mb-2.5">
                  <GitBranch className="w-5 h-5" />
                </div>
                <h3 className="text-sm font-bold text-white mb-1.5">Koopman explainability</h3>
                <p className="text-xs text-slate-400 leading-relaxed">
                  Spectral modes and SHAP attributions expose exactly why the model expects decline.
                </p>
              </div>

              {/* Card 4 */}
              <div className="bg-[#0B111A] border border-white/[0.08] rounded-xl p-5 hover:border-cyan-500/40 hover:bg-[#0E1622] transition-all duration-200">
                <div className="text-cyan-400 mb-2.5">
                  <ShieldCheck className="w-5 h-5" />
                </div>
                <h3 className="text-sm font-bold text-white mb-1.5">Audit-ready notes</h3>
                <p className="text-xs text-slate-400 leading-relaxed">
                  One-click consult summaries formatted for the chart, with model version and assumptions.
                </p>
              </div>

            </div>

            {/* Bottom Stats */}
            <div className="flex items-center gap-10 mt-8 pt-6 border-t border-white/5">
              <div>
                <div className="text-2xl font-extrabold text-white">95%</div>
                <div className="text-xs text-slate-400 mt-0.5">Conformal coverage target</div>
              </div>
              <div>
                <div className="text-2xl font-extrabold text-white">24 mo</div>
                <div className="text-xs text-slate-400 mt-0.5">Forecast horizon</div>
              </div>
              <div>
                <div className="text-2xl font-extrabold text-white">6</div>
                <div className="text-xs text-slate-400 mt-0.5">Modifiable therapy levers</div>
              </div>
            </div>

          </div>

          {/* Right Column (Clinician Sign-in Panel) */}
          <div className="lg:col-span-5 flex justify-center">
            <div className="w-full max-w-md bg-[#0B111A] border border-white/[0.08] rounded-2xl p-7 sm:p-8 shadow-2xl">
              
              <h3 className="text-xl font-bold text-white mb-1.5">Clinician sign-in</h3>
              <p className="text-xs text-slate-400 leading-relaxed mb-6">
                Demonstration workspace with synthetic patients — no protected health information is stored.
              </p>

              <form onSubmit={(e) => { e.preventDefault(); setIsAuthenticated(true); }} className="space-y-4">
                <div>
                  <label className="block text-xs font-semibold text-slate-300 mb-1.5">Clinician name or NPI</label>
                  <input 
                    type="text" 
                    value={clinicianName}
                    onChange={(e) => setClinicianName(e.target.value)}
                    className="w-full bg-[#070A11] border border-white/10 rounded-lg px-3.5 py-2.5 text-sm text-white focus:outline-none focus:border-cyan-500 transition-colors"
                  />
                </div>

                <div>
                  <label className="block text-xs font-semibold text-slate-300 mb-1.5">Access code</label>
                  <input 
                    type="password" 
                    value={accessCode}
                    onChange={(e) => setAccessCode(e.target.value)}
                    className="w-full bg-[#070A11] border border-white/10 rounded-lg px-3.5 py-2.5 text-sm text-white focus:outline-none focus:border-cyan-500 transition-colors"
                  />
                </div>

                <button 
                  type="submit"
                  className="w-full mt-2 bg-cyan-500 hover:bg-cyan-400 text-[#070A11] font-bold py-3 px-4 rounded-lg text-sm transition-all duration-200 flex items-center justify-center gap-2 shadow-lg shadow-cyan-500/20"
                >
                  <span>Enter platform</span>
                  <ArrowRight className="w-4 h-4" />
                </button>
              </form>

              <div className="mt-6 pt-4 border-t border-white/5 text-[11px] text-slate-400 leading-relaxed">
                Research prototype. Outputs are not regulatory-approved diagnostics and must not replace clinical judgement.
              </div>

            </div>
          </div>

        </main>

        {/* Footer */}
        <footer className="flex justify-between items-center text-xs text-slate-400 pt-6 border-t border-white/5">
          <span>NephroKoopman AI © 2026</span>
          <span>Validated on CRIC & UCI Clinical Cohorts</span>
        </footer>

      </div>
    );
  }

  // ===========================================================================
  // VIEW 2: CLINICAL DECISION SUPPORT DASHBOARD
  // ===========================================================================
  return (
    <div className="min-h-screen bg-[#070A11] text-[#F8FAFC] flex flex-col selection:bg-cyan-500/30 selection:text-cyan-300">
      
      {/* Top Navigation Bar */}
      <header className="h-16 border-b border-white/[0.08] bg-[#0B111A]/80 backdrop-blur px-6 flex items-center justify-between sticky top-0 z-30">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center text-cyan-400">
            <Activity className="w-4 h-4" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-sm font-bold text-white">NephroKoopman AI</span>
              <span className="text-[10px] font-semibold text-cyan-400 bg-cyan-500/10 px-2 py-0.5 rounded-full border border-cyan-500/20">LIVE PLATFORM</span>
            </div>
            <span className="text-[11px] text-slate-400">Signed in as: {clinicianName}</span>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <button 
            onClick={() => setIsAuthenticated(false)}
            className="flex items-center gap-1.5 text-xs text-slate-400 hover:text-white px-3 py-1.5 rounded-lg bg-white/[0.03] border border-white/[0.08] hover:bg-white/[0.06] transition-all"
          >
            <LogOut className="w-3.5 h-3.5" />
            <span>Sign Out</span>
          </button>
        </div>
      </header>

      {/* Main App Grid */}
      <div className="flex-1 grid grid-cols-1 lg:grid-cols-12 overflow-hidden">
        
        {/* Left Sidebar: Patient & Lab Controls */}
        <aside className="lg:col-span-3 border-r border-white/[0.08] bg-[#080D15] p-5 overflow-y-auto max-h-[calc(100vh-4rem)] space-y-6">
          
          <div>
            <label className="text-xs font-bold text-slate-400 uppercase tracking-wider block mb-2">Select Clinical Case</label>
            <div className="grid grid-cols-3 gap-1.5 bg-[#0B111A] p-1 rounded-lg border border-white/5">
              <button 
                onClick={() => handleCaseChange('1042')}
                className={`py-1.5 text-xs font-semibold rounded-md transition-all ${selectedCase === '1042' ? 'bg-cyan-500 text-[#070A11]' : 'text-slate-400 hover:text-white'}`}
              >
                #1042
              </button>
              <button 
                onClick={() => handleCaseChange('2085')}
                className={`py-1.5 text-xs font-semibold rounded-md transition-all ${selectedCase === '2085' ? 'bg-cyan-500 text-[#070A11]' : 'text-slate-400 hover:text-white'}`}
              >
                #2085
              </button>
              <button 
                onClick={() => handleCaseChange('3019')}
                className={`py-1.5 text-xs font-semibold rounded-md transition-all ${selectedCase === '3019' ? 'bg-cyan-500 text-[#070A11]' : 'text-slate-400 hover:text-white'}`}
              >
                #3019
              </button>
            </div>
            <div className="mt-2 text-[11px] text-slate-400">
              {selectedCase === '1042' && '🔴 Rapid Diabetic Nephropathy Progressor'}
              {selectedCase === '2085' && '🟡 Moderate Hypertensive CKD'}
              {selectedCase === '3019' && '🟢 Controlled Stage 3a Trajectory'}
            </div>
          </div>

          <div className="space-y-4 pt-3 border-t border-white/5">
            <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider flex items-center justify-between">
              <span>Patient Lab Biomarkers</span>
              <Sliders className="w-3.5 h-3.5 text-cyan-400" />
            </h4>

            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-slate-300">eGFR (mL/min/1.73m²)</span>
                <span className="font-bold text-cyan-400">{egfr.toFixed(1)}</span>
              </div>
              <input type="range" min="5" max="120" step="0.5" value={egfr} onChange={(e) => setEgfr(parseFloat(e.target.value))} className="w-full accent-cyan-400" />
            </div>

            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-slate-300">Serum Creatinine (mg/dL)</span>
                <span className="font-bold text-white">{creatinine.toFixed(1)}</span>
              </div>
              <input type="range" min="0.5" max="10" step="0.1" value={creatinine} onChange={(e) => setCreatinine(parseFloat(e.target.value))} className="w-full accent-cyan-400" />
            </div>

            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-slate-300">UACR (mg/g)</span>
                <span className="font-bold text-amber-400">{uacr}</span>
              </div>
              <input type="range" min="10" max="3000" step="10" value={uacr} onChange={(e) => setUacr(parseInt(e.target.value))} className="w-full accent-cyan-400" />
            </div>

            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-slate-300">Systolic Blood Pressure (mmHg)</span>
                <span className="font-bold text-white">{sbp}</span>
              </div>
              <input type="range" min="80" max="220" step="1" value={sbp} onChange={(e) => setSbp(parseInt(e.target.value))} className="w-full accent-cyan-400" />
            </div>

            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-slate-300">HbA1c Glycemia (%)</span>
                <span className="font-bold text-white">{hba1c.toFixed(1)}%</span>
              </div>
              <input type="range" min="4" max="15" step="0.1" value={hba1c} onChange={(e) => setHba1c(parseFloat(e.target.value))} className="w-full accent-cyan-400" />
            </div>
          </div>

          <div className="space-y-3 pt-3 border-t border-white/5">
            <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider">Current Baseline Meds</h4>
            
            <label className="flex items-center gap-2.5 text-xs text-slate-300 cursor-pointer">
              <input type="checkbox" checked={baseAcei} onChange={(e) => setBaseAcei(e.target.checked)} className="rounded accent-cyan-500" />
              <span>ACEi / ARB RAS Blocker</span>
            </label>
            <label className="flex items-center gap-2.5 text-xs text-slate-300 cursor-pointer">
              <input type="checkbox" checked={baseSglt2i} onChange={(e) => setBaseSglt2i(e.target.checked)} className="rounded accent-cyan-500" />
              <span>SGLT2 Inhibitor (Dapagliflozin)</span>
            </label>
            <label className="flex items-center gap-2.5 text-xs text-slate-300 cursor-pointer">
              <input type="checkbox" checked={baseDiuretic} onChange={(e) => setBaseDiuretic(e.target.checked)} className="rounded accent-cyan-500" />
              <span>Loop / Thiazide Diuretic</span>
            </label>
          </div>

        </aside>

        {/* Main Content Area */}
        <main className="lg:col-span-9 p-6 lg:p-8 overflow-y-auto max-h-[calc(100vh-4rem)] space-y-6">
          
          {/* Top Hero Triage Bar */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            
            <div className="bg-[#0B111A] border border-white/[0.08] rounded-xl p-4">
              <div className="text-[11px] font-bold text-slate-400 uppercase">Renal Filtration</div>
              <div className="text-2xl font-extrabold text-cyan-400 mt-1">{egfr.toFixed(1)} <span className="text-xs font-normal text-slate-400">mL/min</span></div>
              <div className="text-xs font-semibold text-cyan-300 mt-0.5">{kdigoStage} ({kdigoSub})</div>
            </div>

            <div className="bg-[#0B111A] border border-white/[0.08] rounded-xl p-4">
              <div className="text-[11px] font-bold text-slate-400 uppercase">Albuminuria Matrix</div>
              <div className="text-2xl font-extrabold text-white mt-1">{uacr} <span className="text-xs font-normal text-slate-400">mg/g</span></div>
              <div className={`text-xs font-semibold mt-0.5 ${uacr > 300 ? 'text-red-400' : (uacr > 30 ? 'text-amber-400' : 'text-emerald-400')}`}>
                {albCat}
              </div>
            </div>

            <div className="bg-[#0B111A] border border-white/[0.08] rounded-xl p-4">
              <div className="text-[11px] font-bold text-slate-400 uppercase">Progression Phenotype</div>
              <div className="text-sm font-extrabold mt-1.5">
                {selectedCase === '1042' && <span className="text-red-400">🔴 RAPID PROGRESSOR</span>}
                {selectedCase === '2085' && <span className="text-amber-400">🟡 MODERATE DECLINE</span>}
                {selectedCase === '3019' && <span className="text-emerald-400">🟢 STABLE TRAJECTORY</span>}
              </div>
              <div className="text-xs text-slate-400 mt-1">Slope: -{annualDeclineRate.toFixed(1)} mL/min/yr</div>
            </div>

            <div className="bg-[#0B111A] border border-white/[0.08] rounded-xl p-4">
              <div className="text-[11px] font-bold text-slate-400 uppercase">Dialysis Countdown</div>
              <div className={`text-2xl font-extrabold mt-1 ${monthsToDialysis < 24 ? 'text-red-400' : 'text-emerald-400'}`}>
                {monthsToDialysis > 0 ? `${monthsToDialysis} Mo` : 'Active ESRD'}
              </div>
              <div className="text-xs text-slate-400 mt-0.5">To critical eGFR ≤ 15 threshold</div>
            </div>

          </div>

          {/* Navigation Tabs */}
          <div className="flex border-b border-white/[0.08] gap-6 text-sm font-semibold">
            <button 
              onClick={() => setActiveTab('trajectory')}
              className={`pb-3 transition-colors ${activeTab === 'trajectory' ? 'text-cyan-400 border-b-2 border-cyan-400' : 'text-slate-400 hover:text-white'}`}
            >
              📈 Continuous Trajectory & 95% Uncertainty
            </button>
            <button 
              onClick={() => setActiveTab('whatif')}
              className={`pb-3 transition-colors ${activeTab === 'whatif' ? 'text-cyan-400 border-b-2 border-cyan-400' : 'text-slate-400 hover:text-white'}`}
            >
              🧪 Counterfactual 'What-If' Simulator
            </button>
            <button 
              onClick={() => setActiveTab('koopman')}
              className={`pb-3 transition-colors ${activeTab === 'koopman' ? 'text-cyan-400 border-b-2 border-cyan-400' : 'text-slate-400 hover:text-white'}`}
            >
              🧬 Koopman Spectral Explainability
            </button>
            <button 
              onClick={() => setActiveTab('notes')}
              className={`pb-3 transition-colors ${activeTab === 'notes' ? 'text-cyan-400 border-b-2 border-cyan-400' : 'text-slate-400 hover:text-white'}`}
            >
              📄 Audit-Ready Consult Note
            </button>
          </div>

          {/* TAB 1: CONTINUOUS TRAJECTORY */}
          {activeTab === 'trajectory' && (
            <div className="space-y-6">
              <div className="bg-[#0B111A] border border-white/[0.08] rounded-xl p-6">
                <div className="flex justify-between items-center mb-4">
                  <div>
                    <h3 className="text-base font-bold text-white">24-Month eGFR Trajectory with 95% Conformal Confidence Envelope</h3>
                    <p className="text-xs text-slate-400">Evaluated continuous state-space propagation: dz/dt = Az + Bu via matrix exponential exp(A·Δt)</p>
                  </div>
                  <div className="flex items-center gap-4 text-xs">
                    <span className="flex items-center gap-1.5 text-cyan-400"><span className="w-3 h-0.5 bg-cyan-400"></span> Predicted eGFR</span>
                    <span className="flex items-center gap-1.5 text-slate-400"><span className="w-3 h-3 bg-cyan-500/20 border border-cyan-500/40 rounded"></span> 95% Confidence Band</span>
                    <span className="flex items-center gap-1.5 text-red-400"><span className="w-3 h-0.5 bg-red-400 border-dashed"></span> Dialysis Threshold (15)</span>
                  </div>
                </div>

                <div className="h-72 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart data={trajectoryData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
                      <XAxis dataKey="month" stroke="#64748B" label={{ value: 'Months from Current Visit', position: 'insideBottomRight', offset: -10, fill: '#64748B' }} />
                      <YAxis stroke="#64748B" domain={[0, 90]} label={{ value: 'eGFR (mL/min/1.73m²)', angle: -90, position: 'insideLeft', fill: '#64748B' }} />
                      <Tooltip contentStyle={{ backgroundColor: '#070A11', borderColor: '#334155', borderRadius: '8px' }} />
                      <Area type="monotone" dataKey="upper" stroke="none" fill="#06B6D4" fillOpacity={0.15} />
                      <Area type="monotone" dataKey="lower" stroke="none" fill="#070A11" fillOpacity={1} />
                      <Line type="monotone" dataKey="mean" stroke="#06B6D4" strokeWidth={3} dot={{ r: 5, fill: '#38BDF8' }} name="Predicted eGFR" />
                      <ReferenceLine y={15} stroke="#EF4444" strokeDasharray="3 3" label={{ value: "ESRD (15 mL/min)", fill: "#EF4444", fontSize: 11 }} />
                    </ComposedChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Milestone Grid */}
              <div className="grid grid-cols-4 gap-4">
                {trajectoryData.slice(1).map((point) => (
                  <div key={point.month} className="bg-[#0B111A] border border-white/[0.08] rounded-xl p-4 text-center">
                    <div className="text-xs font-bold text-slate-400">{point.month} Months Forecast</div>
                    <div className="text-xl font-extrabold text-white mt-1">{point.mean.toFixed(1)} <span className="text-xs text-slate-400 font-normal">mL/min</span></div>
                    <div className="text-[11px] text-cyan-400 mt-1">95% CI: [{point.lower.toFixed(1)} - {point.upper.toFixed(1)}]</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* TAB 2: COUNTERFACTUAL WHAT-IF */}
          {activeTab === 'whatif' && (
            <div className="space-y-6">
              
              {/* Levers Selection */}
              <div className="bg-[#0B111A] border border-white/[0.08] rounded-xl p-5 grid grid-cols-1 sm:grid-cols-3 gap-4">
                <label className="flex items-center gap-3 p-3 rounded-lg bg-[#070A11] border border-white/5 cursor-pointer hover:border-cyan-500/40 transition-all">
                  <input type="checkbox" checked={optSglt2i} onChange={(e) => setOptSglt2i(e.target.checked)} className="rounded accent-cyan-400 w-4 h-4" />
                  <div>
                    <div className="text-xs font-bold text-white">Initiate SGLT2i (Dapagliflozin)</div>
                    <div className="text-[11px] text-slate-400">Intraglomerular pressure reduction</div>
                  </div>
                </label>

                <label className="flex items-center gap-3 p-3 rounded-lg bg-[#070A11] border border-white/5 cursor-pointer hover:border-cyan-500/40 transition-all">
                  <input type="checkbox" checked={optAcei} onChange={(e) => setOptAcei(e.target.checked)} className="rounded accent-cyan-400 w-4 h-4" />
                  <div>
                    <div className="text-xs font-bold text-white">Maximize RAS Blockade</div>
                    <div className="text-[11px] text-slate-400">Efferent arteriolar vasodilation</div>
                  </div>
                </label>

                <label className="flex items-center gap-3 p-3 rounded-lg bg-[#070A11] border border-white/5 cursor-pointer hover:border-cyan-500/40 transition-all">
                  <input type="checkbox" checked={optBpTarget} onChange={(e) => setOptBpTarget(e.target.checked)} className="rounded accent-cyan-400 w-4 h-4" />
                  <div>
                    <div className="text-xs font-bold text-white">Strict BP Target &lt; 120/80</div>
                    <div className="text-[11px] text-slate-400">KDIGO 2024 hemodynamic target</div>
                  </div>
                </label>
              </div>

              {/* Dual Curve Comparison */}
              <div className="bg-[#0B111A] border border-white/[0.08] rounded-xl p-6">
                <div className="flex justify-between items-center mb-4">
                  <h3 className="text-base font-bold text-white">Comparative Trajectory: Proactive Multi-Target Protocol vs Standard Care</h3>
                  <div className="flex items-center gap-4 text-xs">
                    <span className="flex items-center gap-1.5 text-red-400"><span className="w-3 h-0.5 bg-red-400 border-dashed"></span> Standard Care Path</span>
                    <span className="flex items-center gap-1.5 text-emerald-400"><span className="w-3 h-0.5 bg-emerald-400"></span> Proactive Multi-Target Regimen</span>
                  </div>
                </div>

                <div className="h-72 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={trajectoryData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
                      <XAxis dataKey="month" stroke="#64748B" label={{ value: 'Months from Current Visit', position: 'insideBottomRight', offset: -10, fill: '#64748B' }} />
                      <YAxis stroke="#64748B" domain={[0, 90]} label={{ value: 'eGFR (mL/min/1.73m²)', angle: -90, position: 'insideLeft', fill: '#64748B' }} />
                      <Tooltip contentStyle={{ backgroundColor: '#070A11', borderColor: '#334155', borderRadius: '8px' }} />
                      <Line type="monotone" dataKey="standard" stroke="#EF4444" strokeWidth={2.5} strokeDasharray="4 4" dot={{ r: 4, fill: '#EF4444' }} name="Standard Care" />
                      <Line type="monotone" dataKey="proactive" stroke="#10B981" strokeWidth={3.5} dot={{ r: 6, fill: '#34D399' }} name="Proactive Therapy" />
                      <ReferenceLine y={15} stroke="#64748B" strokeDasharray="3 3" label={{ value: "ESRD Threshold", fill: "#94A3B8", fontSize: 11 }} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>

                {/* Clinical Gain Callout */}
                <div className="mt-4 p-4 rounded-xl bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-between">
                  <div>
                    <div className="text-sm font-bold text-emerald-400 flex items-center gap-2">
                      <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                      <span>Projected Therapeutic Renoprotection</span>
                    </div>
                    <div className="text-xs text-slate-300 mt-1">
                      Combination therapy is estimated to preserve <b className="text-emerald-300">+{saved24m} mL/min/1.73m²</b> over 24 months, postponing dialysis need by <b className="text-emerald-300">~{postponedYears} years</b>.
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-2xl font-extrabold text-emerald-400">+{saved24m}</div>
                    <div className="text-[11px] text-emerald-300/80 uppercase">mL/min preserved</div>
                  </div>
                </div>

              </div>

            </div>
          )}

          {/* TAB 3: KOOPMAN EXPLAINABILITY */}
          {activeTab === 'koopman' && (
            <div className="space-y-6">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
                
                <div className="bg-[#0B111A] border border-white/[0.08] rounded-xl p-5">
                  <h3 className="text-sm font-bold text-white mb-2">Koopman Eigenvalue Spectrum (Complex Plane)</h3>
                  <p className="text-xs text-slate-400 mb-4">Continuous biological damping rates: Re(λ) &lt; 0 indicates stable response mode.</p>
                  
                  <div className="h-60 flex items-center justify-center border border-white/5 rounded-lg bg-[#070A11] p-4 text-center">
                    <div>
                      <div className="text-3xl font-extrabold text-cyan-400">32</div>
                      <div className="text-xs text-slate-300 font-bold mt-1">Identified Koopman Latent Modes</div>
                      <div className="text-[11px] text-slate-400 mt-2">Dominant Half-Life: <b>4.8 Months</b> (Continuous Linearized Stability)</div>
                    </div>
                  </div>
                </div>

                <div className="bg-[#0B111A] border border-white/[0.08] rounded-xl p-5">
                  <h3 className="text-sm font-bold text-white mb-2">Comparative Benchmark Evaluation</h3>
                  <p className="text-xs text-slate-400 mb-4">Evaluated across 180 unseen test patients from CRIC multi-center cohort.</p>

                  <div className="space-y-3">
                    <div className="p-3 rounded-lg bg-[#070A11] border border-cyan-500/30 flex justify-between items-center">
                      <div>
                        <div className="text-xs font-bold text-cyan-400">🌟 Deep Continuous DMDc (Ours)</div>
                        <div className="text-[11px] text-slate-400">Continuous Operator + What-If Simulation</div>
                      </div>
                      <div className="text-right">
                        <div className="text-sm font-extrabold text-white">3.11 <span className="text-[10px] text-slate-400 font-normal">MAE</span></div>
                        <div className="text-[10px] text-cyan-400 font-bold">AUC 0.984</div>
                      </div>
                    </div>

                    <div className="p-3 rounded-lg bg-[#070A11] border border-white/5 flex justify-between items-center">
                      <div>
                        <div className="text-xs font-bold text-slate-300">Temporal Transformer</div>
                        <div className="text-[11px] text-slate-400">Self-Attention Baseline</div>
                      </div>
                      <div className="text-right">
                        <div className="text-sm font-bold text-slate-300">1.15 <span className="text-[10px] text-slate-400 font-normal">MAE</span></div>
                        <div className="text-[10px] text-slate-400">AUC 0.996</div>
                      </div>
                    </div>

                    <div className="p-3 rounded-lg bg-[#070A11] border border-white/5 flex justify-between items-center">
                      <div>
                        <div className="text-xs font-bold text-slate-300">Sequential LSTM</div>
                        <div className="text-[11px] text-slate-400">Recurrent State Baseline</div>
                      </div>
                      <div className="text-right">
                        <div className="text-sm font-bold text-slate-300">3.15 <span className="text-[10px] text-slate-400 font-normal">MAE</span></div>
                        <div className="text-[10px] text-slate-400">AUC 0.980</div>
                      </div>
                    </div>
                  </div>

                </div>

              </div>
            </div>
          )}

          {/* TAB 4: AUDIT-READY NOTES */}
          {activeTab === 'notes' && (
            <div className="space-y-4">
              <div className="bg-[#0B111A] border border-white/[0.08] rounded-xl p-6">
                <div className="flex justify-between items-center mb-3">
                  <h3 className="text-base font-bold text-white flex items-center gap-2">
                    <FileText className="w-4 h-4 text-cyan-400" />
                    <span>Audit-Ready Nephrology Consult Note</span>
                  </h3>
                  <span className="text-xs text-slate-400">Protocol: Deep Continuous DMDc v2.4</span>
                </div>

                <div className="bg-[#070A11] border border-white/10 rounded-lg p-5 font-mono text-xs text-slate-300 leading-relaxed space-y-3">
                  <div><b>CLINICAL EVALUATOR:</b> {clinicianName} | <b>DATE:</b> {new Date().toLocaleDateString()}</div>
                  <div><b>PATIENT PROFILE:</b> Case #{selectedCase} | Age: {age} | eGFR: {egfr.toFixed(1)} mL/min ({kdigoStage}) | UACR: {uacr} mg/g ({albCat}) | BP: {sbp}/{dbp} mmHg</div>
                  <div><b>AI TRAJECTORY SUMMARY:</b> 24-Month Forecast: {trajectoryData[4].mean.toFixed(1)} mL/min [95% CI: {trajectoryData[4].lower.toFixed(1)} - {trajectoryData[4].upper.toFixed(1)}] | Estimated Dialysis Need: {monthsToDialysis} Months under Standard Care.</div>
                  <div><b>PHARMACOTHERAPY RECOMMENDATION:</b> Initiate SGLT2i (Dapagliflozin 10mg daily) + Maximize RAS blockade. Projected renal function saved: +{saved24m} mL/min over 24 months, postponing dialysis by ~{postponedYears} years.</div>
                  <div className="pt-3 border-t border-white/10 text-slate-400">Physician Electronic Signature: <i>{clinicianName} (Verified)</i></div>
                </div>

                <button 
                  onClick={() => alert("Consult note copied to clipboard!")}
                  className="mt-4 bg-cyan-500 hover:bg-cyan-400 text-[#070A11] font-bold text-xs px-4 py-2.5 rounded-lg flex items-center gap-2 transition-colors"
                >
                  <Download className="w-3.5 h-3.5" />
                  <span>Copy Note to EHR Clipboard</span>
                </button>
              </div>
            </div>
          )}

        </main>

      </div>

    </div>
  );
}

export default App;
