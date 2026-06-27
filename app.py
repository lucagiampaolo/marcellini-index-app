"""
Marcellini Index DM2 — Calcolatore Clinico V18
AUSL Romagna Rimini · Uso riservato al team di ricerca
"""

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import json
import math
import os
from pathlib import Path
import plotly.graph_objects as go

# ════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Marcellini Index DM2",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS minimale ──────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background: #F1F5F9; }
    [data-testid="stSidebar"] { background: #0F2D54; }
    .block-container { padding-top: 1rem; }
    div[data-testid="metric-container"] > div { font-size: 13px; }
    hr { margin: 6px 0; }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# AUTH
# ════════════════════════════════════════════════════════════════════════════
def check_password() -> bool:
    try:
        pwd = st.secrets.get("password")
    except Exception:
        pwd = None

    if pwd is None:
        return True  # dev mode: no password set

    if st.session_state.get("authenticated"):
        return True

    st.markdown("## 🩺 Marcellini Index DM2")
    st.markdown("**Calcolatore clinico — AUSL Romagna Rimini · Uso riservato**")
    st.divider()
    entered = st.text_input("Password di accesso:", type="password", key="pwd_input")
    if st.button("Accedi", type="primary"):
        if entered == pwd:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("❌ Password errata.")
    return False

if not check_password():
    st.stop()

# ════════════════════════════════════════════════════════════════════════════
# COSTANTI
# ════════════════════════════════════════════════════════════════════════════
BASE_DIR   = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR   = BASE_DIR / "data"

FEATURES_BASE = [
    "EtaBasale", "Sesso", "EtaDiabeteMellito",
    "HbA1cBasale", "HbA1c_ER_target_basale",
    "eGFR_basale", "ASCVD_basale", "TOD_ER_max_basale",
    "DCSI_basale", "DrugScore_basale",
    "TerapiaRischioIpo", "Insulina_multiniettiva_basale_trans",
]

# Domini DCSI: (id, emoji, etichetta, punteggio_max, è_TOD, è_ASCVD)
DCSI_DOMAINS = [
    ("nefropatia",          "🫘", "Nefropatia",              3, True,  False),
    ("retinopatia",         "👁",  "Retinopatia",             2, True,  False),
    ("neuropatia",          "⚡",  "Neuropatia periferica",   2, True,  False),
    ("cardiovascolare",     "🫀",  "Cardiopatia ischemica",   3, True,  True),
    ("cerebrovascolare",    "🧠",  "Vasculopatia cerebrale",  2, True,  True),
    ("vascolare_periferica","🦵",  "Arteriopatia periferica", 2, True,  True),
    ("metabolica",          "⚗️",  "Crisi metaboliche",       2, False, False),
]

# Descrizioni cliniche per ogni dominio DCSI × grado
# Fonte: Tabella 2 — Linee di Indirizzo ER 2017 (gestione integrata DM2)
DCSI_DESC: dict[str, dict[int, str]] = {
    "nefropatia": {
        0: "Assente — eGFR ≥60, normoalbuminuria",
        1: "Lieve — Microalbuminuria (20–200 μg/ml), GFR ≥60 ml/min",
        2: "Moderata-Severa — Macroalbuminuria (>200 μg/ml) oppure GFR 30–59 ml/min",
        3: "Severa — GFR <30 ml/min oppure dialisi",
    },
    "retinopatia": {
        0: "Assente — fondo oculare negativo",
        1: "Lieve — rari microaneurismi e/o microemorragie",
        2: "Moderata-Severa — microaneurismi numerosi o maculari, essudati, IRMA, "
           "edema maculare, aree ischemiche, proliferazione di neovasi",
    },
    "neuropatia": {
        0: "Assente — esame neurologico nella norma",
        1: "Lieve — parestesie lievi e transitorie agli arti inferiori",
        2: "Moderata-Severa — parestesie dolorose, deficit sensitivo-motorio "
           "clinicamente evidenziabile, disautonomia",
    },
    "cardiovascolare": {
        0: "Assente — nessuna storia di cardiopatia ischemica",
        1: "Lieve (stabile) — nota, clinicamente stabile e compensata, "
           "monitorata in ambiente specialistico",
        2: "Moderata (instabile) — primo esordio clinico oppure già nota clinicamente instabile",
        3: "Severa — scompenso cardiaco congestizio, cardiopatia severa",
    },
    "cerebrovascolare": {
        0: "Assente — nessun evento cerebrovascolare",
        1: "Lieve (stabile) — TIA o ictus noti, stabile e compensata",
        2: "Moderata-Severa — primo riscontro oppure già nota e clinicamente instabile",
    },
    "vascolare_periferica": {
        0: "Assente — polsi periferici normali, no claudicatio",
        1: "Lieve (stabile) — claudicatio intermittens, no lesioni trofiche, stabile",
        2: "Moderata-Severa — primo esordio, o in evoluzione, dolore a riposo, "
           "lesioni trofiche / esiti di amputazione",
    },
    "metabolica": {
        0: "Assente — nessuna crisi metabolica acuta in anamnesi",
        1: "Episodio risolto — DKA o coma iperosmolare pregresso, risolto",
        2: "Ricorrente — ipoglicemie gravi recidivanti o DKA/iperosmolare ricorrente",
    },
}

# Farmaci dose-dipendenti
DRUGS_VARIABLE = {
    "Metformina":     {"unit": "mg/die",   "t": [(999,7),(1499,9),(1999,12),(1e9,13)]},
    "Dapagliflozin":  {"unit": "mg/die",   "t": [(9,9),(1e9,11)]},
    "Empagliflozin":  {"unit": "mg/die",   "t": [(24,8),(1e9,9)]},
    "Canagliflozin":  {"unit": "mg/die",   "t": [(299,10),(1e9,12)]},
    "Ertugliflozin":  {"unit": "mg/die",   "t": [(14,11),(1e9,13)]},
    "Liraglutide":    {"unit": "mg/die",   "t": [(1.79,10),(1e9,13)]},
    "Semaglutide SC": {"unit": "mg/sett",  "t": [(0.99,16),(1.99,17),(1e9,20)]},
    "Semaglutide OS": {"unit": "mg/die",   "t": [(6,7),(13,11),(1e9,13)]},
    "Dulaglutide":    {"unit": "mg/sett",  "t": [(1.49,12),(2.99,17),(4.49,19),(1e9,20)]},
    "Tirzepatide":    {"unit": "mg/sett",  "t": [(9,22),(14,24),(1e9,25)]},
}

# Farmaci punteggio fisso
DRUGS_FIXED_CATS = {
    "💉 Insuline basali": {
        "items":  ["Insulina Degludec","Insulina Glargine","Insulina Detemir","Insulina NPH"],
        "scores": [18, 17, 17, 17],
        "cat":    "ib",
    },
    "💉 Insuline rapide": {
        "items":  ["Insulina Aspart","Insulina Lispro","Insulina Glulisina","Insulina Umana Reg."],
        "scores": [11, 11, 11, 11],
        "cat":    "ir",
    },
    "🧬 GLP-1 iniettivi (fissi)": {
        "items":  ["Exenatide","Lixisenatide"],
        "scores": [21, 7],
        "cat":    "glp",
    },
    "🔵 DPP-4 inibitori": {
        "items":  ["Sitagliptin","Saxagliptin","Linagliptin","Alogliptin","Vildagliptin"],
        "scores": [7, 7, 7, 11, 10],
        "cat":    "dpp",
    },
    "🟠 Sulfoniluree": {
        "items":  ["Gliclazide","Glimepiride","Glibenclamide","Glipizide","Gliquidone"],
        "scores": [14, 14, 14, 14, 14],
        "cat":    "su",
    },
    "🟢 Altri": {
        "items":  ["Repaglinide","Pioglitazone","Acarbose","Fenformina"],
        "scores": [12, 11, 9, 7],
        "cat":    "alt",
    },
}

# Categorie per derivazione flag terapia
INS_BASALE = ["Insulina Degludec","Insulina Glargine","Insulina Detemir","Insulina NPH"]
INS_RAPIDA = ["Insulina Aspart","Insulina Lispro","Insulina Glulisina","Insulina Umana Reg."]
SULFONILUR = ["Gliclazide","Glimepiride","Glibenclamide","Glipizide","Gliquidone"]

# Thresholds (real Youden from N6_Youden_Completa)
# "event_label": descrizione dell'evento predetto
# "time":        orizzonte temporale
# "direction":   "bad" = p alta è peggio, "good" = p alta è meglio
THRESHOLDS = {
    "D1": {
        "outcome":     "Target_Gravi_5a",
        "label":       "D1 — Rischio strutturale",
        "event_label": "evento grave composito\n(decesso o progressione DCSI≥2 o ASCVD incidente)",
        "time":        "5 anni",
        "direction":   "bad",
        "soglia":0.402, "auc":0.6689, "ci_lo":0.6555, "ci_hi":0.6811,
        "sens":77.5, "spec":46.9, "ppv":55.3, "npv":71.1, "n":7190,
        "invert": False,
    },
    "D2": {
        "outcome":     "TIR_target_3a",
        "label":       "D2 — Controllo metabolico",
        "event_label": "raggiungimento TIR ≥70%\n(tempo in range glicemico ottimale)",
        "time":        "3 anni",
        "direction":   "good",
        "soglia":0.423, "auc":0.7204, "ci_lo":0.7091, "ci_hi":0.7328,
        "sens":61.4, "spec":72.3, "ppv":62.5, "npv":71.4, "n":7856,
        "invert": True,
    },
    "D3": {
        "outcome":     "Trans_MMG_6m",
        "label":       "D3 — Trasferibilità MMG",
        "event_label": "transizione stabile a MMG ≥6 mesi\n(gestione integrata MMG-Diabetologo)",
        "time":        "follow-up disponibile",
        "direction":   "good",
        "soglia":0.211, "auc":0.892, "ci_lo":0.8851, "ci_hi":0.8991,
        "sens":96.5, "spec":72.8, "ppv":53.9, "npv":98.4, "n":7917,
        "invert": True,
    },
    "Mort": {
        "outcome":     "Decesso_L_5a",
        "label":       "Mortalità",
        "event_label": "decesso per qualsiasi causa",
        "time":        "5 anni",
        "direction":   "bad",
        "soglia":0.111, "auc":0.8289, "ci_lo":0.8156, "ci_hi":0.8411,
        "sens":79.8, "spec":69.8, "ppv":28.1, "npv":95.9, "n":7188,
        "invert": False,
    },
    "DCSI": {
        "outcome":     "DCSI_0to2_5a",
        "label":       "Progressione DCSI",
        "event_label": "progressione DCSI ≥2 punti\n(solo pazienti senza complicanze a baseline)",
        "time":        "5 anni",
        "direction":   "bad",
        "soglia":0.361, "auc":0.745, "ci_lo":0.7249, "ci_hi":0.764,
        "sens":62.4, "spec":73.7, "ppv":53.2, "npv":80.4, "n":3061,
        "invert": False,
    },
    "dDCSI": {
        "outcome":     "DeltaDCSI_gte2_5a",
        "label":       "ΔDCSI — Intera coorte",
        "event_label": "incremento DCSI ≥2 punti\n(intera coorte, DCSI basale ≤10)",
        "time":        "5 anni",
        "direction":   "bad",
        "soglia":0.265, "auc":0.7252, "ci_lo":0.7122, "ci_hi":0.7381,
        "sens":73.2, "spec":60.1, "ppv":43.8, "npv":84.1, "n":6261,
        "invert": False,
    },
}
KM6_CENTROIDS = np.array([
    [0.271, 0.452, 0.597],  # A — Basso rischio · Alta mobilità MMG
    [0.389, 0.777, 0.172],  # B — Basso-mod. · Ottimo metabolismo
    [0.421, 0.330, 0.028],  # C — Moderato · Scarso metabolismo · Specialistica
    [0.517, 0.368, 0.485],  # D — Moderato-alto · Media mobilità
    [0.655, 0.419, 0.128],  # E — Alto rischio · Bassa mobilità
    [0.693, 0.579, 0.040],  # F — Massima complessità
])
KM6_NAMES = [
    ("A", "#059669", "Basso rischio · Alta mobilità MMG"),
    ("B", "#0D9488", "Basso-mod. · Ottimo controllo metabolico"),
    ("C", "#D97706", "Moderato · Scarso metabolismo · Specialistica"),
    ("D", "#EA580C", "Moderato-alto · Media mobilità"),
    ("E", "#DC2626", "Alto rischio · Bassa mobilità"),
    ("F", "#7C3AED", "Massima complessità clinica"),
]

# ════════════════════════════════════════════════════════════════════════════
# CARICAMENTO MODELLI
# ════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="⏳ Caricamento modelli GBM V18…")
def load_models() -> dict:
    # Verifica compatibilità sklearn
    import sklearn
    sk_v = sklearn.__version__
    if sk_v != "1.8.0":
        st.warning(
            f"⚠️ scikit-learn {sk_v} installato, PKL creati con 1.8.0. "
            "Aggiorna requirements.txt → `scikit-learn==1.8.0`"
        )
    m = {}
    model_files = {
        "imputer": "imputer_V18.pkl",
        "D1":      "model_Target_Gravi_5a_V18.pkl",
        "D2":      "model_TIR_target_3a_V18.pkl",
        "D3":      "model_Trans_MMG_6m_V18.pkl",
        "Mort":    "model_Decesso_L_5a_V18.pkl",
        "DCSI":    "model_DCSI_0to2_5a_V18.pkl",
        "dDCSI":   "model_DeltaDCSI_gte2_5a_V18.pkl",
    }
    for key, fname in model_files.items():
        p = MODELS_DIR / fname
        if p.exists():
            with open(p, "rb") as f:
                m[key] = pickle.load(f)
        else:
            m[key] = None
    return m

@st.cache_data
def load_thresholds() -> dict:
    p = DATA_DIR / "thresholds_V18.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}

# ════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════
def drug_score_v(drug: str, dose: float) -> int:
    if dose <= 0:
        return 0
    for thr, sc in DRUGS_VARIABLE[drug]["t"]:
        if dose <= thr:
            return sc
    return 0

def mi(hba1c: float, ds: int) -> float | None:
    if hba1c > 0 and ds > 0:
        return round(hba1c * math.log(1 + ds) / 10, 2)
    return None

def mie(hba1c: float, ds: int, dur: int, tod: int, ascvd: bool) -> float | None:
    if hba1c > 0 and ds > 0:
        av = 2 if ascvd else 0
        return round(
            hba1c * math.log(1 + ds)
            * (1 + 0.35 * math.log(1 + dur))
            * (1 + 0.45 * math.log(1 + tod + av))
            / 10, 2
        )
    return None

def safe_predict(model, X: np.ndarray) -> float | None:
    if model is None:
        return None
    try:
        return float(model.predict_proba(X)[0, 1])
    except Exception:
        return None

def nearest_cluster(d1: float, d2: float, d3: float) -> int:
    pt = np.array([d1, d2, d3])
    return int(np.argmin(np.linalg.norm(KM6_CENTROIDS - pt, axis=1)))

def triage_color(p: float | None, thr: float, invert: bool) -> tuple:
    """(icon, hex_color, text)"""
    if p is None:
        return ("—", "#94A3B8", "n.d.")
    good = (p >= thr) if invert else (p < thr)
    mid  = (p < thr * 1.35) if not invert else (p >= thr * 0.7)
    if good:
        return ("✓", "#059669", "Favorevole")
    elif mid:
        return ("△", "#D97706", "Intermedio")
    else:
        return ("✗", "#DC2626", "Sfavorevole")

# ── Radar Plot ────────────────────────────────────────────────────────────
def radar_chart(pD1, pD2, pD3, pMort, dcsi_n, ds_n) -> go.Figure:
    """Tutti gli assi: 0=centro=ottimale, 1=esterno=peggiore."""
    cfg = [
        ("D1\nStrutturale",    pD1,    False),
        ("D2\nMetabolico",     pD2,    True),   # invert
        ("D3\nMMG",            pD3,    True),   # invert
        ("Mortalità",          pMort,  False),
        ("Burden\nDCSI",       dcsi_n, False),
        ("Complessità\nfarm.", ds_n,   False),
    ]
    labels, vals = [], []
    for lbl, v, inv in cfg:
        labels.append(lbl)
        v_n = 0 if v is None else (1 - v if inv else v)
        vals.append(round(min(max(float(v_n), 0), 1), 3))

    mean_v = np.mean(vals)
    if mean_v < 0.33:
        fill_col, line_col = "rgba(5,150,105,0.22)",   "#059669"
    elif mean_v < 0.58:
        fill_col, line_col = "rgba(217,119,6,0.22)",   "#D97706"
    else:
        fill_col, line_col = "rgba(220,38,38,0.22)",   "#DC2626"

    closed_v = vals + [vals[0]]
    closed_l = labels + [labels[0]]

    fig = go.Figure()
    # Soglia 50%
    fig.add_trace(go.Scatterpolar(
        r=[0.5] * 7, theta=closed_l, mode="lines",
        line=dict(color="#CBD5E1", dash="dot", width=1.5),
        showlegend=False, hoverinfo="skip",
    ))
    # Paziente
    fig.add_trace(go.Scatterpolar(
        r=closed_v, theta=closed_l,
        mode="lines+markers",
        fill="toself", fillcolor=fill_col,
        line=dict(color=line_col, width=2.5),
        marker=dict(size=7, color=line_col, line=dict(color="white", width=1.5)),
        showlegend=False,
        hovertemplate="<b>%{theta}</b><br>%{r:.3f}<extra></extra>",
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, range=[0, 1],
                tickvals=[0.25, 0.5, 0.75, 1.0],
                tickfont=dict(size=8, color="#94A3B8"),
                gridcolor="#E2E8F0",
            ),
            angularaxis=dict(
                tickfont=dict(size=10, family="Inter, system-ui"),
            ),
            bgcolor="white",
        ),
        paper_bgcolor="white",
        margin=dict(t=30, b=30, l=50, r=50),
        height=340,
    )
    return fig

# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════
def main():
    models     = load_models()
    thresholds = load_thresholds()
    models_ok  = models.get("imputer") is not None and models.get("D1") is not None

    # ── Header ────────────────────────────────────────────────────────────
    st.markdown("""
    <div style='background:#0F2D54;padding:13px 18px;border-radius:8px;margin-bottom:14px;
                display:flex;align-items:center;justify-content:space-between'>
        <div>
            <div style='color:white;font-size:19px;font-weight:800;letter-spacing:-0.02em'>
                🩺 Marcellini Index DM2 — Calcolatore V18
            </div>
            <div style='color:#94A3B8;font-size:10px;margin-top:2px'>
                AUSL Romagna Rimini · Solo per uso del team di ricerca · Non uso clinico routinario
            </div>
        </div>
        <div style='color:#64748B;font-size:10px;text-align:right'>
            GBM n=12.160<br>FEATURES 12V
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not models_ok:
        st.warning(
            "⚠️ **Modelli GBM non trovati** nella cartella `models/`.  \n"
            "Copia i file PKL da Google Drive → `Output_V18/` nella cartella `models/` del repository e riavvia l'app.  \n"
            "File necessari: `imputer_V18.pkl`, `model_Target_Gravi_5a_V18.pkl`, "
            "`model_TIR_target_3a_V18.pkl`, `model_Trans_MMG_6m_V18.pkl`, "
            "`model_Decesso_L_5a_V18.pkl`, `model_DCSI_0to2_5a_V18.pkl`, "
            "`model_DeltaDCSI_gte2_5a_V18.pkl`"
        )

    # ── LAYOUT ────────────────────────────────────────────────────────────
    col_in, col_out = st.columns([11, 12], gap="medium")

    # ════════════════════════════════════════════════════════════════════════
    # COLONNA SINISTRA: INPUT
    # ════════════════════════════════════════════════════════════════════════
    with col_in:

        # ── Paziente ──────────────────────────────────────────────────────
        with st.expander("👤 Paziente", expanded=True):
            r1c1, r1c2 = st.columns(2)
            with r1c1:
                eta    = st.number_input("Età (anni)", 18, 100, value=65, step=1)
                sesso  = st.selectbox("Sesso", ["M", "F"])
            with r1c2:
                eta_dm    = st.number_input("Età diagnosi DM", 10, 99, value=55, step=1)
                durata_dm = max(0, int(eta) - int(eta_dm))
                st.metric("Durata DM", f"{durata_dm} anni")

        # ── Laboratorio ───────────────────────────────────────────────────
        with st.expander("🔬 Laboratorio basale", expanded=True):
            r2c1, r2c2 = st.columns(2)
            with r2c1:
                hba1c = st.number_input(
                    "HbA1c (mmol/mol)", 20.0, 180.0, value=63.0, step=0.5,
                    help="Se in %: HbA1c_mmol = (% − 2.15) × 10.929"
                )
            with r2c2:
                egfr = st.number_input(
                    "eGFR (mL/min/1.73m²)", 5.0, 150.0, value=75.0, step=1.0
                )
            hba1c_target_auto = hba1c < 64.0
            hba1c_target = st.checkbox(
                "HbA1c a target ER 2017",
                value=hba1c_target_auto,
                help=(
                    "**Target glicemici individualizzati — Linee di Indirizzo ER 2017**\n\n"
                    "| Target | Condizione clinica |\n"
                    "|---|---|\n"
                    "| **≤48 mmol/mol** (≤6,5%) | Nuova diagnosi o DM <10 anni, "
                    "no precedenti CV, buon compenso abituale, no comorbilità rilevanti |\n"
                    "| **<53 mmol/mol** (<7,0%) | Obiettivo generale per adulti — "
                    "previene complicanze micro- e macrovascolari |\n"
                    "| **≤64 mmol/mol** (≤8,0%) | DM lunga durata (>10 anni), "
                    "precedenti cardiovascolari, storia di inadeguato compenso, "
                    "fragilità per età o comorbilità |\n\n"
                    "Seleziona se il paziente è attualmente **entro il proprio target individualizzato**. "
                    "Il valore auto-suggerito usa la soglia 64 mmol/mol come conservativa."
                ),
            )

        # ── Complicanze DCSI ──────────────────────────────────────────────
        with st.expander("🏥 Complicanze basali — DCSI", expanded=True):
            dcsi_vals  = {}
            dcsi_total = 0
            tod_count  = 0
            ascvd_auto = False

            for dom_id, emoji, label, max_sc, is_tod, is_ascvd in DCSI_DOMAINS:
                st.markdown(f"**{emoji} {label}**")
                descs = DCSI_DESC[dom_id]

                v = st.select_slider(
                    f"Grado {label}",
                    options=list(range(max_sc + 1)),
                    value=0,
                    format_func=lambda x, d=dom_id: (
                        f"{x} — " + DCSI_DESC[d][x].split(" — ")[0]
                    ),
                    key=f"dcsi_{dom_id}",
                    label_visibility="collapsed",
                )

                # Descrizione clinica del grado selezionato
                desc_full = descs.get(v, "")
                if v == 0:
                    badge_col, badge_bg = "#059669", "#F0FDF4"
                elif v == 1:
                    badge_col, badge_bg = "#D97706", "#FFFBEB"
                else:
                    badge_col, badge_bg = "#DC2626", "#FEF2F2"

                st.markdown(
                    f"<div style='background:{badge_bg};border-left:3px solid {badge_col};"
                    f"padding:4px 10px;border-radius:0 4px 4px 0;font-size:11px;"
                    f"color:#374151;margin-bottom:8px'>"
                    f"<b>Grado {v}:</b> {desc_full}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                dcsi_vals[dom_id] = v
                dcsi_total += v
                if v > 0 and is_tod:
                    tod_count += 1
                if v > 0 and is_ascvd:
                    ascvd_auto = True

            ascvd_flag = st.checkbox(
                "ASCVD basale confermata",
                value=ascvd_auto,
                help="Storia di evento CV/cerebrovascolare/arteriopatia. Auto-calcolata dai domini sopra."
            )
            tod_er_max = min(tod_count, 3)

            # Badge riepilogo
            bg_c = "#F0FDF4" if dcsi_total == 0 else "#FFF7ED" if dcsi_total <= 3 else "#FEF2F2"
            st.markdown(
                f"<div style='background:{bg_c};border-radius:5px;padding:7px 11px;"
                f"font-size:12px;margin-top:4px'>"
                f"<b>DCSI basale = {dcsi_total}</b> &nbsp;|&nbsp; "
                f"TOD = {tod_er_max} &nbsp;|&nbsp; "
                f"ASCVD = {'✓' if ascvd_flag else '✗'}"
                f"</div>",
                unsafe_allow_html=True,
            )

        # ── Farmaci dose-dipendenti ────────────────────────────────────────
        with st.expander("💊 Farmaci dose-dipendenti", expanded=False):
            vd_scores = {}
            for drug, cfg in DRUGS_VARIABLE.items():
                c1d, c2d = st.columns([3, 2])
                with c1d:
                    dose = st.number_input(
                        f"{drug} ({cfg['unit']})",
                        min_value=0.0, value=0.0, step=0.5,
                        key=f"vd_{drug}",
                    )
                sc = drug_score_v(drug, dose) if dose > 0 else 0
                vd_scores[drug] = sc
                with c2d:
                    if dose > 0:
                        st.markdown(
                            f"<div style='margin-top:26px;font-weight:700;color:#1A6DB5'>"
                            f"→ {sc} pt</div>",
                            unsafe_allow_html=True,
                        )

        # ── Farmaci punteggio fisso ────────────────────────────────────────
        with st.expander("💊 Farmaci punteggio fisso", expanded=False):
            fd_active  = {}
            fd_scores  = {}
            for cat_name, cat in DRUGS_FIXED_CATS.items():
                st.markdown(f"**{cat_name}**")
                cs = st.columns(2)
                for idx, (drug, score) in enumerate(zip(cat["items"], cat["scores"])):
                    with cs[idx % 2]:
                        active = st.checkbox(f"{drug} (+{score})", key=f"fd_{drug}")
                        fd_active[drug] = active
                        if active:
                            fd_scores[drug] = score

        # ── DrugScore totale + flag derivati ──────────────────────────────
        ds_total     = sum(vd_scores.values()) + sum(fd_scores.values())
        ha_ins_b     = any(fd_active.get(d) for d in INS_BASALE)
        ha_ins_r     = any(fd_active.get(d) for d in INS_RAPIDA)
        ha_sulfo     = any(fd_active.get(d) for d in SULFONILUR)
        ha_repag     = fd_active.get("Repaglinide", False)
        ins_multi    = ha_ins_b and ha_ins_r
        rischio_ipo  = ha_ins_b or ha_ins_r or ha_sulfo or ha_repag

        flags_html = ""
        if rischio_ipo:
            flags_html += "<span style='background:#FEF3C7;color:#92400E;padding:2px 7px;" \
                          "border-radius:4px;font-size:11px;font-weight:700;margin-left:6px'>⚠ Rischio ipo</span>"
        if ins_multi:
            flags_html += "<span style='background:#FEE2E2;color:#991B1B;padding:2px 7px;" \
                          "border-radius:4px;font-size:11px;font-weight:700;margin-left:6px'>💉 Basal-bolus</span>"

        st.markdown(
            f"<div style='background:#EFF6FF;border:1.5px solid #93C5FD;border-radius:6px;"
            f"padding:8px 13px;margin-top:4px'>"
            f"<b style='color:#1A6DB5;font-size:15px'>DrugScore = {ds_total}</b>"
            f"{flags_html}</div>",
            unsafe_allow_html=True,
        )

    # ════════════════════════════════════════════════════════════════════════
    # FEATURE VECTOR
    # ════════════════════════════════════════════════════════════════════════
    features = {
        "EtaBasale":                           float(eta),
        "Sesso":                               1.0 if sesso == "M" else 0.0,
        "EtaDiabeteMellito":                   float(eta_dm),
        "HbA1cBasale":                         float(hba1c),
        "HbA1c_ER_target_basale":              1.0 if hba1c_target else 0.0,
        "eGFR_basale":                         float(egfr),
        "ASCVD_basale":                        1.0 if ascvd_flag else 0.0,
        "TOD_ER_max_basale":                   float(tod_er_max),
        "DCSI_basale":                         float(dcsi_total),
        "DrugScore_basale":                    float(ds_total),
        "TerapiaRischioIpo":                   1.0 if rischio_ipo else 0.0,
        "Insulina_multiniettiva_basale_trans":  1.0 if ins_multi else 0.0,
    }
    X_raw = np.array([[features[f] for f in FEATURES_BASE]])

    # ── Predizioni GBM ────────────────────────────────────────────────────
    preds: dict[str, float | None] = {k: None for k in ["D1","D2","D3","Mort","DCSI","dDCSI"]}

    if models_ok:
        try:
            import warnings as _w
            with _w.catch_warnings():
                _w.filterwarnings("ignore", message="Skipping features without any observed values")
                X_imp = models["imputer"].transform(X_raw)
            # Ripristina Sesso (feature 1) dopo imputer: il pkl ha median=NaN per Sesso
            # perché in Colab era ancora stringa durante il fit — lo sovrascriviamo
            X_imp[0, 1] = 1.0 if sesso == "M" else 0.0
            for key in ["D1","D2","D3","Mort","DCSI","dDCSI"]:
                preds[key] = safe_predict(models[key], X_imp)
        except Exception as e:
            st.error(f"Errore predizione: {e}")

    # ── MI / MIE ──────────────────────────────────────────────────────────
    mi_val  = mi(hba1c, ds_total)
    mie_val = mie(hba1c, ds_total, durata_dm, tod_er_max, ascvd_flag)

    # ── Cluster k=6 (nearest centroid in D1/D2/D3) ────────────────────────
    cluster_id = None
    if preds["D1"] and preds["D2"] and preds["D3"]:
        cluster_id = nearest_cluster(preds["D1"], preds["D2"], preds["D3"])

    # ════════════════════════════════════════════════════════════════════════
    # COLONNA DESTRA: OUTPUT
    # ════════════════════════════════════════════════════════════════════════
    with col_out:

        # ── Triage (regola 3 criteri) ──────────────────────────────────────
        triage_pos = not ascvd_flag and tod_count == 0 and not ins_multi
        bg_t = "#F0FDF4" if triage_pos else "#FEF2F2"
        br_t = "#86EFAC" if triage_pos else "#FCA5A5"
        t_icon = "✅" if triage_pos else "🏥"
        t_title = "PERCORSO MMG — Regola 3 criteri (NPV 94.8%)" if triage_pos \
                  else "GESTIONE SPECIALISTICA"
        t_body_parts = []
        if not triage_pos:
            if ascvd_flag:  t_body_parts.append("ASCVD")
            if tod_count:   t_body_parts.append(f"TOD ({tod_count})")
            if ins_multi:   t_body_parts.append("Schema basal-bolus")
        t_body = " · ".join(t_body_parts) if t_body_parts else "No ASCVD · No TOD · No basal-bolus"

        st.markdown(
            f"<div style='background:{bg_t};border:2px solid {br_t};border-radius:8px;"
            f"padding:10px 15px;margin-bottom:10px'>"
            f"<b>{t_icon} {t_title}</b><br>"
            f"<span style='font-size:11px;color:#64748B'>{t_body}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # ── Radar ─────────────────────────────────────────────────────────
        dcsi_n = min(dcsi_total / 13.0, 1.0)
        ds_n   = min(ds_total / 55.0, 1.0)
        fig    = radar_chart(preds["D1"], preds["D2"], preds["D3"],
                              preds["Mort"], dcsi_n, ds_n)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        st.caption("Centro = ottimale · Esterno = peggiore · Linea tratteggiata = soglia 50% · "
                   "D2 e D3 invertiti (alto = buono)")

        # ── Score GBM + soglie ─────────────────────────────────────────────
        st.markdown("#### Score dimensionali GBM")

        if not models_ok:
            st.info("ℹ️ Predizioni non disponibili — modelli non caricati.")
        else:
            SCORE_ORDER = ["D1","Mort","DCSI","dDCSI","D2","D3"]
            DIR_ICON = {"bad": "📈", "good": "📉"}

            for key in SCORE_ORDER:
                p    = preds.get(key)
                cfg  = THRESHOLDS.get(key, {})
                inv  = cfg.get("invert", False)
                sog  = cfg.get("soglia", 0.5)
                auc  = cfg.get("auc", 0.0)
                ci_lo= cfg.get("ci_lo", 0.0)
                ci_hi= cfg.get("ci_hi", 0.0)
                lbl  = cfg.get("label", key)
                evnt = cfg.get("event_label", "evento")
                time = cfg.get("time", "—")
                dirn = cfg.get("direction", "bad")
                icon, col, text = triage_color(p, sog, inv)

                # Probabilità in percentuale
                pct = None if p is None else p * 100

                # Colore sfondo card
                if p is None:
                    card_bg, card_br = "#F8FAFC", "#CBD5E1"
                elif (dirn == "bad" and p >= sog) or (dirn == "good" and p < sog):
                    card_bg, card_br = "#FEF2F2", "#FCA5A5"
                elif (dirn == "bad" and p >= sog * 0.75) or (dirn == "good" and p >= sog * 0.75):
                    card_bg, card_br = "#FFFBEB", "#FDE68A"
                else:
                    card_bg, card_br = "#F0FDF4", "#86EFAC"

                # Frase probabilità
                if p is None:
                    prob_sentence = "Predizione non disponibile"
                elif dirn == "bad":
                    prob_sentence = (
                        f"Probabilità di <b>{evnt.split(chr(10))[0]}</b> "
                        f"a <b>{time}</b>: <span style='font-size:20px;font-weight:900;"
                        f"color:{col}'>{pct:.1f}%</span>"
                    )
                else:
                    prob_sentence = (
                        f"Probabilità di <b>{evnt.split(chr(10))[0]}</b> "
                        f"entro <b>{time}</b>: <span style='font-size:20px;font-weight:900;"
                        f"color:{col}'>{pct:.1f}%</span>"
                    )

                # Nota evento (seconda riga dell'event_label)
                evnt_lines = evnt.split("\n")
                evnt_note  = evnt_lines[1] if len(evnt_lines) > 1 else ""

                st.markdown(
                    f"<div style='background:{card_bg};border:1.5px solid {card_br};"
                    f"border-radius:8px;padding:10px 14px;margin-bottom:8px'>"

                    # Titolo riga 1
                    f"<div style='display:flex;justify-content:space-between;align-items:center'>"
                    f"<b style='color:#0F2D54;font-size:13px'>{lbl}</b>"
                    f"<span style='font-weight:700;font-size:13px;color:{col}'>"
                    f"{icon} {text}</span>"
                    f"</div>"

                    # Probabilità
                    f"<div style='margin:5px 0 2px'>{prob_sentence}</div>"

                    # Nota evento (solo se c'è)
                    + (f"<div style='font-size:10px;color:#64748B;margin-bottom:3px'>"
                       f"<i>{evnt_note}</i></div>" if evnt_note else "")

                    # Footer: AUC, CI, soglia
                    + f"<div style='display:flex;gap:12px;margin-top:5px;flex-wrap:wrap'>"
                    f"<span style='font-size:10px;color:#94A3B8'>"
                    f"AUC {auc:.3f} [IC95%: {ci_lo:.3f}–{ci_hi:.3f}]</span>"
                    f"<span style='font-size:10px;color:#94A3B8'>"
                    f"Soglia Youden: {sog} "
                    f"(Sens {cfg.get('sens',0):.1f}% · Spec {cfg.get('spec',0):.1f}% · "
                    f"NPV {cfg.get('npv',0):.1f}%)</span>"
                    f"</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        # ── Cluster k=6 ────────────────────────────────────────────────────
        if cluster_id is not None:
            lbl, col_c, name = KM6_NAMES[cluster_id]
            st.markdown(
                f"<div style='background:{col_c}18;border:2px solid {col_c}60;"
                f"border-radius:8px;padding:9px 14px;margin-bottom:10px'>"
                f"<b>Cluster k=6 — {lbl} · {name}</b><br>"
                f"<span style='font-size:10px;color:#64748B'>"
                f"Assegnazione per prossimità centroide in spazio D1/D2/D3</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

        # ── Marcellini Index ────────────────────────────────────────────────
        st.markdown("#### Marcellini Index")
        m1, m2 = st.columns(2)
        with m1:
            st.metric(
                "MI classico",
                f"{mi_val:.2f}" if mi_val else "—",
                help="HbA1c × ln(1 + DrugScore) / 10",
            )
        with m2:
            st.metric(
                "MI Evoluto",
                f"{mie_val:.2f}" if mie_val else "—",
                help="MI × (1 + 0.35·ln(1+DurataDM)) × (1 + 0.45·ln(1+TOD+ASCVD_pesata))",
            )

        # ── Feature vector (debug expander) ────────────────────────────────
        with st.expander("🔍 Vettore di predizione (12 variabili basali)"):
            df_feat = pd.DataFrame({
                "Feature": FEATURES_BASE,
                "Valore":  [features[f] for f in FEATURES_BASE],
            })
            st.dataframe(df_feat, width="stretch", hide_index=True, height=320)
            st.caption("Questi sono i valori esatti passati al GBM dopo imputazione mediana.")


if __name__ == "__main__":
    main()
