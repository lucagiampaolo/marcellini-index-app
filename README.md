# Marcellini Index DM2 — Calcolatore Clinico V18

Applicazione Streamlit per il calcolo bedside degli score predittivi DM2.  
**Uso riservato al team di ricerca · AUSL Romagna Rimini**

---

## Deploy su Streamlit Community Cloud

### 1. Prerequisiti
- Account GitHub (free)
- Account Streamlit Cloud (free): https://streamlit.io/cloud

### 2. Primo setup (una volta sola)

```bash
# Clona il repo privato
git clone https://github.com/<tuo-org>/mi_dm2_app.git
cd mi_dm2_app

# Copia i modelli da Google Drive → Output_V18/
# nella cartella models/ (NON vengono committati su git)
cp /path/to/Drive/Output_V18/*.pkl models/
```

### 3. Deploy su Streamlit Cloud

1. Vai su https://share.streamlit.io → "New app"
2. Seleziona il tuo repository privato
3. Branch: `main` · File: `app.py`
4. Clicca **Deploy**

### 4. Imposta la password

In Streamlit Cloud → Settings → **Secrets**:

```toml
password = "la_tua_password_sicura"
```

### 5. Aggiorna i modelli

Se esegui nuovamente il notebook V18 e generi nuovi PKL:

```bash
# Sostituisci i PKL nella cartella models/
cp /path/to/new_models/*.pkl models/

# Commit e push (solo i file non in .gitignore)
git add data/
git commit -m "aggiorna soglie/dati"
git push
# Streamlit Cloud si riavvia automaticamente
```

---

## Struttura

```
mi_dm2_app/
├── app.py                  ← App principale
├── requirements.txt
├── .streamlit/
│   ├── config.toml         ← Tema e configurazione
│   └── secrets.toml        ← PASSWORD (non su git)
├── models/                 ← PKL files (non su git)
│   ├── imputer_V18.pkl
│   ├── model_Target_Gravi_5a_V18.pkl
│   ├── model_TIR_target_3a_V18.pkl
│   ├── model_Trans_MMG_6m_V18.pkl
│   ├── model_Decesso_L_5a_V18.pkl
│   ├── model_DCSI_0to2_5a_V18.pkl
│   └── model_DeltaDCSI_gte2_5a_V18.pkl
└── data/
    ├── thresholds_V18.json ← Soglie Youden reali da N6
    └── features_V18.json   ← Lista 12 variabili
```

---

## Score calcolati

| Score | Outcome | AUC Globale | Soglia Youden |
|---|---|---|---|
| p(D1) | Target_Gravi_5a | 0.669 | 0.402 |
| p(D2) | TIR_target_3a | 0.720 | 0.423 |
| p(D3) | Trans_MMG_6m | 0.892 | 0.211 |
| p(Mort) | Decesso_L_5a | 0.829 | 0.111 |
| p(DCSI≥2) | DCSI_0to2_5a | 0.745 | 0.361 |
| p(ΔDCSI≥2) | DeltaDCSI_gte2_5a | 0.725 | 0.265 |

**Modello:** GBM (n=12.160, 5-fold OOF) · 12 variabili basali  
**Validazione temporale:** AUC media 0.764 (Excl→VAL2019)

---

## Note metodologiche

- **Cluster k=6**: assegnato per prossimità centroide in spazio D1/D2/D3
- **Regola triage 3 criteri**: NPV 94.8% (no ASCVD + no TOD + no basal-bolus)
- I modelli sono addestrati su dati AUSL Romagna Rimini — non trasferibili senza ricalibrazione
