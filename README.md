# Jet Engine Remaining Useful Life (RUL) Predictor

An industrial-grade condition-based maintenance pipeline and interactive Streamlit application designed to forecast the Remaining Useful Life (RUL) of aircraft turbofan engines. Built on NASA's C-MAPSS dataset (FD001), this project transitions machine learning from standard error minimization into a safety-critical maintenance tool. By re-engineering the model's hyperparameter search around NASA's asymmetric scoring penalty, the system prioritizes human safety by penalizing dangerous late-failure predictions far more severely than conservative early alerts.

**Key Technical Highlights**
* **Data Leakage Prevention:** Implements 5-fold `GroupKFold` cross-validation anchored on engine `unit_number`, guaranteeing that data from a single engine never spans both training and validation folds.
* **Signal Denoising:** Feature engineering pipeline extracts 5-cycle rolling averages (`_roll5`) across 14 active sensor channels ($T24$, $T30$, $Ps30$, $Nc$, etc.) to smooth out physical sensor noise.
* **Capped Degradation Target:** RUL target labels are clipped at a 125-cycle upper limit to accurately model early-life healthy operational plateaus.

**Safety-Optimized Model Tuning**

Standard loss metrics like RMSE treat overpredicting remaining engine life (catastrophic failure risk) and underpredicting life (early maintenance) as equal errors. This pipeline customizes `RandomizedSearchCV` across 125 XGBoost model configurations using NASA's Asymmetric Scoring Function as the explicit objective:

$$d = y_{\text{pred}} - y_{\text{true}}$$

$$\text{Penalty} = \begin{cases} e^{-d/13} - 1 & \text{if } d < 0 \quad \text{(early / safe prediction)} \\ e^{d/10} - 1 & \text{if } d \ge 0 \quad \text{(late / dangerous prediction)} \end{cases}$$

**Interactive Streamlit Dashboard**

| View | Primary Function | Key Capabilities |
| :--- | :--- | :--- |
| **Fleet Overview** | Fleet Triage | Simulates fleet-wide degradation state at selected lifecycle points (Green/Yellow/Red status). |
| **Engine Inspector** | Deep-Dive Diagnostics | Cycle-by-cycle slider, predicted vs. actual RUL decay curves, and real sensor telemetry plots. |
| **Model Trust** | Live Model Validation | Computes live RMSE, MAE, $R^2$, and directional bias metrics dynamically upon launch. |

**Quickstart**

1. Clone the repository:
   ```bash
   git clone https://github.com/parvthepilot-svg/Engine-RUL-Predictor.git
   cd Engine-RUL-Predictor
