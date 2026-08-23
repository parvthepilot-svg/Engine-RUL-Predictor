**Interactive Streamlit Dashboard**

A streamlined single-page dashboard designed for real-time engine telemetry and condition monitoring:

* **Engine Triage & Lifecycle Slider:** Interactively cycle through individual turbofan units and cycle steps.
* **Health & Sensor Telemetry:** Real-time sensor trend plotting with color-coded operational status (Green/Yellow/Red).
* **RUL Decay & Model Bias:** Visualizes predicted vs. actual Remaining Useful Life and tracks directional prediction bias.

**Safety-Critical Evaluation**

Standard loss metrics like RMSE treat overpredicting engine life (catastrophic failure risk) and underpredicting life (early maintenance) as equal errors. This pipeline evaluates predictions using NASA's Asymmetric Scoring Function to penalize late-failure predictions far more severely than conservative early alerts:

$$d = y_{\text{pred}} - y_{\text{true}}$$

$$\text{Penalty} = \begin{cases} e^{-d/13} - 1 & \text{if } d < 0 \text{ (early / safe prediction)} \\ e^{d/10} - 1 & \text{if } d \ge 0 \text{ (late / dangerous prediction)} \end{cases}$$
