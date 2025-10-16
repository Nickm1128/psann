DO NOT RUN ANYTHING COMPUTATIONALLY INTENSE LOCALLY! WE WILL RUN THESE EXPERIMENTS IN COLAB. OUR LOCAL MACHINE IS A CHEAP LAPTOP!

**what we're doing here**
- We have created a python package, psann, which has an sklearn-style, accessible API for developers. I want to produce a research paper on the properties and advantages of using the PSANN architecture based on a performance analysis of psann models and alternatives in different cases with different datasets. Your goal is to create a notebook that will run in Google colab for GPU acceleration that will collect all of the data we will need to assemble our findings in the paper. Use the psann package instead of the local files since we will have to install via pip install in colab. 

**dataset qualifiers**
- I was unable to collect all of the requested datasets in the plan. In most cases, I collected an alternative datasets. Please review the shape of the data that I collected and adapt the plan to use the datasets available. 

**plan**
Working Title

Parameterized Sine‑Activated Networks under Compute Parity: Robust Forecasting on Nonstationary and High‑Dimensional Data

1. Hypothesis

H1 (Core): Under strict compute parity, Residual PSANN (ResPSANN) achieves equal or better generalization than MLP/CNN baselines on nonstationary and high‑dimensional forecasting tasks, and narrows the gap to sequence‑native models (LSTM/TCN) when equipped with a lightweight, fair temporal spine.

H2 (Information Usage): PSANN leverages more of the available input information than ReLU MLPs at the same compute, measurable by permutation score drop (PSD) and complementary attribution metrics.

H3 (Spectral Geometry): PSANN exhibits favorable spectral/geometry profiles (Jacobian/NTK spectra and participation ratio) consistent with stable training and emergent complexity capture under residual wiring.

H4 (Robustness): PSANN maintains performance under distribution shift (regime changes, heavy‑tailed noise, missingness) at least as well as MLPs, and sometimes better, when matched for compute.

H5 (Limits): Without explicit temporal inductive bias, PSANN trails RNN/TCN on long‑memory tasks; this gap can be reduced with a tiny temporal preencoder that preserves compute parity.

2. Contributions

Compute‑parity evaluation protocol for time‑series models (wall‑clock cap + param/flop budget calibration).

Residual PSANN variants with optional temporal spines (strided conv or single‑head attention) that preserve fairness while exposing order information.

Information‑usage diagnostics: PSD (exogenous vs. endogenous channels), SHAP/KL‑style ablations, and mutual‑information proxies.

Spectral geometry analysis: Jacobian/NTK spectra and participation ratio across training—linking inductive bias to stability.

Comprehensive empirical study over synthetic families and real datasets spanning nonstationarity, dimension, noise, and output type.

3. Background & Related Work (concise)

Periodic activations (e.g., SIREN) and Fourier features; their expressivity and frequency bias.

Residual connections for stabilizing gradient flow through highly oscillatory activations.

Classical time‑series architectures: MLP, 1D‑CNN/TCN, LSTM/GRU, small Transformers.

Information‑usage and attribution metrics for multivariate forecasting.

Geometry‑based diagnostics (NTK/PR) as lenses on trainability and capacity.

4. The PSANN Family (Method)

PSANN block: parameterized sine activation; weight init and scaling.

Residual PSANN (ResPSANN): skip connections to stabilize phase/amplitude search.

Temporal spines (optional):

Conv‑spine: few strided Conv1d layers (no global pooling) → compact but information‑rich embedding.

Attention‑spine (tiny): single‑head temporal self‑attention over the window; cost‑controlled to match compute.

Training regime: standard regression/classification losses; early stopping or budget‑limited training; learning‑rate warmup (optional, budget permitting).

Fairness constraints:

Wall‑clock budget per model (seconds).

Parameter (and rough flop) budget matching across baselines.

Throughput calibration to set epochs/steps so each model consumes similar compute.

5. Experimental Protocol
5.1 Baselines

MLP (ReLU) with matched params.

1D‑CNN / TCN (light).

LSTM/GRU (1–2 layers, hidden sizes matched).

Small Transformer‑lite (very shallow, single head) for fairness.

Ablations: PSANN‑plain vs. Res; sine vs. ReLU/Tanh; with/without temporal spine; WaveResNet variant (if available).

5.2 Metrics

Forecasting: RMSE/MAE, R²; sMAPE and MASE for comparability with time‑series literature.

By‑regime breakdown (for synthetic/real regime shifts).

Information usage: PSD (permute exogenous only, then past‑y only), SHAP summary, MI proxy (kNN‑MI or InfoNCE‑style).

Robustness: performance vs. noise level; missingness masks; shift to held‑out regimes.

Resource: wall‑time, params, peak memory.

5.3 Train/Validation/Test Splits

Temporal splits (no leakage): 60/20/20 or rolling‑window for real series.

Multi‑seed (≥5) for CIs; paired tests (e.g., Wilcoxon) for significance under same data/resamples.

6. Datasets
6.1 Synthetic families (controlled stress tests)

Nonstationary sinusoid mixtures with regime changes (frequency & amplitude drift), with/without exogenous drivers.

Heavy‑tailed/mixture noise: Student‑t, Laplace, and Gaussian‑mixture noise regimes.

Chaotic dynamics: Lorenz or Mackey‑Glass (short/long memory variants).

Discrete targets: regime classification from multivariate signals.

Dimensionality sweeps: input dims {4, 16, 32, 128}; output dims {1, 8, 32} (multi‑horizon).

Irregular sampling: synthetic time‑jitter + masking indicators.

6.2 Real‑world (typical Kaggle‑available or mirrored)

Retail demand: Rossmann Store Sales, M5 Forecasting – Accuracy (multivariate exogenous; long horizon).

Traffic/Views: Web Traffic Time Series Forecasting (Wikipedia pageviews).

Energy/Climate: Jena Climate 2009–2016; Household Power Consumption; Appliances Energy Prediction.

Air quality: Beijing Multi‑Site Air‑Quality, Air Pollution in Seoul.

Sensors/HAR: Human Activity Recognition with Smartphones (HAR) (classification; discrete labels).

Finance (optional): Cryptocurrency Historical Prices (BTC/ETH/others); S&P 500 stocks (multivariate).

IoT/Industry: compressor/temperature anomaly sets; transformer temperature (ETT) if accessible.

Each dataset selected to vary stationarity, dimensionality, noise/tailedness, regularity of sampling, and target type (continuous vs discrete).

7. Experiments
7.1 Nonstationary Univariate & Low‑Dim Multivariate (Continuous Targets)

Goal: Base case; demonstrate PSANN vs MLP under drift.

Compare: MLP, TCN, LSTM, PSANN‑plain, ResPSANN, ResPSANN+spine.

Measurements: R², sMAPE by regime; training stability (grad norms).

7.2 High‑Dim Multivariate Forecasting (Continuous)

Goal: Stress “uses more information” hypothesis.

Design: 32–128 exogenous features with correlated structure; multi‑step outputs (e.g., 24‑step ahead).

Compare: MLP, TCN, LSTM, ResPSANN, ResPSANN+spine.

Diagnostics: PSD(exogenous) vs PSD(past‑y), SHAP, MI proxy.

7.3 Heavy‑Tails & Outliers

Goal: Test robustness to nonstandard noise.

Design: Add Student‑t/Laplace noise and sparse outliers at random times.

Compare: MLP, LSTM, ResPSANN (with/without spine).

Metrics: Robust RMSE (median absolute error), sMAPE; breakdown by noise regime.

7.4 Irregular & Event‑Driven Sequences

Goal: Continuous vs discrete/irregular timing.

Design: Timestamp deltas as inputs; event counts (Poisson‑like) or binary events.

Compare: Small Transformer‑lite, LSTM, ResPSANN+attention‑spine vs ResPSANN (flat).

Metrics: NLL for counts; AUC/F1 for events.

7.5 Classification (HAR / sensor)

Goal: Test discrete targets and representation learning.

Compare: 1D‑CNN/TCN, LSTM, ResPSANN(+spine).

Metrics: Accuracy, F1‑macro; calibration (ECE).

7.6 Robustness to Shift & Missingness

Goal: Stress test generalization.

Design: Hold‑out regimes; randomly mask portions of exogenous channels.

Metrics: ΔR² / ΔMASE vs % missing; recovery with simple imputation.

8. Ablations (What matters inside PSANN?)

Residual vs plain (expected: residual consistently better).

Activation: sine vs ReLU/Tanh/GELU; frequency parameterizations (init scales).

Temporal spine choice: strided‑conv vs tiny attention; pooling strategy (no global avg when we need phase/ordering).

Head width/depth under same time budget: (layers {1–3}, units {32–128}) matched to embed size.

Learning‑rate schedule: warmup vs fixed LR under seconds cap.

Regularization: noise injection in input vs weight decay; dropout in spines.

9. Information Usage & Interpretability

PSD by channel group: exogenous vs past‑y; plot mean RMSE increase.

SHAP summary for flat vs spine variants to visualize feature reliance.

MI proxy between inputs and predictions (kNN‑MI or InfoNCE‑style), normalized by baseline.

CKA similarity between PSANN activations and engineered features (spectral bands) to reveal alignment.

10. Geometry & Spectral Diagnostics

Jacobian singular spectrum across epochs (curves for ResPSANN vs MLP).

NTK eigenvalue decay (power‑law fits; participation ratio).

Frequency response probe: inject sinusoidal test signals of varying frequency into inputs and measure gain—“poor‑man’s Bode plot” showing how PSANN vs MLP treat different bands.

11. Results & Analysis

Compute‑parity tables: wall‑time, params, R²/sMAPE/MASE ± CI across seeds.

By‑regime plots (nonstationarity): error vs time; change‑point sensitivity.

PSD/Shap bar charts: information uptake; spine vs flat.

Geometry panels: Jacobian/NTK/PR comparisons (with effect sizes).

Significance tests: paired non‑parametric tests over seeds; report p‑values and effect sizes.

12. Discussion

When PSANN excels: nonstationary, high‑dimensional forecasting with moderate horizons; residual wiring + non‑squashing temporal spines.

When PSANN lags: very long memory without temporal inductive bias; overly aggressive pooling spines.

Why: periodic activations + residuals favor capturing multi‑scale oscillations; strong exogenous signals unlock PSANN’s information appetite.

Practical guidance: spine designs, embedding sizes, and budget tuning.

13. Threats to Validity

Dataset selection bias; tuning effort imbalance.

Wall‑clock parity vs. hardware variability; report repeats, seeds, and CI.

Leakage risks in temporal splits; document preprocessing.

14. Conclusion

Summary of validated hypotheses (where supported), nuanced limitations (where falsified).

Takeaway: With a fair temporal scaffold and compute parity, ResPSANN is a practical, interpretable contender for nonstationary, multivariate forecasting; it scales with dimensionality and offers meaningful diagnostic handles.

15. Reproducibility & Artifacts

Public code (pin to repo commit hash), notebooks, and single‑file result bundles per experiment.

Environment snapshot (CUDA/PyTorch), seeds, and exact data splits.

Clear instructions for Kaggle downloads (data loaders with checksums).

Execution Plan (short)

Finalize fairness protocol (seconds cap + param count targets); lock baselines.

Re‑run synthetic suite to populate all axes (stationarity, noise, dims, outputs).

Run real datasets with temporal splits; collect metrics + PSD/SHAP; save bundles.

Run geometry diagnostics on trained checkpoints (small batches to keep runtime sane).

Statistical aggregation over seeds; produce final tables/figures.

Write: Intro → Methods → Results (with figures) → Discussion → Limitations → Appendix.

**info table**
🧩 PSANN Research Experiment Tracker														
Section	Dataset	Data Type	Target Type	Input Dim	Output Dim	Stationarity	Noise/Distribution	Real/Synthetic	Model Variants	Temporal Spine	Compute Budget (s)	Metrics	Diagnostics	Notes
Synthetic – Core	Nonstationary Sine Mix (3-regime)	Continuous	Regression	6	1	Nonstationary	Gaussian	Synthetic	MLP, LSTM, ResPSANN, ResPSANN+ConvSpine	None / Strided Conv	25	R², RMSE, MAE	PSD (exo vs past-y), by-regime	Core baseline we already ran
Synthetic – High Dim	Nonstationary High-Dim (20–128 inputs)	Continuous	Regression	32 / 64 / 128	1	Nonstationary	Gaussian	Synthetic	MLP, LSTM, ResPSANN, ResPSANN+ConvSpine	Strided Conv	25	R², RMSE, MAE	PSD, SHAP, MI proxy	Stress test for dimensionality scaling
Synthetic – Heavy Tails	Sine Mix + Student-t noise	Continuous	Regression	8	1	Nonstationary	Student-t(ν=3)	Synthetic	MLP, LSTM, ResPSANN, ResPSANN+ConvSpine	Strided Conv	25	R², Robust RMSE, MAE	PSD, Outlier Robustness	Tests resilience to heavy-tailed noise
Synthetic – Long Memory	Mackey-Glass / Lorenz System	Continuous	Regression	4	1	Nonstationary	Deterministic Chaos	Synthetic	MLP, LSTM, TCN, ResPSANN, ResPSANN+AttnSpine	Attention	25	R², MASE	PSD, Frequency Response	Long temporal dependencies
Synthetic – Discrete Targets	Regime Classifier (from sine mix)	Discrete	Classification	10	3	Nonstationary	Gaussian	Synthetic	MLP, LSTM, ResPSANN, ResPSANN+ConvSpine	Strided Conv	25	Accuracy, F1	Confusion, ECE	Multi-class classification
Synthetic – Missingness	Nonstationary Sine (masked 20%)	Continuous	Regression	8	1	Nonstationary	Gaussian + missing	Synthetic	MLP, LSTM, ResPSANN, ResPSANN+ConvSpine	Strided Conv	25	R², MASE	ΔR² vs %missing	Test robustness to data gaps
Real – Retail	Kaggle Rossmann Store Sales	Tabular Time Series	Regression	~10	1	Nonstationary	Heavy-tailed	Real	XGBoost, MLP, LSTM, ResPSANN, ResPSANN+ConvSpine	Strided Conv	30	RMSE, sMAPE	PSD, SHAP	Business forecast (structured tabular)
Real – Energy	Jena Climate 2009–2016	Sensor Series	Regression	14	1	Nonstationary	Gaussian	Real	MLP, LSTM, ResPSANN, ResPSANN+ConvSpine	Strided Conv	25	R², RMSE	PSD, SHAP, NTK Spectrum	Classic climate benchmark
Real – Finance	Crypto Prices (BTC, ETH, SOL, etc.)	Financial	Regression	10	1	Nonstationary	Heavy-tailed	Real	MLP, LSTM, ResPSANN, ResPSANN+ConvSpine	Strided Conv	25	R², Sharpe proxy	PSD, Rolling R²	Regime volatility robustness
Real – Traffic / Web	Wikipedia Traffic Forecasting	Time Series	Regression	20	1	Nonstationary	Poisson	Real	MLP, TCN, ResPSANN, ResPSANN+ConvSpine	Strided Conv	30	sMAPE, RMSE	PSD	High-dim, long-range daily cycles
Real – Sensor / HAR	Human Activity Recognition (UCI HAR)	Sensor	Classification	9	6	Stationary	Gaussian	Real	1D CNN, LSTM, ResPSANN+ConvSpine	Strided Conv	25	Accuracy, F1	Confusion, PSD	Discrete classification
Real – IoT / ETT	Electricity Transformer Temperature	Sensor	Regression	7	1	Nonstationary	Gaussian	Real	MLP, TCN, ResPSANN, ResPSANN+ConvSpine	Strided Conv	25	RMSE, sMAPE	PSD, NTK Spectrum	Medium-term horizon prediction
Real – Air Quality	Beijing Air Quality (Kaggle)	Sensor	Regression	15	1	Nonstationary	Gaussian	Real	MLP, LSTM, ResPSANN	Strided Conv	25	RMSE, R²	PSD, SHAP	Exogenous features (weather + pollution)
Synthetic – Geometry Probe	Random Feature Maps	Continuous	Regression	32	1	Stationary	Gaussian	Synthetic	ResPSANN, MLP	None	15	None	Jacobian/NTK/PR	For geometry analysis only
Real – Anomaly Detection	NASA Turbofan Degradation (CMAPSS)	Sensor	Regression (RUL)	24	1	Nonstationary	Non-Gaussian	Real	MLP, LSTM, ResPSANN+ConvSpine	Strided Conv	30	RMSE, R²	PSD, PR	Multivariate degradation curves
⚙️ Column definitions														
														
Model Variants: always include at least MLP (ReLU baseline), LSTM (sequence), ResPSANN (flat), and ResPSANN + Spine (temporal).														
														
Temporal Spine:														
														
None: direct flattening.														
														
Strided Conv: 2–3 Conv1d layers with downsample stride.														
														
Attention: 1-head self-attention on window (cheap transformer-lite).														
														
Diagnostics:														
														
PSD (Permutation Score Drop) — exogenous & past-y separately.														
														
SHAP — importance over channels.														
														
NTK Spectrum / PR — geometry analysis.														
														
ΔR² vs noise/missingness — robustness.														


**CODEX SECTION**

**data descriptions**

Jena Climate Dataset
Context
Jena Climate is weather timeseries dataset recorded at the Weather Station of the Max Planck Institute for Biogeochemistry in Jena, Germany.

Content
Jena Climate dataset is made up of 14 different quantities (such air temperature, atmospheric pressure, humidity, wind direction, and so on) were recorded every 10 minutes, over several years. This dataset covers data from January 1st 2009 to December 31st 2016.


Human Activity Recognition

Human Activity Recognition (HAR) using smartphones dataset and an LSTM RNN. Classifying the type of movement amongst six categories:

WALKING,
WALKING_UPSTAIRS,
WALKING_DOWNSTAIRS,
SITTING,
STANDING,
LAYING.

The sensor signals (accelerometer and gyroscope) were pre-processed by applying noise filters and then sampled in fixed-width sliding windows of 2.56 sec and 50% overlap (128 readings/window). The sensor acceleration signal, which has gravitational and body motion components, was separated using a Butterworth low-pass filter into body acceleration and gravity. The gravitational force is assumed to have only low frequency components, therefore a filter with 0.3 Hz cutoff frequency was used.

Industrial Data from the Electric Arc Furnace

1. General design of an Electric Arc Furnace (EAF)
Basic components of modern electric arc furnaces include:

mechanical frame;
electric circuits;
equipment to deliver process gases, powdery and bulk materials into the working chamber;
process waste removal and gas scrubbing system;
automated process control system.
Modern electric arc furnaces are built with the following structural elements:

foundation;
tilt platform;
furnace body;
roof;
graphite electrodes;
electrode arms;
lifting and rotating mechanism for the roof and electrodes;
operating door.
2. EAF basic specifications
2.1 Electric arc furnace includes the following components:
refractory-lined lower casing with an eccentric bottom tapping system;
upper casing with water-cooled panels;
roof (large water-cooled and small uncooled) with refractory concrete lining.
2.2 Basic technical and performance specifications of DSP-120 electric arc furnace are as follows:
Rated furnace size — 142 m3
Furnace capacity — 140 tons
Rated tapping heat size — 125 tons
Furnace heel — 15-20 tons (± 3)
Transformer power — 120 MVA
Heat tapping type — eccentric bottom tapping
Tapping hole diameter — 180 mm
Operating door size — 1,000x1,200 mm
Transformer operating stages — 17
Primary voltage — 35 kV ± 10%
Roof lift — 400 mm
Roof speed — 40 mm/s
Furnace tilt angle for steel tapping — Max 15°
Furnace tilt angle for slag skimming — Max 8°
Maximum secondary voltage — 1,250 V
Rated secondary current — 70 kA
Electrode pitch circle diameter — 1,300 mm
Inner hearth diameter — 7,100 mm
Inner casing diameter — 7,300 mm
Electrode diameter — 610 mm
Electrode length — 2,400 mm
2.3 EAF gas-oxygen modules.
The system of gas-oxygen modules (burners) includes 4 multi-fuel gas-oxygen burners positioned inside vertical water-cooled panels.
The multi-fuel gas-oxygen burners can operate in burner mode to heat and melt the bulk charge, as well as in supersonic oxygen injection mode for bath lancing to trim the charge and foam the slag during the refining period.

2.4 Arrangement for injection of powdery carbon-containing materials into the EAF.
This arrangement enables:

partial deoxidation of furnace slag;
slag foaming and foam maintenance to protect the lining from arc heat and stabilize the lining.
Three carbon injectors are built into the side panels of the furnace to inject carbon-containing materials (CCM). The injectors are installed above the threshold level to keep them safe from any damage. Continuous purging with compressed air prevents the injectors from clogging with slag or metal splash.

Technical specifications of the arrangement for injection of powdery carbon-containing materials (fine coke, 0-3 mm crushed graphite) are as follows:

Loading hopper size — 50 m3
Flow rate per injector — 1 × 15-25 kg/min
Delivery medium — compressed air
Carrier gas pressure — 4-6 MPa
2.5 Equipment for feeding ferrous alloys and bulk materials. Equipment used to store and feed materials into the EAF and pouring ladle includes a set of receiving hoppers, storage hoppers, intermediate weighing hoppers, feeders, and conveyor belts.
3. EAF semi-product smelting process
First, the charge is filled into EAF. Melting of the charge begins at the lowest voltage levels with a minimum arc length. The voltage level is then increased. The arc energy melts the metal charge and slagging materials, heating the metal to tapping temperature and balancing the heat losses. Arc power and length are adjusted by selecting the appropriate transformer stage.
Slagging materials, additions, deoxidizers, and other materials are delivered to the furnaces by a power-driven transport hopper system.
To enable dephosphorization and make the furnace lining more durable, the slag in the EAF must be highly basic and magnesial. Magnesium-lime flux is added to the furnace mix to obtain a 7-10% range of MgO content in the slag.
Carbon-containing material is fed into the furnace using a hopper system to stabilize the arc, carbonize the metal in the furnace, and foam the slag (if carbon injectors malfunction).
To protect the walls and roof of the arc furnace from the arc heat, maximum shielding of the arc with slag is used during the charge melting and oxidation period. Slagging materials are added in measured portions through the top (roof) feeding hole to maintain the desired composition of the slag in accordance with the melting energy process conditions.

4. Oxidation period
Goals of the oxidation period:

oxidize carbon and generate additional chemical energy followed by heat release (in addition to supplied electricity) for smelting operations due to exothermic reactions;
remove phosphorus down to values that ensure desired chemical composition due to slag introduction;
ensure boiling and mixing of metal thanks to production of carbon monoxide, homogenize metal over temperature, and prevent saturation of metal with nitrogen and hydrogen thanks to foamed slag and arc shielding.
The oxidation period begins after complete melting of the charge and achievement of a “flat” bath. During this period, a sample is drawn to control the chemical composition of metal and measure temperature with MORE automatic unit.
Oxidation of impurities in the molten mass is achieved by purging the bath with gaseous oxygen through the multi-fuel gas-oxygen modules.
During the oxidation period, foaming of the slag by injecting carbon-containing material (CCM) through carbon injector is required with the purpose of shielding the arc to reduce saturation of the metal with gases and ensure complete transfer of the arc energy into the metal “bath”.
At the end of the oxidation period, shopfloor manager decides whether or not to tap the melt heat, based on the findings of chemical analysis of the metal to determine carbon content carried out by an express test lab, or based on the carbon content measured by Multi-Lab III Celox device, oxidation and heating of the metal to the required tapping temperature.
Oxidation of the metal should be measured before tapping the melt heat.

5. Tapping of the melt heat
By the end of the tapping, the weight of metal in the ladle varies from 120 to 125 tons (as measured by ladle car scales).
The metal is tapped into the ladle with maximum cut-off of oxidizing furnace slag. If furnace slag gets into the steel ladle when the melt heat is tapped, the foaming slag should be deposited by releasing aluminum pellets or "ingots".
Once the heat is tapped and the ladle with metal is removed from under the furnace, the metal in the ladle must be subject to a compulsory “soft” purging with argon for 1-3 minutes.

6. Deoxidation and alloying of metal
Deoxidizers and ferrous alloys are released into the ladle from the supply bins during the melt heat tapping process.
The delivery rate of deoxidizers and alloying agents (A), tons, is calculated based on the average content of the element in the finished steel using the following formula:

     А = ((B – C) × D × 100) / (E × R),                 (1)
where:
A – weight of ferrous alloy, tons;
B – mean content of element in finished steel, %;
C – content of element in steel prior to deoxidation, %;
D – weight of metal, including metal from previous melt, tons;
E – content of deoxidizer element in ferrous alloys, %;
R – recovery of deoxidizer element, %.

7. Metal temperature and oxidation degree control
The initial measurement of metal temperature and oxidation degree is carried out immediately after a flat bath is achieved and 42-46 MW of power is used up.
Readings of metal temperature and oxidation degree before the melt heat tapping are recorded in the melting chart. The time from the latest temperature measurement to the start of melt heat tapping should not exceed 3 minutes.
Intermediate measurement of the metal temperature is recommended (depending on the heating stage).

8. Description of data files
1 Chemical measurements in EAF (eaf_final_chemical_measurements.csv)

HEATID – heat identification number
POSITIONROW – measurement number
DATETIME – measurement date and time
VALC, VALSI, VALMN, VALP, VALS, VALCU, VALCR, VALMO, VALNI, VALAS, VALSN, VALN, VALZN – values of chemical elements, %.
Note: not all heats have details on chemical composition in EAF.
2 Measurements of temperature and oxidation degree in EAF (eaf_temp.csv)

HEATID – heat identification number
DATETIME – measurement date and time
TEMP – temperature, °C
VALO2_PPM – oxidation degree, ppm
Note: if oxidation degree is 0, it means no oxidation measurement was done (only temperature measurement)
3 Initial chemical measurement at ladle furnace (lf_initial_chemical_measurements.csv)

HEATID – heat identification number
POSITIONROW – measurement number
DATETIME – measurement date and time
VALC, VALSI, VALMN, VALP, VALS, VALAL, VALCU, VALCR, VALMO, VALNI, VALV, VALTI, VALNB, VALCA, VALW, VALB, VALAS, VALSN, VALN – , %.
4 Additions at EAF tapping (ladle_tapping.csv)

HEATID – heat identification number
MAT_CODE – material code
MAT_DEC – material name
CHARGE_AMOUNT – weight of addition
DATETIME – date and time of addition
5 Loading furnace from the basket (basket_charged.csv)

HEATID – heat identification number
MAT_CODE – material code
MAT_DEC – material name
CHARGE_AMOUNT – weight of addition
DATETIME – date and time of addition
6 Additional charge and loading of furnace (eaf_added_materials.csv)

HEATID – heat identification number
MAT_CODE – material code
MAT_DEC – material name
CHARGE_AMOUNT – weight of addition
DATETIME – date and time of addition
7 Additions to ladle furnace before initial chemical measurement (lf_added_materials.csv)

DATETIME – date and time of addition
HEATID – heat identification number
MAT_CODE – material code
DESCR – material name
MAT_CHARGED – weight of addition
8 EAF transformer data (eaf_transformer.csv)

TAP – transformer stage
HEATID – heat identification number
STARTTIME – electrode operation start time
DURATION – electrode operation duration
MW – electricity consumption
9 Usage of injected carbon in EAF (inj_mat.csv)

REVTIME – date and time
INJ_AMOUNT_CARBON – carbon usage amount
INJ_FLOW_CARBON – carbon injection flow rate
HEATID – heat identification number
Note: Usage of carbon for smelting starts at zero and adds up. Please mind that it may take some time to reset carbon usage for new melt heat.
10 Gas and oxygen usage in EAF (eaf_gaslance_mat.csv)

REVTIME – date and time
O2_AMOUNT – oxygen usage amount
GAS_AMOUNT – gas usage amount
O2_FLOW – oxygen flow rate
GAS_FLOW – gas flow rate
HEATID – heat identification number
Note: Usage of oxygen and gas for smelting starts at zero and adds up. Please mind that it may take some time to reset carbon usage for new melt heat.
11 Materials description with the values of chemical elements in it (ferro.csv)

9. Main problems
Temperature forecasting (target is in the "eaf_temp.csv" file);
Oxidation of steel forecasting (target is in the "eaf_temp.csv" file);
Chemical composition of steel after tapping steel from an electric arc furnace (target is in the "eaf_final_chemical_measurements.csv" file).

Beijing Multi-Site Air-Quality Data Set

About Dataset
Context
PM2.5 readings are often included in air quality reports from environmental authorities and companies. PM2.5 refers to atmospheric particulate matter (PM) that have a diameter less than 2.5 micrometers. In other words, it's used as a measure of pollution.

Content
This data set includes hourly air pollutants data from 12 nationally-controlled air-quality monitoring sites. The air-quality data are from the Beijing Municipal Environmental Monitoring Center. The meteorological data in each air-quality site are matched with the nearest weather station from the China Meteorological Administration. The time period is from March 1st, 2013 to February 28th, 2017.
**revised plan**

**cells to run in colab, separate the cells with a row of '=' symbols**