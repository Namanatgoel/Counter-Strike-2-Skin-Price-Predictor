# CS2 Skin Price Prediction Framework

## Complete Setup & User Guide — No Prior Experience Required

---

## Table of Contents

1. [What Does This Project Do?](#1-what-does-this-project-do)
2. [How the Prediction Works — Plain English](#2-how-the-prediction-works--plain-english)
3. [Project Structure — Every File Explained](#3-project-structure--every-file-explained)
4. [Machine Learning Models Used](#4-machine-learning-models-used)
5. [System Requirements](#5-system-requirements)
6. [Installation — Step by Step](#6-installation--step-by-step)
7. [How to Run the Project](#7-how-to-run-the-project)
8. [Understanding the Output](#8-understanding-the-output)
9. [The Complete Data Pipeline](#9-the-complete-data-pipeline)
10. [Features Explained](#10-features-explained)
11. [Troubleshooting](#11-troubleshooting)
12. [How to Extend the Framework](#12-how-to-extend-the-framework)

---

## 1. What Does This Project Do?

This is a **machine learning pipeline** that predicts CS2 (Counter-Strike 2) skin prices **8 days into the future**.

You give it historical price data (stored as JSON files) for CS2 skins across multiple marketplaces (Steam, Skinport, etc.), and it:

- **Learns patterns** from that historical data — trends, volatility spikes, momentum reversals
- **Predicts** where the price of each skin is most likely to be in 8 days
- **Quantifies uncertainty** — instead of one number, it gives you a range: "The price will be between $10 and $15, most likely $12"
- **Explains its reasoning** — shows you which signals (rising RSI? falling momentum?) drove each prediction
- **Generates interactive charts** you can open in any web browser

A key goal is detecting **pump-and-dump schemes** common in CS2 skin markets — where a price is artificially inflated and then crashes. The features engineered here (trend duration, RSI, volatility) are specifically chosen to flag these patterns.

---

## 2. How the Prediction Works — Plain English

Imagine you are an experienced trader looking at a skin's price chart. You notice things like:

- "The price has been going up for 7 straight days — that's usually a pump, a crash is coming"
- "The RSI is at 85 — extremely overbought, prices rarely stay this high"
- "Volatility has spiked — the market is unsettled, hard to predict"
- "The price is touching the top of its Bollinger Band — historically, it reverts from here"

This project teaches a computer to notice exactly those same patterns, across 20,000+ skins simultaneously, much faster than any human can.

**The Training Phase** (done once):
The model reads 31 days of historical price data, computes dozens of indicators for each skin, and learns the statistical relationship between "what the chart looked like on day X" and "what the price was on day X+8."

**The Inference Phase** (for predictions):
When you want to predict a skin's price in 8 days, you feed the model today's indicators for that skin, and it applies what it learned during training to estimate the future price.

**Why give a range instead of one number?**
The future is uncertain. A narrow range (e.g., $10–$10.50) means the model is very confident. A wide range ($8–$15) means there is high uncertainty (high volatility, conflicting signals). The confidence score (0 to 1) summarizes this — a score of 0.9 means the model is highly confident; 0.1 means it's essentially guessing.

---

## 3. Project Structure — Every File Explained

```
cs2_prediction/
│
├── main.py                        ← START HERE. Runs the entire pipeline.
├── requirements.txt               ← List of Python libraries to install.
├── README.md                      ← This file.
│
├── src/                           ← All source code modules
│   ├── ingestion.py               ← Loads JSON data, cleans, resamples to daily
│   ├── features.py                ← Computes RSI, MACD, Bollinger Bands, etc.
│   ├── training.py                ← Trains HistGBM, XGBoost, LightGBM models
│   ├── inference.py               ← Generates probabilistic (P5/P50/P95) forecasts
│   ├── tuning.py                  ← Finds optimal hyperparameters with Optuna
│   └── validation.py              ← Backtesting with TimeSeriesSplit cross-validation
│
├── data/                          ← Input JSON data files (you provide these)
│   ├── model_training_skin.json   ← 31+ days of historical price data (training)
│   └── model_test_skin.json       ← Recent price data for model evaluation (test)
│
└── output/                        ← Generated charts (created automatically)
    ├── forecast_intervals.html    ← Price forecast with confidence band
    ├── shap_summary.html          ← Feature importance visualization
    └── volatility_regimes.html    ← Price + volatility overlay chart
```

### What Each Source File Does

| File | Responsibility | Key Functions |
|------|---------------|---------------|
| `ingestion.py` | Reads and cleans raw data | `load_and_flatten_json`, `clean_and_resample` |
| `features.py` | Computes technical indicators | `build_internal_features`, `merge_external_factors` |
| `training.py` | Trains all three ML models | `MultiHorizonTrainer.train_models` |
| `inference.py` | Generates probabilistic forecasts | `QuantileInferenceEngine.predict_with_confidence` |
| `tuning.py` | Optimizes model hyperparameters | `HyperparameterOptimizer.optimize` |
| `validation.py` | Backtests model accuracy | `backtest_pipeline` |

---

## 4. Machine Learning Models Used

This project trains three different gradient boosting models and compares them. Here is a plain-English explanation of each.

### What is Gradient Boosting?

Gradient boosting is a machine learning technique that builds a series of **decision trees** one at a time. Each new tree learns from the mistakes of all previous trees. The final prediction is the sum of all trees' outputs.

Think of it like a team of advisors: the first advisor makes predictions, the second advisor studies where the first was wrong and corrects those mistakes, the third advisor corrects what the second missed, and so on. After 100–300 rounds of this, the collective prediction is very accurate.

---

### Model 1: HistGradientBoostingRegressor (scikit-learn)

**Library**: scikit-learn (the most widely used Python ML library)

**How it's different from standard gradient boosting**:
Instead of looking at every possible split point for every feature (which is slow), it first groups feature values into "bins" (histograms), then finds the best split among those bins. This makes it:
- **Faster** to train on large datasets
- **Memory efficient** — fewer values to compare
- **Handles missing values natively** — you don't need to fill NaN before training

**Best for**: Large datasets where training speed matters.

**Hyperparameter used**: `random_state=42` (ensures reproducible results — the same data always produces the same model)

---

### Model 2: XGBoost (eXtreme Gradient Boosting)

**Library**: xgboost

**Why it's popular**: XGBoost became famous for winning many Kaggle competitions around 2014–2018. It adds **regularization** (mathematical penalties for complexity) on top of standard gradient boosting, which helps prevent overfitting — a situation where the model memorizes the training data but fails on new data.

**Key parameters used**:
- `objective='reg:squarederror'`: The model minimizes the squared difference between its prediction and the actual value (standard regression target)
- `n_jobs=-1`: Uses all available CPU cores for faster training
- `random_state=42`: Reproducibility

**Best for**: Balanced performance across many types of tabular data; good default choice.

---

### Model 3: LightGBM (Light Gradient Boosting Machine)

**Library**: lightgbm

**Why it's special**: LightGBM uses **leaf-wise tree growth** instead of the level-wise growth used by XGBoost and HistGBM. In level-wise growth, all leaves at a given depth are split before going deeper. In leaf-wise growth, the algorithm always splits whichever single leaf will reduce the loss the most — regardless of tree depth.

This means LightGBM:
- Trains **3–10× faster** than XGBoost on the same dataset
- Often achieves **better accuracy** with the same number of trees
- Is more prone to overfitting on very small datasets (less of a concern here)

**Key parameters used**:
- `verbose=-1`: Suppresses per-iteration training logs (cleaner output)
- `n_jobs=-1`: All CPU cores
- `random_state=42`: Reproducibility

**Best for**: Large datasets, speed-sensitive environments, and this project's primary model.

**LightGBM is used for the quantile regression (uncertainty) models** because its `objective='quantile'` implementation is the most mature and accurate of the three libraries.

---

### Quantile Regression — The Secret Behind Confidence Scores

Normal regression predicts the **average** outcome. Quantile regression predicts a **specific percentile** of the outcome distribution.

By training three models with quantile targets of 0.05, 0.50, and 0.95:
- **P5 model**: Tries to predict a value that the true price will exceed 95% of the time (pessimistic lower bound)
- **P50 model**: Tries to predict the median — half of true prices will be above, half below (the main prediction)
- **P95 model**: Tries to predict a value that the true price will be below 95% of the time (optimistic upper bound)

The [P5, P95] range is the **90% prediction interval**.

**Confidence Score** = 1 − (interval_width / median_prediction)

A narrow interval → confidence near 1.0.
A wide interval → confidence near 0.0.

---

### Hyperparameter Optimization with Optuna (Bayesian Search)

**Library**: optuna

Hyperparameters are settings you choose before training — like `num_leaves`, `learning_rate`, etc. Finding the right values can dramatically improve model performance.

**Three approaches compared**:

| Method | How it works | Problem |
|--------|-------------|---------|
| Grid Search | Test every combination | Exponentially slow (10×10×10 = 1000 tests) |
| Random Search | Test random combinations | Wastes trials on bad regions |
| **Bayesian Optimization (Optuna TPE)** | Builds a model of which settings work, searches smartly | **Fast, intelligent** |

**TPE = Tree-structured Parzen Estimator**: Optuna builds two probability distributions — one of parameters that gave good results, one of parameters that gave bad results — and samples new parameters from regions where "good result" probability is high.

With 20 trials, Optuna finds near-optimal settings that Random Search would need 200+ trials to discover.

**GroupKFold Cross-Validation**: During optimization, the data is split into 5 folds using `GroupKFold`, which ensures that all price data for a given skin item is always entirely in training or entirely in testing — never split across folds. This prevents data leakage.

---

### SHAP Values — Understanding Why the Model Predicts What It Does

**Library**: shap

After training, we use SHAP (SHapley Additive exPlanations) to understand which features drove each prediction.

**The intuition**: Imagine explaining a prediction by asking "how much would the prediction change if I didn't know this feature?" SHAP computes a mathematically rigorous answer to that question for every feature and every prediction.

- **Positive SHAP value** → that feature pushed the prediction higher (more bullish)
- **Negative SHAP value** → that feature pushed the prediction lower (more bearish)
- **Large absolute SHAP value** → that feature was very important for this prediction
- **SHAP near 0** → that feature barely mattered for this prediction

The SHAP summary chart shows this for all training observations simultaneously, revealing which features are globally important and in which direction they push predictions.

---

## 5. System Requirements

### Operating System
- Windows 10 or later
- macOS 10.15 (Catalina) or later
- Ubuntu 20.04 or later (or any modern Linux)

### Hardware
- RAM: 8 GB minimum (16 GB recommended for the full 110MB training file)
- CPU: Any modern processor (more cores = faster training due to `n_jobs=-1`)
- Storage: 2 GB free space (for data files and Python packages)

### Software Prerequisites
- Python 3.10 or later
- Anaconda or Miniconda (strongly recommended — see installation steps)
- Internet connection (for installing packages)

---

## 6. Installation — Step by Step

**What is Anaconda/Miniconda?** It is a tool that creates isolated "environments" — separate Python installations that don't interfere with each other. This keeps the CS2 predictor's libraries separated from any other Python projects you might have.

---

### Step 1: Install Miniconda

**Windows:**
1. Go to: https://docs.conda.io/en/latest/miniconda.html
2. Click "Miniconda3 Windows 64-bit" and download the installer (.exe file)
3. Double-click the downloaded file and follow the installer wizard
4. When asked "Add Miniconda3 to my PATH", **check this box**
5. Click Finish

**Mac:**
1. Go to: https://docs.conda.io/en/latest/miniconda.html
2. Download the "Miniconda3 macOS" installer for your chip (Intel or Apple Silicon/M1/M2)
3. Open Terminal (press Cmd+Space, type "Terminal", press Enter)
4. Navigate to your Downloads folder: `cd ~/Downloads`
5. Run: `bash Miniconda3-latest-MacOSX-x86_64.sh` (adjust filename to what you downloaded)
6. Follow the prompts and type "yes" when asked

**Linux:**
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```
Follow prompts and type "yes" when asked to initialize conda.

---

### Step 2: Open the Correct Terminal

**Windows:** Open "Anaconda Prompt" from the Start Menu (NOT regular Command Prompt — Anaconda Prompt has conda available).

**Mac/Linux:** Open "Terminal".

You should see a prompt that starts with `(base)` — this means conda is active.

---

### Step 3: Create an Isolated Environment

This creates a fresh Python 3.10 environment named `cs2_predictor`:

```bash
conda create -n cs2_predictor python=3.10 -y
```

This downloads Python 3.10 and sets up a clean environment. It takes 1–3 minutes.

---

### Step 4: Activate the Environment

```bash
conda activate cs2_predictor
```

Your prompt should now start with `(cs2_predictor)` instead of `(base)`. **You must run this activation command every time you open a new terminal and want to use this project.**

---

### Step 5: Navigate to the Project Folder

First, make sure you have extracted the project ZIP file. Then navigate to it:

**Windows** (example — adjust path to where you extracted the files):
```bash
cd C:\Users\YourName\Downloads\cs2_prediction
```

**Mac/Linux** (example):
```bash
cd ~/Downloads/cs2_prediction
```

To check you are in the right folder, run:
```bash
ls
```
You should see: `main.py`, `requirements.txt`, `src/`, `data/`, `output/`

---

### Step 6: Install All Required Libraries

```bash
pip install -r requirements.txt
```

This reads `requirements.txt` and installs all Python libraries the project needs. It will download and install approximately 500 MB of packages and may take 3–10 minutes depending on your internet speed.

**What gets installed:**

| Library | Version | Purpose |
|---------|---------|---------|
| `pandas` | latest | Data manipulation, DataFrame operations |
| `numpy` | latest | Fast numerical arrays and math |
| `scikit-learn` | latest | HistGradientBoosting, cross-validation, metrics |
| `xgboost` | latest | XGBoost gradient boosting model |
| `lightgbm` | latest | LightGBM gradient boosting model |
| `optuna` | latest | Bayesian hyperparameter optimization |
| `ujson` | latest | Fast JSON parsing (C extension) |
| `plotly` | latest | Interactive HTML charts |
| `shap` | latest | Model explainability (SHAP values) |

---

### Step 7: Verify Installation

Run this to confirm everything installed correctly:

```bash
python -c "import pandas, numpy, sklearn, xgboost, lightgbm, optuna, ujson, plotly, shap; print('All packages OK!')"
```

You should see:
```
All packages OK!
```

If you see an error like `ModuleNotFoundError: No module named 'lightgbm'`, re-run `pip install -r requirements.txt` and check your internet connection.

---

## 7. How to Run the Project

### Prerequisites Before Running

1. Your terminal must be in the project root directory (the folder containing `main.py`)
2. Your conda environment must be activated (`(cs2_predictor)` visible in prompt)
3. Both data files must be present in the `data/` folder:
   - `data/model_training_skin.json`
   - `data/model_test_skin.json`

### The Only Command You Need

From the project root directory:

```bash
python main.py
```

That's it. The entire pipeline runs automatically. You will see progress printed to the terminal:

```
Loading data/model_training_skin.json...
Building features...

--- Hyperparameter Optimization (8-Day Horizon) ---
Best Hyperparameters Found:
  learning_rate: 0.023
  num_leaves: 87
  max_depth: 7
  min_child_samples: 34

--- Training Multi-Algorithm Models ---

--- Processing Test Set ---
Loading data/model_test_skin.json...
Building features...

--- Evaluation (8-Day Horizon) ---
  Test RMSE:               $0.0142
  Test MAPE:               4.37%
  Directional Accuracy:    61.82%

--- Generating Visualizations ---
  Plotting forecast for: AK-47 | Redline (Field-Tested)

✓ Pipeline Complete.
  Outputs saved to output/:
    - forecast_intervals.html  (open in any web browser)
    - shap_summary.html        (open in any web browser)
    - volatility_regimes.html  (open in any web browser)
```

**Expected runtime**: 5–20 minutes depending on your hardware and data size.

### Viewing the Charts

After the pipeline completes, navigate to the `output/` folder and double-click any `.html` file. It opens in your default web browser (Chrome, Firefox, Edge, Safari — all work).

---

## 8. Understanding the Output

### Performance Metrics Printed to Terminal

**Test RMSE (Root Mean Squared Error)**
- Unit: dollars (same unit as the price)
- Meaning: On average, the model's price prediction is off by this many dollars from the actual future price
- Lower is better
- Example: RMSE of $0.014 on a $10 item = average error of 0.14%

**Test MAPE (Mean Absolute Percentage Error)**
- Unit: percent (%)
- Meaning: On average, the model's prediction differs from the true price by this percentage
- Lower is better
- A MAPE of 4.37% means predictions are off by 4.37% on average
- Good financial forecasting typically achieves 3–8% MAPE on short horizons

**Directional Accuracy**
- Unit: percent (%)
- Meaning: How often did the model correctly predict whether the price would go UP or DOWN?
- 50% = random guessing; higher is better
- 60%+ is considered meaningful for financial forecasting
- 70%+ would be excellent

### Chart 1: forecast_intervals.html

This is the main prediction chart for one randomly selected skin.

**What you see:**
- **Black solid line**: The actual historical prices (what really happened)
- **Blue dashed line**: The model's P50 (median) prediction — where it thinks the price will be
- **Blue shaded region**: The 90% prediction interval (P5 to P95 band)

**How to read it:**
- Where the band is **narrow**: the model is confident
- Where the band is **wide**: the model is uncertain
- The blue dashed line extending beyond the black line = the forecast for future dates

**Interactive features:**
- Hover over any point to see exact date and price values
- Click and drag to zoom in on a specific date range
- Double-click to zoom back out
- Click a series name in the legend to hide/show it

### Chart 2: shap_summary.html

This chart explains which features influenced the model's predictions the most.

**What you see:**
- Each **row** is one feature (e.g., rsi_14, volatility_14, mom_7)
- Each **dot** is one training observation
- **Dot position left/right** = how strongly that feature pushed the prediction down/up
- **Dot color** = the feature's value (purple=low, yellow=high, from the Viridis colorscale)

**How to read it:**
- Features at the **top** are most globally important
- Yellow dots (high feature value) clustered on the right → high feature value = bullish prediction
- Purple dots (low feature value) clustered on the left → low feature value = bearish prediction

**Example interpretation:**
"rsi_14 appears at the top with yellow dots on the right: a high RSI (overbought) tends to push the model's prediction higher, consistent with momentum continuation behavior."

### Chart 3: volatility_regimes.html

This chart shows price and volatility on the same timeline with high-volatility periods highlighted.

**What you see:**
- **Dark blue line** (left axis): The skin's spot price over time
- **Orange line** (right axis): 14-day rolling volatility (how much prices fluctuated day-to-day)
- **Red shaded regions**: Periods where volatility was in the top 10% of all observed values

**How to read it:**
- Red regions = turbulent market periods (spikes, pumps, dumps)
- If price drops occur within red regions → volatility was a valid warning signal
- Calm periods (no red) = stable pricing, more predictable

---

### Excel / CSV Predictions Output

The pipeline saves a tidy table of test-set predictions to `output/predictions.xlsx` (Excel) when an Excel writer is available. If the `openpyxl` package is not installed, the pipeline falls back to `output/predictions.csv` (CSV).

File location:
- `output/predictions.xlsx` or `output/predictions.csv`

Columns included (one row per prediction):
- `item_name`: item identifier used in the dataset
- `current_date`: the date for which the model used features (ISO 8601 timestamp)
- `predicted_date`: the forecast date (usually `current_date` + 8 days)
- `current_price`: spot price at `current_date`
- `true_price`: the actual observed price at `predicted_date` (if available)
- `P5`: 5th-percentile predicted price (lower bound)
- `Prediction_P50`: median predicted price (primary forecast)
- `P95`: 95th-percentile predicted price (upper bound)
- `Confidence_Score`: model confidence in [0.0, 1.0] (higher = narrower interval)

Notes:
- To enable Excel writing, make sure `openpyxl` is installed. It's included in `requirements.txt`; run:

```bash
pip install -r requirements.txt
```

- If Excel writing fails, the pipeline always writes `output/predictions.csv` as a compatible fallback.
- You can load the file into pandas:

```python
import pandas as pd
df = pd.read_excel('output/predictions.xlsx')  # or pd.read_csv('output/predictions.csv')
```

## 9. The Complete Data Pipeline

Here is every step the code executes, in order, with the exact function responsible:

### Step 1 — JSON Loading (`ingestion.py: load_and_flatten_json`)

The input data is a nested JSON file with this structure:
```json
{
  "knife": {
    "Karambit | Fade (Factory New)": {
      "steam": {
        "1716998400": 1250.00,
        "1717084800": 1255.50
      },
      "skinport": {
        "1716998400": 1230.00
      }
    }
  }
}
```

The function:
1. Reads the entire file using `ujson` (3–5× faster than Python's built-in `json`)
2. Splits the top-level categories across multiple CPU cores using `multiprocessing.Pool`
3. Each CPU core converts its assigned category into a flat list of records
4. All records are combined into one pandas DataFrame
5. DataFrame is indexed by timestamp and sorted chronologically

**Output**: A DataFrame with columns: `timestamp` (index), `category`, `item_name`, `marketplace_provider`, `spot_price`

---

### Step 2 — Resampling & Cleaning (`ingestion.py: clean_and_resample`)

The raw data has irregular timestamps — some items might have 5 prices on Monday and none on Tuesday. This step:

1. **Groups** by each unique (category, item, provider) combination
2. **Resamples** each group to exactly one price per calendar day (keeping the last price of each day)
3. **Forward-fills** up to 3 consecutive missing days (no-trade days carry the last known price)
4. **Drops** any days that are still missing after the 3-day fill window

**Output**: One row per (item, provider, day), clean daily prices

---

### Step 3 — Feature Engineering (`features.py: build_internal_features`)

For each item on each marketplace, computes:

| Feature | Formula | What it measures |
|---------|---------|-----------------|
| `sma_3` | 3-day SMA / price − 1 | How far price is from 3-day average |
| `sma_7` | 7-day SMA / price − 1 | How far price is from 7-day average |
| `sma_14` | 14-day SMA / price − 1 | How far price is from 14-day average |
| `mom_3` | (price − price[t−3]) / price[t−3] | 3-day momentum (% change) |
| `mom_7` | (price − price[t−7]) / price[t−7] | 7-day momentum (% change) |
| `volatility_14` | std of daily returns over 14 days | Price instability |
| `trend_duration` | Consecutive days in same direction | How long the current trend has lasted |
| `rsi_14` | RSI formula with 14-day window | Overbought/oversold signal (0–100) |
| `macd` | (EMA12 − EMA26) / price | Short-term vs long-term momentum ratio |
| `macd_signal` | EMA9 of MACD / price | Smoothed MACD trigger |
| `macd_histogram` | MACD − MACD Signal / price | Momentum acceleration |
| `bb_upper` | Upper Bollinger Band / price − 1 | Distance to upper boundary |
| `bb_middle` | 20-day SMA / price − 1 | Distance to middle band |
| `bb_lower` | Lower Bollinger Band / price − 1 | Distance to lower boundary |
| `bb_position` | (price − lower) / (upper − lower) | Position within band (0=bottom, 1=top) |

---

### Step 4 — External Factor Merging (`features.py: merge_external_factors`)

Merges two additional columns into the feature matrix:
- `liquidity_vol`: Total trading volume across all marketplaces that day
- `cs2_players`: Steam concurrent player count for CS2 that day

Currently simulated with random numbers — replace with real API data in production.

---

### Step 5 — Target Computation (`main.py: compute_targets`)

Computes the prediction target for each horizon:

```
target_return = price(t + horizon) / price(t) - 1
```

For the 8-day horizon: how much does the price change in 8 days, as a fraction?
- +0.05 = price increased 5% in 8 days
- −0.03 = price decreased 3% in 8 days

The model learns to predict this return. At inference time, the return is converted back to an absolute price.

---

### Step 6 — Hyperparameter Optimization (`tuning.py: HyperparameterOptimizer.optimize`)

Runs 20 trials of Bayesian optimization to find the best LightGBM settings.
Each trial trains 5 cross-validated models (GroupKFold), totaling 100 model fits.
The best configuration (lowest mean RMSE across folds) is stored.

---

### Step 7 — Multi-Algorithm Training (`training.py: MultiHorizonTrainer.train_models`)

Trains three algorithms:
- `HistGradientBoostingRegressor` (sklearn)
- `XGBRegressor` (xgboost)
- `LGBMRegressor` (lightgbm)

Each using default hyperparameters (for benchmark comparison).
Also computes Permutation Feature Importance for each trained model.

---

### Step 8 — Quantile Inference Training (`inference.py: QuantileInferenceEngine.fit_horizon`)

Trains three LightGBM models with quantile objectives:
- One for P5 (pessimistic lower bound)
- One for P50 (median / primary prediction)
- One for P95 (optimistic upper bound)

---

### Step 9 — Test Set Evaluation

1. Runs Steps 1–5 on the test JSON file
2. Generates P5/P50/P95 predictions for all test observations
3. Computes RMSE, MAPE, and Directional Accuracy

---

### Step 10 — Visualization (`visualization.py`)

Generates three interactive HTML charts and saves them to the `output/` directory.

---

## 10. Features Explained

### Why These Specific Features?

Each feature was chosen because it captures a pattern known to affect CS2 skin prices:

**Simple Moving Averages (sma_3, sma_7, sma_14)**
When a skin's price rises far above its moving average, it often "reverts to the mean" — returning toward its average. This mean-reversion tendency is strong in illiquid markets. These features let the model detect how far prices have strayed.

**Momentum (mom_3, mom_7)**
CS2 skin pumps often show 3–7 consecutive days of rising momentum before the dump. If momentum_7 is very high and rsi_14 is above 70, that combination strongly suggests an imminent price reversal.

**Volatility (volatility_14)**
High-volatility periods are harder to predict. The model uses this to reduce confidence (widen the prediction interval) during turbulent periods.

**Trend Duration**
A 14-day unbroken uptrend is fundamentally different from a 2-day uptrend. Trend duration captures this persistence. Classic pump-and-dump cycles in CS2 often have trend durations of 5–12 days.

**RSI (rsi_14)**
- RSI > 70: Overbought — price rose very fast, correction likely
- RSI < 30: Oversold — price fell very fast, bounce likely
- RSI 40–60: Neutral — no strong signal

**MACD (macd, macd_signal, macd_histogram)**
The MACD histogram turning from positive to negative (crossing zero) is a classic signal that upward momentum is exhausting — directly relevant to catching the top of a pump.

**Bollinger Bands (bb_*)**
When price touches the upper Bollinger Band (bb_position near 1.0), it has moved 2 standard deviations above its 20-day average — statistically uncommon, often followed by reversion.

---

## 11. Troubleshooting

### "python: command not found" or "ModuleNotFoundError"

**Cause**: The conda environment is not activated.

**Fix**:
```bash
conda activate cs2_predictor
```

Then try again.

---

### "FileNotFoundError: data/model_training_skin.json"

**Cause**: The data files are missing from the `data/` folder.

**Fix**: Make sure both JSON files are in the `data/` directory inside the project folder. The expected paths are:
```
cs2_prediction/data/model_training_skin.json
cs2_prediction/data/model_test_skin.json
```

---

### Training is very slow (taking more than 30 minutes)

**Cause**: Large dataset or slow hardware.

**Quick fix**: Reduce the number of optimization trials in `main.py` line `optimizer = HyperparameterOptimizer(n_trials=20, ...)` — change `20` to `5` for faster runs at the cost of potentially worse hyperparameters.

---

### "MemoryError" during training

**Cause**: The full training JSON (~110MB) may require 4–6 GB of RAM to process.

**Fix**: Close other applications to free RAM, or use the smaller `model_test_skin.json` for both training and testing while developing.

---

### Chart files are generated but look empty or broken

**Cause**: Outdated browser or plotly compatibility issue.

**Fix**: Open with a modern browser (Chrome or Firefox recommended). Internet Explorer and very old Edge versions do not support modern JavaScript.

---

## 12. How to Extend the Framework

The code is designed to be extended. Here are the most common modifications:

### Adding a New Feature

In `src/features.py`, inside the `compute_metrics` function:
```python
# Example: Add a 30-day momentum feature
group['mom_30'] = price.pct_change(periods=30)
```
That's it. The new column automatically flows into training and inference.

### Adding Real External Data

In `main.py`, in the `prepare_pipeline` function, replace:
```python
# PLACEHOLDER
liquidity_df = pd.DataFrame({'liquidity_vol': np.random.randint(...)}, index=timestamps)
```
With:
```python
# Real data from your API
liquidity_df = fetch_real_liquidity_data(timestamps)  # your function here
```

### Adding a New Algorithm

In `src/training.py`, in `_get_base_estimator`:
```python
elif algo_name == 'catboost':
    from catboost import CatBoostRegressor
    return CatBoostRegressor(random_seed=42, verbose=0)
```
And add `'catboost': {}` to the `self.models` dict in `__init__`.

### Changing the Forecast Horizon

In `main.py`, change:
```python
horizons = [8]
```
To any horizon you want, for example:
```python
horizons = [1, 3, 8]  # predict 1, 3, and 8 days ahead
```
Note: horizons above 8 require more than 31 days of training data (see data constraint explanation in main.py).

### Running on Your Own Data

Your JSON file must follow this exact structure:
```json
{
  "category_name": {
    "item_name": {
      "marketplace_name": {
        "unix_timestamp_as_string": price_as_number
      }
    }
  }
}
```
Then in `main.py`, change the file paths:
```python
X_train, y_train_dict, df_train = prepare_pipeline('data/YOUR_TRAINING_FILE.json', horizons)
X_test, y_test_dict, df_test = prepare_pipeline('data/YOUR_TEST_FILE.json', horizons)
```

### Adding Your Own Data — Detailed Guide

Where to place files
- Put your datasets in the `data/` folder at the project root. The pipeline expects two files by default:
  - `data/model_training_skin.json` — historical data used for training
  - `data/model_test_skin.json`     — held-out recent data used for evaluation and inference

File format and timestamps
- Files must match the JSON nesting shown above. Keys for timestamps must be UNIX seconds as strings (e.g. "1716998400").
- Each timestamp represents a daily price for that item on that `marketplace_name`.
- The ingestion step resamples to one row per calendar day (keeps the last price for a day) and forward-fills up to 3 missing days. For best results provide daily or near-daily snapshots.

How predictions map to your data
- The model predicts a horizon `h` days ahead (default `h = 8`). For a row with `current_date = t`, the prediction refers to `predicted_date = t + h`.
- The pipeline only evaluates predictions where the true observed price at `t + h` exists in your test dataset. In other words, if your last timestamp in `data/model_test_skin.json` is `D_last`, the latest `current_date` that can be evaluated is `D_last - h`.

How many prediction days you will get
- Let `T` be the number of calendar days present in your test dataset (after resampling). Let `W` be the warm-up days required for feature computation (this project uses `W = 20` for 20-day Bollinger Bands). Then the number of usable rows per item for a forecast horizon `h` is approximately:

  usable_rows = T - W - h

Examples:
- If you provide 31 days of history (T=31), with W=20 and h=8 → usable_rows = 3 (this repo's default example)
- To have 30 usable rows for h=8, you need T = 30 + 20 + 8 = 58 days of data

If you want predictions for future dates beyond what exists in your test file
- Option 1 (recommended): Extend `data/model_test_skin.json` with real prices up to the date you want evaluated. The pipeline will then produce evaluated predictions up to `D_last - h`.
- Option 2 (forecast without true labels): If you only want point forecasts (no evaluation) for the most recent `current_date`s even though `t + h` is not present, you can run inference on all rows and accept that `true_price` will be NaN and metrics cannot be computed for those rows. To do this, edit `main.py` and replace the valid-mask at the evaluation step:

```python
# Original (only where true t+h exists):
valid_idx = y_target_8d_ret.notna()

# Predict for all rows (no true future required):
valid_idx = pd.Series(True, index=X_test_features.index)
```

After this change the pipeline will still save `output/predictions.csv`/`.xlsx` containing forecasts for all `current_date`s; `true_price` will be empty for rows where `t + h` was not available.

Changing the forecast horizon
- The forecast horizon(s) are defined in `main.py` at the top of the `main()` flow. To change them edit:

```python
horizons = [8]
```

and replace with any list of integers, e.g. `horizons = [1, 3, 8]`.
- Important: Increasing `h` requires more historical days (see usable_rows formula). If you set `h` greater than your available data permits, the computed targets will be mostly NaN and training/evaluation will fail or produce no usable rows.

Practical tips
- Start small: if you only have a subset of data, run the pipeline on `model_test_skin.json` alone to sanity-check ingestion and feature engineering.
- Use longer history (60–120 days) when you want multi-horizon forecasts (1, 3, 8, 16 days). More history improves feature warm-up and stability.
- Keep timestamps in UTC and use end-of-day or daily snapshots for consistency.
- If you only want deployment-style forecasts (no evaluation), consider creating a thin `data/deploy.json` that contains the latest prices and run a small wrapper that loads the trained models and produces predictions without computing metrics (I can help scaffold this if you want).

---

## Quick Reference: All Commands

```bash
# 1. Install Miniconda (first time only — see installation section)

# 2. Open Anaconda Prompt (Windows) or Terminal (Mac/Linux)

# 3. Create environment (first time only)
conda create -n cs2_predictor python=3.10 -y

# 4. Activate environment (EVERY TIME you open a new terminal)
conda activate cs2_predictor

# 5. Navigate to project folder (adjust path to where you extracted the files)
cd path/to/cs2_prediction

# 6. Install libraries (first time only)
pip install -r requirements.txt

# 7. Verify installation (first time only)
python -c "import pandas, numpy, sklearn, xgboost, lightgbm, optuna, ujson, plotly, shap; print('All OK')"

# 8. Run the pipeline
python main.py

# 9. View results — open any of these in your browser:
#    output/forecast_intervals.html
#    output/shap_summary.html
#    output/volatility_regimes.html
```

---
