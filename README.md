# CS2 Market Item Price Forecasting Framework

## System Architecture Overview
This framework implements a Direct Multi-Horizon forecasting pipeline for Counter-Strike 2 item prices. It processes hierarchical JSON data structures, engineers time-series features (momentum, volatility, trend duration), and predicts market prices $t+1$ to $t+16$ days into the future. 

The pipeline utilizes tree-based gradient boosting models (HistGradientBoosting, XGBoost, LightGBM) to detect structural market changes ("pump-and-dump" schemes). A strictly chronological `TimeSeriesSplit` cross-validation ensures robust backtesting without data leakage. The inference engine leverages Quantile Regression to provide statistical confidence scores based on prediction interval variance.

## Environment Setup
Use an isolated conda environment to prevent dependency conflicts.

```bash
conda create -n cs2_predictor python=3.10 -y
conda activate cs2_predictor
pip install pandas numpy scikit-learn xgboost lightgbm ujson