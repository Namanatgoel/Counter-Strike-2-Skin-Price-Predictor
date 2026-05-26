import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap

def plot_forecast_intervals(df_actual: pd.DataFrame, df_pred: pd.DataFrame, item_name: str, output_path: str = None):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_actual.index, y=df_actual['spot_price'],
        mode='lines', name='Actual Price', line=dict(color='black', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=df_pred.index, y=df_pred['Prediction_P50'],
        mode='lines', name='P50 (Median)', line=dict(color='blue', width=2, dash='dash')
    ))
    
    fig.add_trace(go.Scatter(
        x=df_pred.index, y=df_pred['P95'],
        fill=None, mode='lines', line_color='rgba(0,0,255,0)', showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=df_pred.index, y=df_pred['P5'],
        fill='tonexty', mode='lines', line_color='rgba(0,0,255,0)',
        name='P5-P95 Interval', fillcolor='rgba(0,0,255,0.2)'
    ))
    
    fig.update_layout(
        title=f'Price Forecast with Confidence Intervals - {item_name}',
        xaxis_title='Date', yaxis_title='Price', hovermode='x unified', template='plotly_white'
    )
    if output_path: fig.write_html(output_path)
    return fig

def plot_shap_summary(model, X_train: pd.DataFrame, output_path: str = None):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    if isinstance(shap_values, list): shap_values = shap_values[0]
        
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[-15:][::-1]
    
    fig = go.Figure()
    for i, feature_idx in enumerate(top_indices):
        feature_name = X_train.columns[feature_idx]
        shap_vals = shap_values[:, feature_idx]
        feature_vals = X_train.iloc[:, feature_idx].values
        
        fig.add_trace(go.Scatter(
            x=shap_vals, y=[feature_name] * len(shap_vals),
            mode='markers', name=feature_name,
            marker=dict(size=5, color=feature_vals, colorscale='Viridis', showscale=(i == 0)),
            hovertemplate=f'<b>{feature_name}</b><br>SHAP: %{{x:.4f}}<br>Value: %{{marker.color:.4f}}<extra></extra>'
        ))
        
    fig.update_layout(
        title='SHAP Summary Plot - Top 15 Features',
        xaxis_title='SHAP Value', yaxis_title='Features',
        hovermode='closest', template='plotly_white', showlegend=False
    )
    if output_path: fig.write_html(output_path)
    return fig

def plot_volatility_regimes(df_item: pd.DataFrame, output_path: str = None):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(go.Scatter(
        x=df_item.index, y=df_item['spot_price'],
        mode='lines', name='Price', line=dict(color='darkblue', width=2)
    ), secondary_y=False)
    
    fig.add_trace(go.Scatter(
        x=df_item.index, y=df_item['volatility_14'],
        mode='lines', name='Volatility (14-day)', line=dict(color='orange', width=2)
    ), secondary_y=True)
    
    if 'volatility_14' in df_item.columns:
        vol_90 = df_item['volatility_14'].quantile(0.90)
        high_vol_mask = df_item['volatility_14'] > vol_90
        
        # Group contiguous high volatility periods
        changes = high_vol_mask.astype(int).diff()
        starts = df_item.index[changes == 1].tolist()
        ends = df_item.index[changes == -1].tolist()
        
        if high_vol_mask.iloc[0]: starts.insert(0, df_item.index[0])
        if high_vol_mask.iloc[-1]: ends.append(df_item.index[-1])
            
        for start, end in zip(starts, ends):
            fig.add_vrect(
                x0=start, x1=end, fillcolor='red', opacity=0.15, layer='below', line_width=0
            )
            
    fig.update_layout(
        title='Price and Volatility Regimes',
        xaxis_title='Date', hovermode='x unified', template='plotly_white'
    )
    fig.update_yaxes(title_text='Price', secondary_y=False)
    fig.update_yaxes(title_text='Volatility', secondary_y=True)
    
    if output_path: fig.write_html(output_path)
    return fig
