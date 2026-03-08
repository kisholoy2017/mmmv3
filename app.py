import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import statsmodels.api as sm
from scipy import stats
from scipy.optimize import minimize
from statsmodels.stats.outliers_influence import variance_inflation_factor  # NEW: For VIF
from functools import partial
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="MMM Analysis Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        font-weight: bold;
        margin-top: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stDataFrame {
        border: 2px solid #1f77b4;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'data_uploaded' not in st.session_state:
    st.session_state.data_uploaded = False
if 'media_data' not in st.session_state:
    st.session_state.media_data = {}
if 'kpi_data' not in st.session_state:
    st.session_state.kpi_data = None
if 'combined_data' not in st.session_state:
    st.session_state.combined_data = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'promotion_data' not in st.session_state:
    st.session_state.promotion_data = None

# Helper functions for data cleaning and MMM
def clean_numeric_column(series):
    """Clean numeric column - remove commas, convert to float"""
    if series.dtype == 'object':
        try:
            return pd.to_numeric(series.astype(str).str.replace(',', ''), errors='coerce')
        except:
            return series
    return series

def clean_dataframe_numeric_columns(df, exclude_cols=None):
    """Clean all numeric columns in dataframe"""
    if exclude_cols is None:
        exclude_cols = []
    
    df = df.copy()
    for col in df.columns:
        if col not in exclude_cols:
            if df[col].dtype == 'object':
                try:
                    cleaned = df[col].astype(str).str.replace(',', '')
                    df[col] = pd.to_numeric(cleaned, errors='ignore')
                except:
                    pass
    return df

def adstock_transformation(x, alpha=0.5):
    """Apply adstock (geometric decay) transformation"""
    y = np.zeros_like(x, dtype=float)
    if len(x) > 0:
        y[0] = x[0]
    for t in range(1, len(x)):
        y[t] = x[t] + alpha * y[t-1]
    return y

def hill_transformation(x, alpha, gamma):
    """
    Apply Hill saturation transformation with gamma-based inflection point
    
    Parameters:
    - x: Input values (adstocked spend)
    - alpha: Steepness of saturation curve (0.5-3.0)
    - gamma: Inflection point position (0.0-1.0)
             gamma=0.3 → saturation at 30% of range
             gamma=0.5 → saturation at midpoint
             gamma=1.0 → saturation at maximum
    
    Returns:
    - Saturated values between 0 and 1
    """
    x = np.asarray(x, dtype=float)
    x = np.maximum(x, 0.0)
    
    # Calculate inflection point (gamma-based)
    x_min = np.min(x)
    x_max = np.max(x)
    inflexion = x_min * (1 - gamma) + x_max * gamma
    
    # Ensure inflexion is positive and finite
    inflexion = max(float(inflexion), 1e-9)
    
    # Hill transformation: x^alpha / (x^alpha + inflexion^alpha)
    x_alpha = np.power(x, alpha)
    inflexion_alpha = np.power(inflexion, alpha)
    
    saturated = x_alpha / (x_alpha + inflexion_alpha)
    
    return saturated

def hill_derivative(x, alpha, gamma, x_range_min, x_range_max):
    """
    Calculate derivative of Hill function for marginal ROAS
    
    Parameters:
    - x: Input values
    - alpha: Steepness parameter
    - gamma: Inflection point position
    - x_range_min: Minimum of original x range (for inflection calculation)
    - x_range_max: Maximum of original x range (for inflection calculation)
    """
    x = np.asarray(x, dtype=float)
    x = np.maximum(x, 1e-9)  # Avoid division by zero
    
    # Calculate inflection point
    inflexion = x_range_min * (1 - gamma) + x_range_max * gamma
    inflexion = max(float(inflexion), 1e-9)
    
    # Derivative: (alpha × x^(alpha-1) × inflexion^alpha) / (x^alpha + inflexion^alpha)^2
    x_alpha = np.power(x, alpha)
    inflexion_alpha = np.power(inflexion, alpha)
    
    numerator = alpha * np.power(x, alpha - 1) * inflexion_alpha
    denominator = np.power(x_alpha + inflexion_alpha, 2)
    
    derivative = numerator / denominator
    
    return derivative

def calculate_metrics(y_true, y_pred):
    """Calculate model performance metrics"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) if mask.sum() > 0 else 0
    wmape = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) if np.sum(np.abs(y_true)) > 0 else 0
    
    return r2, mape, wmape

def add_seasonality_features(df, date_col, include_dow=True, include_month=True):
    """Add seasonality features: day of week and/or month"""
    df = df.copy()
    
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce', dayfirst=True)
    
    dummies_to_add = []
    
    if include_dow:
        df['day_of_week'] = df[date_col].dt.dayofweek
        day_dummies = pd.get_dummies(df['day_of_week'], prefix='dow', drop_first=True)
        dummies_to_add.append(day_dummies)
    
    if include_month:
        df['month'] = df[date_col].dt.month
        month_dummies = pd.get_dummies(df['month'], prefix='month', drop_first=True)
        dummies_to_add.append(month_dummies)
    
    if dummies_to_add:
        df_with_seasonality = pd.concat([df] + dummies_to_add, axis=1)
    else:
        df_with_seasonality = df
    
    return df_with_seasonality

def process_promotion_variable(df, promo_col):
    """Process promotion variable - convert to dummy if string, use as numeric if numeric"""
    df = df.copy()
    
    if df[promo_col].dtype == 'object' or df[promo_col].dtype.name == 'category':
        promo_dummies = pd.get_dummies(df[promo_col], prefix='promo', drop_first=True)
        df = pd.concat([df, promo_dummies], axis=1)
        feature_cols = promo_dummies.columns.tolist()
        is_dummy = True
    else:
        feature_cols = [promo_col]
        is_dummy = False
    
    return df, feature_cols, is_dummy

# NEW: DECOMP.RSSD calculation
def calculate_decomp_rssd(test_df, contributions, media_cols):
    """
    Calculate DECOMP.RSSD metric - measures spend vs effect share alignment
    Lower values indicate better alignment between spending and effectiveness
    """
    # Calculate total spend across all media
    total_spend = sum([test_df[col].sum() for col in media_cols if col in test_df.columns])
    
    # Calculate total effect (contribution) across media only
    media_contributions = {k: v for k, v in contributions.items() if k in media_cols}
    total_effect = sum(media_contributions.values())
    
    if total_spend == 0 or total_effect == 0:
        return 0, {}, {}
    
    # Calculate spend share for each channel
    spend_share = {}
    for col in media_cols:
        if col in test_df.columns:
            spend_share[col] = test_df[col].sum() / total_spend
    
    # Calculate effect share for each channel
    effect_share = {}
    for col in media_cols:
        effect_share[col] = media_contributions.get(col, 0) / total_effect
    
    # Calculate RSSD (Root Sum of Squared Differences)
    squared_diffs = []
    for col in media_cols:
        diff = effect_share.get(col, 0) - spend_share.get(col, 0)
        squared_diffs.append(diff ** 2)
    
    rssd = np.sqrt(sum(squared_diffs))
    
    return rssd, spend_share, effect_share

# Main app
st.markdown('<p class="main-header">📊 MMM Platform: Advanced Hill + No Standardization + Control Variables</p>', unsafe_allow_html=True)

# VERSION STAMP - VERIFY YOU'RE RUNNING THE RIGHT FILE
st.success("🔥 **VERSION: MEAN SCALING FIX v4.0 - 2026-03-08** 🔥")
st.info("✅ This version: Features scaled by MEAN spend (not max) to fix negative baseline")

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/analytics.png", width=100)
    st.markdown("### Navigation")
    tab_selection = st.radio(
        "Select a section:",
        ["📤 Data Upload", "🔍 Data Overview", "🎯 Marketing Mix Modeling", "📈 Results & Insights"],
        key="navigation"
    )
    
    st.markdown("---")
    st.markdown("### About")
    st.info("""
    **Enhanced Features:**
    - 🔬 **Gamma-based Hill saturation** (Reference Script method)
    - ❌ **No standardization** (original transformed space)
    - 📊 **Control variables upload** (weather, events, etc.)
    - 📈 DECOMP.RSSD metric
    - 🔍 VIF analysis
    - 📉 95% Confidence Intervals
    - 🎯 Budget optimization (scipy)
    """)

# TAB 1: Data Upload
if tab_selection == "📤 Data Upload":
    st.markdown('<p class="sub-header">Upload Your Marketing Data</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 KPI Data (Revenue)")
        st.info("Upload your store/Shopify revenue data. Must include: **Date** and **Revenue** columns")
        
        kpi_file = st.file_uploader(
            "Choose KPI CSV file",
            type=['csv'],
            key='kpi_upload',
            help="Upload CSV with Date and Revenue columns"
        )
        
        if kpi_file:
            try:
                kpi_df = pd.read_csv(kpi_file)
                kpi_df = clean_dataframe_numeric_columns(kpi_df, exclude_cols=[kpi_df.columns[0]])
                st.session_state.kpi_data = kpi_df
                
                st.success(f"✅ KPI data uploaded successfully! ({len(kpi_df)} rows)")
                
                with st.expander("Preview KPI Data"):
                    st.dataframe(kpi_df.head(10), width='stretch')
                    st.markdown("**Data Info:**")
                    st.write(f"- Columns: {', '.join(kpi_df.columns.tolist())}")
                    st.write(f"- Date range: {kpi_df.iloc[:, 0].min()} to {kpi_df.iloc[:, 0].max()}")
                    
            except Exception as e:
                st.error(f"Error loading KPI data: {str(e)}")
    
    with col2:
        st.markdown("#### 💰 Media Spend Data")
        st.info("Upload media channel data. Must include: **Date** and **Cost** columns")
        
        num_channels = st.number_input("Number of media channels", min_value=1, max_value=10, value=2, key='num_channels')
        
        for i in range(num_channels):
            st.markdown(f"**Channel {i+1}:**")
            channel_name = st.text_input(f"Channel name", value=f"Channel_{i+1}", key=f'channel_name_{i}')
            channel_file = st.file_uploader(
                f"Upload {channel_name} CSV",
                type=['csv'],
                key=f'channel_file_{i}'
            )
            
            if channel_file:
                try:
                    channel_df = pd.read_csv(channel_file)
                    channel_df = clean_dataframe_numeric_columns(channel_df, exclude_cols=[channel_df.columns[0]])
                    st.session_state.media_data[channel_name] = channel_df
                    st.success(f"✅ {channel_name} uploaded ({len(channel_df)} rows)")
                    
                    with st.expander(f"Preview {channel_name}"):
                        st.dataframe(channel_df.head(5), width='stretch')
                        
                except Exception as e:
                    st.error(f"Error loading {channel_name}: {str(e)}")
    
    # Promotion/Discount variable upload
    st.markdown("---")
    st.markdown("#### 🎁 Promotion/Discount Data (Optional)")
    st.info("""
    Upload promotion data with **Date** and **Promotion** columns.
    - **String values** (e.g., 'Yes'/'No') → Converted to dummy variables
    - **Numeric values** (e.g., 10%, 0.15) → Used as continuous variable
    """)
    
    promo_file = st.file_uploader(
        "Upload Promotion CSV (optional)",
        type=['csv'],
        key='promo_upload',
        help="CSV with Date and Promotion columns"
    )
    
    if promo_file:
        try:
            promo_df = pd.read_csv(promo_file)
            
            if len(promo_df.columns) > 2:
                for col in promo_df.columns[2:]:
                    promo_df[col] = clean_numeric_column(promo_df[col])
            
            st.session_state.promotion_data = promo_df
            
            st.success(f"✅ Promotion data uploaded! ({len(promo_df)} rows)")
            
            with st.expander("Preview Promotion Data"):
                st.dataframe(promo_df.head(10), width='stretch')
                
                promo_col = promo_df.columns[1]
                if promo_df[promo_col].dtype == 'object':
                    st.info(f"✓ Detected **categorical** promotion: {promo_df[promo_col].unique()[:5]}")
                else:
                    st.info(f"✓ Detected **numeric** promotion: Range {promo_df[promo_col].min():.2f} - {promo_df[promo_col].max():.2f}")
                    
        except Exception as e:
            st.error(f"Error loading promotion data: {str(e)}")
    
    # Control Variables upload (NEW)
    st.markdown("---")
    st.markdown("#### 📊 Control Variables (Optional)")
    st.info("""
    Upload additional control variables (weather, competitor activity, events, economics, etc.)
    - **Date** column required in each file
    - **String values** → Converted to dummy variables
    - **Numeric values** → Used as continuous variables
    - Examples: Temperature, GDP, Holidays, Competitor_Spend
    """)
    
    num_control_vars = st.number_input(
        "Number of control variable files",
        min_value=0,
        max_value=10,
        value=0,
        key='num_control_vars',
        help="Upload separate CSV files for different control variables"
    )
    
    if 'control_data' not in st.session_state:
        st.session_state.control_data = {}
    
    for i in range(num_control_vars):
        st.markdown(f"**Control Variable {i+1}:**")
        
        control_name = st.text_input(
            f"Control variable name",
            value=f"Control_{i+1}",
            key=f'control_name_{i}',
            help="E.g., Weather, Events, Competitor_Activity"
        )
        
        control_file = st.file_uploader(
            f"Upload {control_name} CSV",
            type=['csv'],
            key=f'control_file_{i}',
            help="CSV with Date and control variable columns"
        )
        
        if control_file:
            try:
                control_df = pd.read_csv(control_file)
                
                # Clean numeric columns (except date and potential categorical columns)
                for col in control_df.columns[1:]:
                    if control_df[col].dtype not in ['object', 'category']:
                        control_df[col] = clean_numeric_column(control_df[col])
                
                st.session_state.control_data[control_name] = control_df
                
                st.success(f"✅ {control_name} uploaded! ({len(control_df)} rows)")
                
                with st.expander(f"Preview {control_name}"):
                    st.dataframe(control_df.head(5), width='stretch')
                    
                    # Show data types
                    st.markdown("**Detected variable types:**")
                    for col in control_df.columns[1:]:
                        if control_df[col].dtype == 'object':
                            st.write(f"- {col}: Categorical → Will create dummies")
                        else:
                            st.write(f"- {col}: Numeric → Will use as continuous")
                            
            except Exception as e:
                st.error(f"Error loading {control_name}: {str(e)}")
    
    # Combine data button
    st.markdown("---")
    if st.button("🔗 Combine All Data", type="primary", width='stretch'):
        if st.session_state.kpi_data is None:
            st.error("❌ Please upload KPI data first!")
        elif len(st.session_state.media_data) == 0:
            st.error("❌ Please upload at least one media channel!")
        else:
            with st.spinner("Combining data..."):
                try:
                    combined = st.session_state.kpi_data.copy()
                    date_col = combined.columns[0]
                    combined[date_col] = pd.to_datetime(combined[date_col], errors='coerce', dayfirst=True)
                    
                    for channel_name, channel_df in st.session_state.media_data.items():
                        channel_df = channel_df.copy()
                        channel_date_col = channel_df.columns[0]
                        channel_df[channel_date_col] = pd.to_datetime(channel_df[channel_date_col], errors='coerce', dayfirst=True)
                        
                        rename_dict = {}
                        for col in channel_df.columns:
                            if col.lower() not in ['date']:
                                rename_dict[col] = f"{channel_name}_{col}"
                        channel_df = channel_df.rename(columns=rename_dict)
                        channel_df = channel_df.rename(columns={channel_date_col: date_col})
                        combined = combined.merge(channel_df, on=date_col, how='left')
                    
                    if st.session_state.promotion_data is not None:
                        promo_df = st.session_state.promotion_data.copy()
                        promo_date_col = promo_df.columns[0]
                        promo_df[promo_date_col] = pd.to_datetime(promo_df[promo_date_col], errors='coerce', dayfirst=True)
                        promo_df = promo_df.rename(columns={promo_date_col: date_col})
                        combined = combined.merge(promo_df, on=date_col, how='left')
                        
                        promo_col = promo_df.columns[1]
                        if combined[promo_col].dtype == 'object':
                            combined[promo_col] = combined[promo_col].fillna('None')
                        else:
                            combined[promo_col] = combined[promo_col].fillna(0)
                    
                    # Merge control variables if available (NEW)
                    if hasattr(st.session_state, 'control_data') and st.session_state.control_data:
                        for control_name, control_df in st.session_state.control_data.items():
                            control_df = control_df.copy()
                            control_date_col = control_df.columns[0]
                            
                            # Parse dates
                            control_df[control_date_col] = pd.to_datetime(control_df[control_date_col], errors='coerce', dayfirst=True)
                            control_df = control_df.rename(columns={control_date_col: date_col})
                            
                            # Merge
                            combined = combined.merge(control_df, on=date_col, how='left')
                            
                            # Fill missing values
                            for col in control_df.columns[1:]:
                                if combined[col].dtype == 'object':
                                    combined[col] = combined[col].fillna('None')
                                else:
                                    combined[col] = combined[col].fillna(0)
                    
                    cost_cols = [col for col in combined.columns if 'cost' in col.lower() or 'spend' in col.lower()]
                    combined[cost_cols] = combined[cost_cols].fillna(0)
                    combined = clean_dataframe_numeric_columns(combined, exclude_cols=[date_col])
                    combined = combined.dropna(subset=[date_col])
                    
                    st.session_state.combined_data = combined
                    st.session_state.data_uploaded = True
                    
                    st.success("✅ Data combined successfully!")
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"Error combining data: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

# TAB 2: Data Overview
elif tab_selection == "🔍 Data Overview":
    st.markdown('<p class="sub-header">Data Overview & Validation</p>', unsafe_allow_html=True)
    
    if not st.session_state.data_uploaded:
        st.warning("⚠️ Please upload and combine data first in the 'Data Upload' tab!")
    else:
        df = st.session_state.combined_data.copy()
        date_col = df.columns[0]
        
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce', dayfirst=True)
            st.session_state.combined_data[date_col] = df[date_col]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Records", len(df))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            date_range_days = (df[date_col].max() - df[date_col].min()).days
            st.metric("Date Range (Days)", date_range_days)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            date_range_months = date_range_days / 30
            st.metric("Months of Data", f"{date_range_months:.1f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            media_channels = len([col for col in df.columns if 'cost' in col.lower() or 'spend' in col.lower()])
            st.metric("Media Channels", media_channels)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### ✅ Data Validation")
        
        validation_col1, validation_col2, validation_col3 = st.columns(3)
        
        with validation_col1:
            if date_range_months >= 24:
                st.success(f"✅ Sufficient data: {date_range_months:.1f} months")
            else:
                st.warning(f"⚠️ Limited data: {date_range_months:.1f} months (<24 recommended)")
        
        with validation_col2:
            has_revenue = any('revenue' in col.lower() for col in df.columns)
            if has_revenue:
                st.success("✅ Revenue column found")
            else:
                st.error("❌ Revenue column not found")
        
        with validation_col3:
            has_promo = any('promo' in col.lower() or 'discount' in col.lower() for col in df.columns)
            if has_promo:
                st.success("✅ Promotion data included")
            else:
                st.info("ℹ️ No promotion data")
        
        st.markdown("---")
        st.markdown("### 📊 Combined Dataset")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col != date_col]
        
        if numeric_cols:
            st.dataframe(
                df.style.background_gradient(subset=numeric_cols, cmap='Blues'),
                width='stretch',
                height=400
            )
        else:
            st.dataframe(df, width='stretch', height=400)
        
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Combined Data",
            data=csv,
            file_name=f"combined_mmm_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
        st.markdown("---")
        st.markdown("### 📈 Descriptive Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Numerical Summary:**")
            st.dataframe(df.describe(), width='stretch')
        
        with col2:
            st.markdown("**Missing Values:**")
            missing_df = pd.DataFrame({
                'Column': df.columns,
                'Missing Count': df.isnull().sum().values,
                'Missing %': (df.isnull().sum().values / len(df) * 100).round(2)
            })
            st.dataframe(missing_df, width='stretch')
        
        st.markdown("---")
        st.markdown("### 🔥 Correlation Heatmap")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 1:
            fig, ax = plt.subplots(figsize=(12, 8))
            correlation_matrix = df[numeric_cols].corr()
            sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
            plt.title('Correlation Matrix', fontsize=16, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)

# TAB 3: Marketing Mix Modeling
elif tab_selection == "🎯 Marketing Mix Modeling":
    st.markdown('<p class="sub-header">Marketing Mix Modeling</p>', unsafe_allow_html=True)
    
    if not st.session_state.data_uploaded:
        st.warning("⚠️ Please upload and combine data first in the 'Data Upload' tab!")
        st.info("👉 Go to the 'Data Upload' tab to upload your media and KPI data.")
    else:
        df = st.session_state.combined_data.copy()
        date_col = df.columns[0]
        
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce', dayfirst=True)
        
        date_range_months = (df[date_col].max() - df[date_col].min()).days / 30
        
        if date_range_months < 24:
            st.warning(f"⚠️ Limited data: {date_range_months:.1f} months available (24 months recommended)")
        else:
            st.success(f"✅ Data validation passed: {date_range_months:.1f} months available")
        
        st.markdown("---")
        st.markdown("### 🔧 Configure Model Variables")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Target Variable (KPI):**")
            potential_target_cols = [col for col in df.columns if 'revenue' in col.lower() or 'sales' in col.lower()]
            target_col = st.selectbox(
                "Select target/KPI column",
                potential_target_cols if potential_target_cols else df.columns[1:],
                key='target_col_selector'
            )
        
        with col2:
            st.markdown("**Date Column:**")
            date_col_confirm = st.selectbox(
                "Confirm date column",
                [date_col],
                key='date_col_confirm'
            )
        
        st.markdown("**Media Spend Columns:**")
        cost_cols = [col for col in df.columns if 'cost' in col.lower() or 'spend' in col.lower()]
        media_cols = st.multiselect(
            "Select media spend columns",
            cost_cols if cost_cols else df.columns[1:],
            default=cost_cols if cost_cols else [],
            key='media_cols_selector'
        )
        
        if not media_cols:
            st.warning("⚠️ Please select at least one media spend column!")
            st.stop()
        
        st.markdown("**Promotion/Discount Variable (Optional):**")
        promo_options = [col for col in df.columns if ('promo' in col.lower() or 'discount' in col.lower()) 
                         and col not in media_cols and col != target_col and col != date_col]
        
        promo_col = None
        if promo_options:
            use_promo = st.checkbox("Include promotion variable", value=True)
            if use_promo:
                promo_col = st.selectbox("Select promotion column", promo_options, key='promo_col_selector')
        
        st.markdown("**Other Control Variables (Optional):**")
        available_controls = [col for col in df.columns if col not in media_cols and col != target_col 
                             and col != date_col and col != promo_col 
                             and not ('promo' in col.lower() or 'discount' in col.lower())]
        control_cols = st.multiselect(
            "Select additional control variables",
            available_controls,
            key='other_controls_selector'
        )
        
        st.markdown("---")
        st.markdown("### 📅 Seasonality Controls")
        
        season_col1, season_col2, season_col3 = st.columns(3)
        
        with season_col1:
            include_dow = st.checkbox(
                "Include Day of Week", 
                value=True,
                help="Add day-of-week dummy variables (Monday-Sunday)"
            )
        
        with season_col2:
            include_month = st.checkbox(
                "Include Month of Year", 
                value=True,
                help="Add month-of-year dummy variables (Jan-Dec)"
            )
        
        with season_col3:
            if not include_dow and not include_month:
                st.warning("⚠️ No seasonality selected")
            elif include_dow and include_month:
                st.success("✅ Full seasonality (DOW + Month)")
            elif include_dow:
                st.info("📊 Day-of-week only")
            else:
                st.info("📊 Month only")
        
        st.markdown("---")
        st.markdown("### ⚙️ Global Model Parameters")
        
        st.info("💡 **Tip:** Set global defaults here, or use per-channel controls below for fine-tuned parameters")
        
        param_col1, param_col2, param_col3 = st.columns(3)
        
        with param_col1:
            global_adstock = st.slider(
                "Global Adstock Rate (θ)", 
                0.0, 0.9, 0.5, 0.05, 
                help="Default carryover effect (theta) - can override per channel"
            )
        
        with param_col2:
            global_hill_alpha = st.slider(
                "Global Hill Alpha (α)", 
                0.5, 3.0, 1.0, 0.1, 
                help="Default saturation steepness - can override per channel"
            )
        
        with param_col3:
            global_hill_gamma = st.slider(
                "Global Hill Gamma (γ)", 
                0.0, 1.0, 0.5, 0.05, 
                help="Default inflection point - can override per channel"
            )
        
        st.markdown("---")
        st.markdown("### 🎯 Per-Channel Parameter Controls")
        
        use_per_channel = st.checkbox(
            "Enable Per-Channel Parameter Customization", 
            value=False,
            help="Override global parameters for individual channels"
        )
        
        # Store per-channel parameters
        channel_params = {}
        
        if use_per_channel:
            st.info("""
            **Per-Channel Parameters:** Customize adstock and saturation for each media channel.
            
            **Suggested ranges by channel type:**
            - **TV/Video:** θ: 0.3-0.8, α: 0.5-3.0, γ: 0.3-1.0 (high carryover)
            - **Digital (FB/Google):** θ: 0.0-0.3, α: 0.5-3.0, γ: 0.3-1.0 (low carryover)
            - **Traditional (Print/Radio):** θ: 0.1-0.5, α: 0.5-3.0, γ: 0.3-1.0 (medium carryover)
            """)
            
            for media_col in media_cols:
                with st.expander(f"📊 {media_col}"):
                    ch_col1, ch_col2, ch_col3 = st.columns(3)
                    
                    with ch_col1:
                        ch_adstock = st.slider(
                            f"Adstock (θ)",
                            0.0, 0.9, global_adstock, 0.05,
                            key=f"adstock_{media_col}",
                            help=f"Carryover effect for {media_col}"
                        )
                    
                    with ch_col2:
                        ch_alpha = st.slider(
                            f"Alpha (α)",
                            0.5, 3.0, global_hill_alpha, 0.1,
                            key=f"alpha_{media_col}",
                            help=f"Saturation steepness for {media_col}"
                        )
                    
                    with ch_col3:
                        ch_gamma = st.slider(
                            f"Gamma (γ)",
                            0.0, 1.0, global_hill_gamma, 0.05,
                            key=f"gamma_{media_col}",
                            help=f"Inflection point for {media_col}"
                        )
                    
                    channel_params[media_col] = {
                        'adstock': ch_adstock,
                        'alpha': ch_alpha,
                        'gamma': ch_gamma
                    }
        else:
            # Use global parameters for all channels
            for media_col in media_cols:
                channel_params[media_col] = {
                    'adstock': global_adstock,
                    'alpha': global_hill_alpha,
                    'gamma': global_hill_gamma
                }
        
        st.markdown("---")
        train_split_col, _ = st.columns([1, 2])
        with train_split_col:
            train_test_split = st.slider(
                "Train/Test Split", 
                0.6, 0.9, 0.8, 0.05, 
                help="Proportion of data for training"
            )
        
        st.markdown("---")
        if st.button("🚀 Run Marketing Mix Model", type="primary", width='stretch'):
            with st.spinner("Training Marketing Mix Model..."):
                try:
                    st.info("Step 1/7: Preparing daily data...")
                    daily_df = df.copy()
                    daily_df = daily_df.sort_values(date_col).reset_index(drop=True)
                    
                    for col in media_cols:
                        daily_df[col] = pd.to_numeric(daily_df[col], errors='coerce').fillna(0)
                    
                    daily_df[target_col] = pd.to_numeric(daily_df[target_col], errors='coerce')
                    daily_df = daily_df.dropna(subset=[target_col])
                    
                    st.info("Step 2/7: Adding seasonality features...")
                    daily_df = add_seasonality_features(daily_df, date_col, include_dow, include_month)
                    
                    promo_features = []
                    promo_is_dummy = False
                    if promo_col:
                        st.info(f"Step 3/7: Processing promotion variable...")
                        daily_df, promo_features, promo_is_dummy = process_promotion_variable(daily_df, promo_col)
                        st.session_state.promo_is_dummy = promo_is_dummy
                        st.session_state.promo_features = promo_features
                    else:
                        st.info("Step 3/7: No promotion variable selected...")
                        st.session_state.promo_features = []
                        st.session_state.promo_is_dummy = False
                    
                    # Process control variables (NEW)
                    control_features = []
                    original_control_cols = []  # Track original columns to drop
                    
                    if hasattr(st.session_state, 'control_data') and st.session_state.control_data:
                        st.info(f"Step 3b/7: Processing {len(st.session_state.control_data)} control variable(s)...")
                        
                        # Control variables are already merged, identify columns and process
                        for control_name in st.session_state.control_data.keys():
                            control_vars = st.session_state.control_data[control_name]
                            for col in control_vars.columns[1:]:  # Skip date column
                                if col in daily_df.columns:
                                    original_control_cols.append(col)  # Track for removal
                                    
                                    if daily_df[col].dtype == 'object' or daily_df[col].dtype.name == 'category':
                                        # Categorical - convert to dummies
                                        ctrl_dummies = pd.get_dummies(daily_df[col], prefix=f'{col}', drop_first=True)
                                        daily_df = pd.concat([daily_df, ctrl_dummies], axis=1)
                                        control_features.extend(ctrl_dummies.columns.tolist())
                                    else:
                                        # Numeric - use as is
                                        control_features.append(col)
                        
                        # Drop original control variable columns to avoid duplicates
                        daily_df = daily_df.drop(columns=original_control_cols, errors='ignore')
                        
                        st.session_state.control_features = control_features
                    else:
                        st.info("Step 3b/7: No control variables uploaded...")
                        st.session_state.control_features = []
                        control_features = []
                    
                    # Engineer media features (NO STANDARDIZATION)
                    st.info("Step 4/7: Engineering media features (adstock + Hill saturation)...")
                    
                    meta = {}
                    feat_cols = []
                    
                    for media_col in media_cols:
                        # Get channel-specific parameters
                        ch_adstock = channel_params[media_col]['adstock']
                        ch_alpha = channel_params[media_col]['alpha']
                        ch_gamma = channel_params[media_col]['gamma']
                        
                        # Adstock transformation with channel-specific parameter
                        daily_df[f'{media_col}_adstock'] = adstock_transformation(
                            daily_df[media_col].values, alpha=ch_adstock
                        )
                        
                        # Hill saturation with channel-specific parameters
                        saturated_raw = hill_transformation(
                            daily_df[f'{media_col}_adstock'].values,
                            alpha=ch_alpha,
                            gamma=ch_gamma
                        )
                        
                        # SCALE saturated values by MEAN spend to prevent negative baseline
                        # This makes feature means proportional to actual spending patterns
                        # Saturated (0-1) × mean_spend → preserves spend relationships
                        mean_spend = daily_df[media_col].mean()
                        
                        # DEBUG: Show scaling info
                        st.write(f"🔍 DEBUG {media_col}:")
                        st.write(f"  - Raw saturated range: [{saturated_raw.min():.4f}, {saturated_raw.max():.4f}]")
                        st.write(f"  - Mean spend: ${mean_spend:,.2f}")
                        
                        if mean_spend > 0:
                            daily_df[f'{media_col}_saturated'] = saturated_raw * mean_spend
                            st.write(f"  - Scaled range: [${daily_df[f'{media_col}_saturated'].min():,.2f}, ${daily_df[f'{media_col}_saturated'].max():,.2f}]")
                            st.write(f"  - Scaled mean: ${daily_df[f'{media_col}_saturated'].mean():,.2f}")
                            st.success(f"  ✅ Scaling applied: multiplied by ${mean_spend:,.2f} (mean spend)")
                        else:
                            daily_df[f'{media_col}_saturated'] = saturated_raw
                            st.warning(f"  ⚠️ No scaling (mean_spend = 0)")
                        
                        # Store scaling factor for later use
                        feat_name = f'{media_col}_saturated'
                        feat_cols.append(feat_name)
                        
                        # Store metadata
                        meta[feat_name] = {
                            'spend_col': media_col,
                            'alpha': ch_alpha,
                            'gamma': ch_gamma,
                            'adstock_theta': ch_adstock,
                            'scaling_factor': mean_spend,  # Store mean spend for contribution calculation
                            # Store min/max for derivative calculation
                            'x_min': float(daily_df[f'{media_col}_adstock'].min()),
                            'x_max': float(daily_df[f'{media_col}_adstock'].max())
                        }
                    
                    # SANITY CHECK: Verify features are scaled properly
                    st.markdown("---")
                    st.markdown("### 🔍 SANITY CHECK: Feature Scaling Verification")
                    st.info("Checking if features are properly scaled by mean spend...")
                    
                    sanity_check_passed = True
                    for feat in feat_cols:
                        feat_min = daily_df[feat].min()
                        feat_max = daily_df[feat].max()
                        feat_mean = daily_df[feat].mean()
                        
                        st.write(f"**{feat}:**")
                        st.write(f"  - Min: {feat_min:,.2f}, Max: {feat_max:,.2f}, Mean: {feat_mean:,.2f}")
                        
                        # Check if in 0-1 range (BAD) or dollar range (GOOD)
                        if feat_max <= 1.1:  # Some tolerance
                            st.error(f"  ❌ PROBLEM: Feature in 0-1 range! Scaling NOT applied!")
                            sanity_check_passed = False
                        else:
                            st.success(f"  ✅ GOOD: Feature scaled to dollar range (mean spend basis)")
                    
                    if not sanity_check_passed:
                        st.error("🚨 CRITICAL ERROR: Features are NOT scaled! Check code!")
                        st.stop()
                    else:
                        st.success("✅ All features properly scaled by mean spend!")
                    
                    st.markdown("---")
                    
                    st.info("Step 5/7: Splitting data...")
                    split_idx = int(len(daily_df) * train_test_split)
                    train_df = daily_df.iloc[:split_idx].copy()
                    test_df = daily_df.iloc[split_idx:].copy()
                    
                    seasonality_cols = [col for col in daily_df.columns if 'dow_' in col or 'month_' in col]
                    
                    # Combine promo + control + other control cols
                    # Deduplicate while preserving order
                    all_control_cols_raw = control_cols + promo_features + control_features
                    all_control_cols = []
                    seen = set()
                    for col in all_control_cols_raw:
                        if col not in seen and col in daily_df.columns:
                            all_control_cols.append(col)
                            seen.add(col)
                    
                    for ctrl_col in all_control_cols:
                        if ctrl_col in train_df.columns:
                            if train_df[ctrl_col].dtype == 'object':
                                try:
                                    train_df[ctrl_col] = pd.to_numeric(train_df[ctrl_col], errors='coerce').fillna(0)
                                    test_df[ctrl_col] = pd.to_numeric(test_df[ctrl_col], errors='coerce').fillna(0)
                                except:
                                    pass
                    
                    X_train = pd.concat([
                        pd.Series(1.0, index=train_df.index, name='const'),
                        train_df[feat_cols],
                        train_df[all_control_cols] if all_control_cols else pd.DataFrame(index=train_df.index),
                        train_df[seasonality_cols]
                    ], axis=1).astype('float64')
                    
                    X_test = pd.concat([
                        pd.Series(1.0, index=test_df.index, name='const'),
                        test_df[feat_cols],
                        test_df[all_control_cols] if all_control_cols else pd.DataFrame(index=test_df.index),
                        test_df[seasonality_cols]
                    ], axis=1).astype('float64')
                    
                    # Check for and remove duplicate columns
                    if X_train.columns.duplicated().any():
                        duplicate_cols = X_train.columns[X_train.columns.duplicated()].tolist()
                        st.warning(f"⚠️ Removing {len(duplicate_cols)} duplicate column(s): {', '.join(duplicate_cols)}")
                        X_train = X_train.loc[:, ~X_train.columns.duplicated()]
                        X_test = X_test.loc[:, ~X_test.columns.duplicated()]
                    
                    y_train = train_df[target_col].values.astype(float)
                    y_test = test_df[target_col].values.astype(float)
                    
                    st.info("Step 6/7: Training OLS model...")
                    model = sm.OLS(y_train, X_train).fit()
                    
                    # DEBUG: Show coefficients for media features
                    st.write("🔍 DEBUG - Media Feature Coefficients:")
                    for feat in feat_cols:
                        if feat in model.params.index:
                            coef = float(model.params.at[feat])
                            st.write(f"  - {feat}: β = {coef:,.4f}")
                    
                    # Show baseline too
                    if 'const' in model.params.index:
                        baseline_coef = float(model.params.at['const'])
                        st.write(f"  - Baseline (const): β₀ = {baseline_coef:,.2f}")
                    
                    y_train_pred = model.predict(X_train)
                    y_test_pred = model.predict(X_test)
                    
                    st.info("Step 7/7: Calculating metrics...")
                    train_r2, train_mape, train_wmape = calculate_metrics(y_train, y_train_pred)
                    test_r2, test_mape, test_wmape = calculate_metrics(y_test, y_test_pred)
                    
                    st.session_state.model_trained = True
                    st.session_state.model = model
                    st.session_state.meta = meta
                    st.session_state.feat_cols = feat_cols
                    st.session_state.media_cols = media_cols
                    st.session_state.target_col = target_col
                    st.session_state.date_col = date_col
                    st.session_state.train_df = train_df
                    st.session_state.test_df = test_df
                    st.session_state.X_train = X_train
                    st.session_state.X_test = X_test
                    st.session_state.y_train = y_train
                    st.session_state.y_test = y_test
                    st.session_state.y_train_pred = y_train_pred
                    st.session_state.y_test_pred = y_test_pred
                    st.session_state.channel_params = channel_params  # NEW: per-channel parameters
                    st.session_state.include_dow = include_dow  # NEW: seasonality controls
                    st.session_state.include_month = include_month  # NEW: seasonality controls
                    st.session_state.promo_col = promo_col
                    st.session_state.control_cols = control_cols
                    
                    st.success("✅ Model trained successfully!")
                    st.balloons()
                    
                    st.markdown("---")
                    st.markdown("### 📊 Model Performance")
                    
                    metric_col1, metric_col2 = st.columns(2)
                    
                    with metric_col1:
                        st.markdown("**Training Set:**")
                        st.metric("R²", f"{train_r2:.3f}")
                        st.metric("MAPE", f"{train_mape:.2%}")
                        st.metric("wMAPE", f"{train_wmape:.2%}")
                    
                    with metric_col2:
                        st.markdown("**Test Set:**")
                        st.metric("R²", f"{test_r2:.3f}")
                        st.metric("MAPE", f"{test_mape:.2%}")
                        st.metric("wMAPE", f"{test_wmape:.2%}")
                    
                    st.markdown("---")
                    st.markdown("### 📈 Model Fit Visualization")
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                    
                    ax1.plot(train_df[date_col], y_train, label='Actual', color='green', alpha=0.7)
                    ax1.plot(train_df[date_col], y_train_pred, label='Predicted', color='blue', alpha=0.7)
                    ax1.set_title(f'Training Set (R²={train_r2:.3f})', fontsize=14, fontweight='bold')
                    ax1.set_xlabel('Date')
                    ax1.set_ylabel(target_col)
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
                    
                    ax2.plot(test_df[date_col], y_test, label='Actual', color='green', alpha=0.7)
                    ax2.plot(test_df[date_col], y_test_pred, label='Predicted', color='blue', alpha=0.7)
                    ax2.set_title(f'Test Set (R²={test_r2:.3f})', fontsize=14, fontweight='bold')
                    ax2.set_xlabel('Date')
                    ax2.set_ylabel(target_col)
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    st.info("✅ Model complete! Go to 'Results & Insights' →")
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

# TAB 4: Results & Insights
elif tab_selection == "📈 Results & Insights":
    st.markdown('<p class="sub-header">Results & Insights</p>', unsafe_allow_html=True)
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train the model first!")
        st.info("👉 Go to 'Marketing Mix Modeling' tab")
    else:
        model = st.session_state.model
        meta = st.session_state.meta
        feat_cols = st.session_state.feat_cols
        media_cols = st.session_state.media_cols
        target_col = st.session_state.target_col
        date_col = st.session_state.date_col
        test_df = st.session_state.test_df
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test
        y_test_pred = st.session_state.y_test_pred
        channel_params = st.session_state.channel_params  # NEW: per-channel parameters
        include_dow = st.session_state.include_dow  # NEW: seasonality controls
        include_month = st.session_state.include_month  # NEW: seasonality controls
        promo_col = st.session_state.promo_col
        promo_features = st.session_state.promo_features
        control_cols = st.session_state.control_cols
        
        result_tabs = st.tabs([
            "📊 Revenue Decomposition",
            "💰 ROI Analysis + DECOMP.RSSD",
            "📈 Response Curves",
            "🎯 Budget Optimization",
            "📋 Model Summary + VIF"
        ])
        
        # Tab 1: Channel Contribution
        with result_tabs[0]:
            st.markdown("### Complete Revenue Decomposition")
            
            st.info("""
            This shows how each component contributes to total revenue:
            - **Media Channels**: Paid advertising effect
            - **Promotions**: Campaign/discount effects
            - **Control Variables**: External factors (weather, events, inflation, etc.)
            - **Seasonality**: Day-of-week and monthly patterns
            - **Baseline**: Base revenue without any effects
            """)
            
            contributions = {}
            
            # 1. Media channel contributions
            for feat in feat_cols:
                if feat in model.params.index:
                    try:
                        beta = float(model.params.at[feat])
                    except:
                        beta_val = model.params.loc[feat]
                        beta = float(beta_val.iloc[0] if hasattr(beta_val, 'iloc') else beta_val)
                    contrib = np.sum(X_test[feat].values * beta)
                    channel_name = meta[feat]['spend_col']
                    contributions[channel_name] = contrib
            
            # 2. Promotion contributions
            if promo_col and promo_features:
                promo_contrib = 0
                for promo_feat in promo_features:
                    if promo_feat in X_test.columns and promo_feat in model.params.index:
                        try:
                            beta = float(model.params.at[promo_feat])
                        except:
                            beta_val = model.params.loc[promo_feat]
                            beta = float(beta_val.iloc[0] if hasattr(beta_val, 'iloc') else beta_val)
                        promo_contrib += np.sum(X_test[promo_feat].values * beta)
                if promo_contrib != 0:
                    contributions['Promotions'] = promo_contrib
            
            # 3. Control variable contributions (NEW)
            if hasattr(st.session_state, 'control_features') and st.session_state.control_features:
                control_contrib = 0
                for ctrl_feat in st.session_state.control_features:
                    if ctrl_feat in X_test.columns and ctrl_feat in model.params.index:
                        try:
                            # Use .at[] for guaranteed scalar access
                            beta = float(model.params.at[ctrl_feat])
                        except:
                            # Fallback: use .loc[] and extract first element
                            beta_val = model.params.loc[ctrl_feat]
                            beta = float(beta_val.iloc[0] if hasattr(beta_val, 'iloc') else beta_val)
                        control_contrib += np.sum(X_test[ctrl_feat].values * beta)
                if control_contrib != 0:
                    contributions['Control Variables'] = control_contrib
            
            # 4. Other control columns (original controls selected by user)
            if control_cols:
                other_control_contrib = 0
                for ctrl_col in control_cols:
                    if ctrl_col in X_test.columns and ctrl_col in model.params.index:
                        try:
                            beta = float(model.params.at[ctrl_col])
                        except:
                            beta_val = model.params.loc[ctrl_col]
                            beta = float(beta_val.iloc[0] if hasattr(beta_val, 'iloc') else beta_val)
                        other_control_contrib += np.sum(X_test[ctrl_col].values * beta)
                if other_control_contrib != 0:
                    contributions['Other Controls'] = other_control_contrib
            
            # 5. Seasonality contributions (NEW)
            seasonality_cols = [col for col in X_test.columns if 'dow_' in col or 'month_' in col]
            if seasonality_cols:
                seasonality_contrib = 0
                for seas_col in seasonality_cols:
                    if seas_col in model.params.index:
                        try:
                            beta = float(model.params.at[seas_col])
                        except:
                            beta_val = model.params.loc[seas_col]
                            beta = float(beta_val.iloc[0] if hasattr(beta_val, 'iloc') else beta_val)
                        seasonality_contrib += np.sum(X_test[seas_col].values * beta)
                if seasonality_contrib != 0:
                    contributions['Seasonality'] = seasonality_contrib
            
            # 6. Baseline
            if 'const' in model.params.index:
                try:
                    baseline = float(model.params.at['const']) * len(X_test)
                except:
                    beta_val = model.params.loc['const']
                    baseline = float(beta_val.iloc[0] if hasattr(beta_val, 'iloc') else beta_val) * len(X_test)
                contributions['Baseline'] = baseline
            
            # 7. Calculate residual (unexplained variance)
            total_predicted = sum(contributions.values())
            total_actual = np.sum(y_test)
            residual = total_actual - total_predicted
            if abs(residual) > 1:  # Only show if meaningful
                contributions['Residual'] = residual
            
            # Create contribution dataframe
            contrib_df = pd.DataFrame.from_dict(contributions, orient='index', columns=['Contribution'])
            contrib_df['Contribution %'] = 100 * contrib_df['Contribution'] / total_actual
            contrib_df = contrib_df.sort_values('Contribution', ascending=False)
            
            # Summary metrics
            st.markdown("---")
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            
            with metric_col1:
                st.metric("Total Actual Revenue", f"${total_actual:,.0f}")
            with metric_col2:
                st.metric("Total Predicted", f"${total_predicted:,.0f}")
            with metric_col3:
                media_contrib = sum([v for k, v in contributions.items() if k not in ['Baseline', 'Seasonality', 'Promotions', 'Control Variables', 'Other Controls', 'Residual']])
                st.metric("Media Contribution", f"${media_contrib:,.0f}")
            with metric_col4:
                explained = (1 - abs(residual) / total_actual) * 100
                st.metric("Explained Variance", f"{explained:.1f}%")
            
            st.markdown("---")
            
            # Display tables and charts
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("**Complete Contribution Breakdown:**")
                st.dataframe(
                    contrib_df.style.format({
                        'Contribution': '{:,.0f}', 
                        'Contribution %': '{:.1f}%'
                    }).background_gradient(subset=['Contribution'], cmap='RdYlGn'),
                    width='stretch'
                )
                
                # Download button
                csv_contrib = contrib_df.to_csv()
                st.download_button(
                    label="📥 Download Contribution Data",
                    data=csv_contrib,
                    file_name=f"contribution_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    key="download_contrib"
                )
            
            with col2:
                # Pie chart - only positive contributions
                positive_contrib = contrib_df[contrib_df['Contribution'] > 0].copy()
                
                if len(positive_contrib) > 0:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    colors = plt.cm.Set3(range(len(positive_contrib)))
                    ax.pie(positive_contrib['Contribution'], labels=positive_contrib.index, autopct='%1.1f%%',
                           colors=colors, startangle=90)
                    ax.set_title('Revenue Contribution by Channel', fontsize=14, fontweight='bold')
                    st.pyplot(fig)
                    
                    if len(contrib_df) > len(positive_contrib):
                        st.caption(f"⚠️ {len(contrib_df) - len(positive_contrib)} channel(s) with negative contribution excluded")
                else:
                    st.warning("No positive contributions")
            
            st.markdown("---")
            
            # Horizontal bar chart
            fig, ax = plt.subplots(figsize=(12, 6))
            contrib_df['Contribution'].plot(kind='barh', ax=ax, color='steelblue')
            ax.set_xlabel('Revenue Contribution ($)', fontsize=12)
            ax.set_title('All Components Contribution', fontsize=14, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            ax.axvline(x=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
            
            for i, v in enumerate(contrib_df['Contribution']):
                ax.text(v, i, f' ${v:,.0f}', va='center', fontsize=10)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Waterfall chart showing revenue build-up
            st.markdown("---")
            st.markdown("### 📊 Revenue Waterfall Analysis")
            st.info("This shows how each component builds up to total revenue")
            
            # Prepare waterfall data (order: Baseline → Media → Promo → Controls → Seasonality → Residual)
            waterfall_order = []
            waterfall_values = []
            
            # Start with baseline
            if 'Baseline' in contributions:
                waterfall_order.append('Baseline')
                waterfall_values.append(contributions['Baseline'])
            
            # Add media channels
            for ch in contrib_df.index:
                if ch not in ['Baseline', 'Promotions', 'Control Variables', 'Other Controls', 'Seasonality', 'Residual']:
                    waterfall_order.append(ch)
                    waterfall_values.append(contributions[ch])
            
            # Add promotions
            if 'Promotions' in contributions:
                waterfall_order.append('Promotions')
                waterfall_values.append(contributions['Promotions'])
            
            # Add control variables
            if 'Control Variables' in contributions:
                waterfall_order.append('Control Variables')
                waterfall_values.append(contributions['Control Variables'])
            
            if 'Other Controls' in contributions:
                waterfall_order.append('Other Controls')
                waterfall_values.append(contributions['Other Controls'])
            
            # Add seasonality
            if 'Seasonality' in contributions:
                waterfall_order.append('Seasonality')
                waterfall_values.append(contributions['Seasonality'])
            
            # Add residual
            if 'Residual' in contributions:
                waterfall_order.append('Residual')
                waterfall_values.append(contributions['Residual'])
            
            # Create waterfall chart
            fig, ax = plt.subplots(figsize=(14, 7))
            
            # Calculate cumulative values for waterfall
            cumulative = [0]
            for val in waterfall_values:
                cumulative.append(cumulative[-1] + val)
            
            # Plot bars
            colors = []
            for i, (label, val) in enumerate(zip(waterfall_order, waterfall_values)):
                # Color coding
                if label == 'Baseline':
                    color = 'gray'
                elif label == 'Residual':
                    color = 'orange' if val < 0 else 'lightgreen'
                elif label in ['Seasonality', 'Control Variables', 'Other Controls']:
                    color = 'lightblue'
                elif label == 'Promotions':
                    color = 'gold'
                else:  # Media channels
                    color = 'steelblue'
                
                colors.append(color)
                
                # Draw bar from cumulative[i] to cumulative[i+1]
                bottom = cumulative[i]
                height = val
                
                ax.bar(i, height, bottom=bottom, color=color, edgecolor='black', linewidth=0.5)
                
                # Add value label
                y_pos = bottom + height/2
                ax.text(i, y_pos, f'${val:,.0f}', ha='center', va='center', 
                       fontweight='bold', fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Add total line
            ax.plot([-0.5, len(waterfall_order)-0.5], [total_actual, total_actual], 
                   'r--', linewidth=2, label=f'Total Actual: ${total_actual:,.0f}')
            
            # Formatting
            ax.set_xticks(range(len(waterfall_order)))
            ax.set_xticklabels(waterfall_order, rotation=45, ha='right')
            ax.set_ylabel('Revenue ($)', fontsize=12, fontweight='bold')
            ax.set_title('Revenue Waterfall: How Components Build to Total', fontsize=14, fontweight='bold')
            ax.legend(loc='upper left')
            ax.grid(axis='y', alpha=0.3)
            
            # Add zero line
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Component summary
            st.markdown("---")
            st.markdown("### 📋 Component Summary")
            
            comp_col1, comp_col2, comp_col3 = st.columns(3)
            
            with comp_col1:
                st.markdown("**🎯 Marketing Effectiveness:**")
                media_total = sum([v for k, v in contributions.items() 
                                  if k not in ['Baseline', 'Seasonality', 'Promotions', 'Control Variables', 'Other Controls', 'Residual']])
                media_pct = (media_total / total_actual) * 100
                st.metric("Media Contribution", f"${media_total:,.0f}", f"{media_pct:.1f}% of total")
            
            with comp_col2:
                st.markdown("**🎁 Promotional Impact:**")
                promo_total = contributions.get('Promotions', 0)
                promo_pct = (promo_total / total_actual) * 100 if total_actual > 0 else 0
                st.metric("Promotions", f"${promo_total:,.0f}", f"{promo_pct:.1f}% of total")
            
            with comp_col3:
                st.markdown("**📊 External Factors:**")
                control_total = contributions.get('Control Variables', 0) + contributions.get('Other Controls', 0)
                control_pct = (control_total / total_actual) * 100 if total_actual > 0 else 0
                st.metric("Control Variables", f"${control_total:,.0f}", f"{control_pct:.1f}% of total")
            
            st.markdown("---")
            
            comp_col4, comp_col5, comp_col6 = st.columns(3)
            
            with comp_col4:
                st.markdown("**📅 Seasonal Patterns:**")
                seas_total = contributions.get('Seasonality', 0)
                seas_pct = (seas_total / total_actual) * 100 if total_actual > 0 else 0
                st.metric("Seasonality", f"${seas_total:,.0f}", f"{seas_pct:.1f}% of total")
            
            with comp_col5:
                st.markdown("**🏠 Base Business:**")
                base_total = contributions.get('Baseline', 0)
                base_pct = (base_total / total_actual) * 100 if total_actual > 0 else 0
                st.metric("Baseline", f"${base_total:,.0f}", f"{base_pct:.1f}% of total")
            
            with comp_col6:
                st.markdown("**❓ Unexplained:**")
                resid_total = contributions.get('Residual', 0)
                resid_pct = (abs(resid_total) / total_actual) * 100 if total_actual > 0 else 0
                st.metric("Residual", f"${resid_total:,.0f}", f"{resid_pct:.1f}% unexplained")
        
        # Tab 2: ROI Analysis + DECOMP.RSSD
        with result_tabs[1]:
            st.markdown("### ROI Analysis + DECOMP.RSSD")
            
            roi_data = []
            
            for feat in feat_cols:
                channel_name = meta[feat]['spend_col']
                
                if feat not in model.params.index:
                    continue
                
                try:
                    beta = float(model.params.at[feat])
                except:
                    beta_val = model.params.loc[feat]
                    beta = float(beta_val.iloc[0] if hasattr(beta_val, 'iloc') else beta_val)
                    
                contrib = np.sum(X_test[feat].values * beta)
                total_spend = test_df[channel_name].sum()
                
                if total_spend <= 0:
                    continue
                
                roi = contrib / total_spend if total_spend > 0 else 0
                
                # Get Hill parameters and scaling factor
                adstock_theta = meta[feat]['adstock_theta']
                alpha = meta[feat]['alpha']
                gamma = meta[feat]['gamma']
                x_min = meta[feat]['x_min']
                x_max = meta[feat]['x_max']
                scaling_factor = meta[feat]['scaling_factor']
                current_avg_spend = test_df[channel_name].mean()
                
                if current_avg_spend <= 0:
                    marginal_roas = 0
                else:
                    # Calculate adstocked spend
                    A = current_avg_spend / (1 - adstock_theta) if adstock_theta < 1 else current_avg_spend
                    
                    # Marginal ROAS = β × scaling_factor × hill_derivative / (1 - θ)
                    hill_deriv = hill_derivative(A, alpha, gamma, x_min, x_max)
                    marginal_roas = beta * scaling_factor * hill_deriv / (1 - adstock_theta) if adstock_theta < 1 else 0
                
                roi_data.append({
                    'Channel': channel_name.replace('_Cost', '').replace('_cost', ''),
                    'Total Spend': total_spend,
                    'Revenue Contribution': contrib,
                    'ROI (iROAS)': roi,
                    'Marginal ROI': marginal_roas
                })
            
            if not roi_data:
                st.warning("⚠️ No channels with positive spend")
                st.stop()
            
            roi_df = pd.DataFrame(roi_data).sort_values('ROI (iROAS)', ascending=False)
            
            st.dataframe(
                roi_df.style.format({
                    'Total Spend': '{:,.0f}',
                    'Revenue Contribution': '{:,.0f}',
                    'ROI (iROAS)': '{:.2f}',
                    'Marginal ROI': '{:.2f}'
                }).background_gradient(subset=['ROI (iROAS)', 'Marginal ROI'], cmap='RdYlGn'),
                width='stretch'
            )
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(8, 6))
                roi_df.plot(x='Channel', y='ROI (iROAS)', kind='bar', ax=ax, color='coral', legend=False)
                ax.set_title('ROI by Channel', fontsize=14, fontweight='bold')
                ax.set_ylabel('ROI', fontsize=12)
                ax.set_xlabel('')
                ax.axhline(y=1, color='red', linestyle='--', label='Break-even')
                ax.legend()
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                fig, ax = plt.subplots(figsize=(8, 6))
                roi_df.plot(x='Channel', y='Marginal ROI', kind='bar', ax=ax, color='skyblue', legend=False)
                ax.set_title('Marginal ROI', fontsize=14, fontweight='bold')
                ax.set_ylabel('Marginal ROI', fontsize=12)
                ax.set_xlabel('')
                ax.axhline(y=1, color='red', linestyle='--', label='Break-even')
                ax.legend()
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
            
            # NEW: DECOMP.RSSD Section
            st.markdown("---")
            st.markdown("### 🎯 DECOMP.RSSD - Spend vs Effect Share Analysis")
            
            st.info("""
            **DECOMP.RSSD** measures alignment between spend share and effect share.
            - **Lower is better** (0 = perfect alignment)
            - Values < 0.2 indicate good alignment
            - Values > 0.3 suggest misallocation
            """)
            
            rssd, spend_share, effect_share = calculate_decomp_rssd(test_df, contributions, media_cols)
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.metric("DECOMP.RSSD", f"{rssd:.4f}")
                
                if rssd < 0.1:
                    st.success("✅ Excellent alignment")
                elif rssd < 0.2:
                    st.info("ℹ️ Good alignment")
                elif rssd < 0.3:
                    st.warning("⚠️ Moderate misalignment")
                else:
                    st.error("❌ High misalignment - reallocate budget")
            
            with col2:
                comparison_df = pd.DataFrame({
                    'Channel': media_cols,
                    'Spend Share (%)': [spend_share.get(ch, 0) * 100 for ch in media_cols],
                    'Effect Share (%)': [effect_share.get(ch, 0) * 100 for ch in media_cols],
                    'Difference (pp)': [(effect_share.get(ch, 0) - spend_share.get(ch, 0)) * 100 for ch in media_cols]
                })
                
                st.dataframe(
                    comparison_df.style.format({
                        'Spend Share (%)': '{:.2f}',
                        'Effect Share (%)': '{:.2f}',
                        'Difference (pp)': '{:+.2f}'
                    }).background_gradient(subset=['Difference (pp)'], cmap='RdYlGn', vmin=-10, vmax=10),
                    width='stretch'
                )
            
            # Visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            x = np.arange(len(media_cols))
            width = 0.35
            
            ax.bar(x - width/2, [spend_share.get(ch, 0)*100 for ch in media_cols], 
                   width, label='Spend Share', color='steelblue')
            ax.bar(x + width/2, [effect_share.get(ch, 0)*100 for ch in media_cols], 
                   width, label='Effect Share', color='orange')
            
            ax.set_xlabel('Channel')
            ax.set_ylabel('Share (%)')
            ax.set_title('Spend Share vs Effect Share', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([ch.replace('_Cost', '').replace('_cost', '') for ch in media_cols], rotation=45, ha='right')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Media Channel Statistical Significance
            st.markdown("---")
            st.markdown("### 📊 Media Channel Coefficient Confidence Intervals")
            
            st.info("""
            **Statistical Significance Check:** Are the effects real or just noise?
            - **CI excludes 0** → ✅ Statistically significant effect
            - **CI includes 0** → ⚠️ Effect not statistically significant
            """)
            
            # Get confidence intervals for media features
            conf_intervals = model.conf_int(alpha=0.05)  # 95% CI
            
            media_ci_data = []
            for feat in feat_cols:
                channel_name = meta[feat]['spend_col']
                if feat in model.params.index:
                    try:
                        coef = float(model.params.at[feat])
                    except:
                        coef_val = model.params.loc[feat]
                        coef = float(coef_val.iloc[0] if hasattr(coef_val, 'iloc') else coef_val)
                    
                    try:
                        ci_lower = float(conf_intervals.at[feat, 0])
                    except:
                        try:
                            ci_lower = float(conf_intervals.loc[feat].iloc[0])
                        except:
                            ci_lower = 0.0
                    
                    try:
                        ci_upper = float(conf_intervals.at[feat, 1])
                    except:
                        try:
                            ci_upper = float(conf_intervals.loc[feat].iloc[1])
                        except:
                            ci_upper = 0.0
                    
                    try:
                        p_value = float(model.pvalues.at[feat])
                    except:
                        p_val = model.pvalues.loc[feat]
                        p_value = float(p_val.iloc[0] if hasattr(p_val, 'iloc') else p_val)
                    
                    # Check if significant
                    is_significant = (ci_lower > 0 or ci_upper < 0)
                    
                    media_ci_data.append({
                        'Channel': channel_name.replace('_Cost', '').replace('_cost', ''),
                        'Coefficient': coef,
                        'CI Lower (2.5%)': ci_lower,
                        'CI Upper (97.5%)': ci_upper,
                        'P-Value': p_value,
                        'Significant': '✅ Yes' if is_significant else '⚠️ No'
                    })
            
            media_ci_df = pd.DataFrame(media_ci_data)
            
            st.dataframe(
                media_ci_df.style.format({
                    'Coefficient': '{:.4f}',
                    'CI Lower (2.5%)': '{:.4f}',
                    'CI Upper (97.5%)': '{:.4f}',
                    'P-Value': '{:.4f}'
                }).apply(lambda x: ['background-color: #d4edda' if v == '✅ Yes' else 'background-color: #fff3cd' 
                                     for v in x], subset=['Significant']),
                width='stretch'
            )
            
            # Coefficient plot for media channels
            if len(media_ci_df) > 0:
                fig, ax = plt.subplots(figsize=(10, max(5, len(media_ci_df) * 0.5)))
                
                y_pos = np.arange(len(media_ci_df))
                
                # Plot coefficient points
                ax.scatter(media_ci_df['Coefficient'], y_pos, s=120, c='steelblue', zorder=3, edgecolors='black', linewidths=1.5)
                
                # Plot confidence interval error bars
                for i, row in media_ci_df.iterrows():
                    lower = row['CI Lower (2.5%)']
                    upper = row['CI Upper (97.5%)']
                    coef = row['Coefficient']
                    
                    # Color: green if significant, orange if not
                    color = 'green' if row['Significant'] == '✅ Yes' else 'orange'
                    ax.plot([lower, upper], [i, i], color=color, linewidth=3, zorder=2, alpha=0.7)
                
                # Add vertical line at 0
                ax.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.6, label='Zero Effect')
                
                ax.set_yticks(y_pos)
                ax.set_yticklabels(media_ci_df['Channel'])
                ax.set_xlabel('Coefficient Value (Standardized Effect Size)', fontsize=11, fontweight='bold')
                ax.set_title('Media Channel Effects with 95% Confidence Intervals', fontsize=13, fontweight='bold')
                ax.grid(axis='x', alpha=0.3)
                ax.legend(loc='best')
                
                # Add annotation
                ax.text(0.02, 0.98, '🟢 Significant | 🟠 Not Significant', 
                       transform=ax.transAxes, fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
                
                plt.tight_layout()
                st.pyplot(fig)
            
            st.caption("💡 Channels with CIs that don't include zero have statistically significant effects on revenue")
            
            # Insights
            st.markdown("---")
            st.markdown("### 💡 Key Insights")
            
            best_roi_channel = roi_df.iloc[0]
            best_marginal_channel = roi_df.sort_values('Marginal ROI', ascending=False).iloc[0]
            
            insight_col1, insight_col2 = st.columns(2)
            
            with insight_col1:
                st.info(f"""
                **Best Overall ROI:**
                - **{best_roi_channel['Channel']}** has ROI of **{best_roi_channel['ROI (iROAS)']:.2f}**
                - Every $1 spent returns ${best_roi_channel['ROI (iROAS)']:.2f}
                """)
            
            with insight_col2:
                st.info(f"""
                **Best Marginal Efficiency:**
                - **{best_marginal_channel['Channel']}** has marginal ROI of **{best_marginal_channel['Marginal ROI']:.2f}**
                - Most room for additional investment
                """)
        
        # Tab 3: Response Curves
        with result_tabs[2]:
            st.markdown("### Saturation & Response Curves")
            
            selected_channel = st.selectbox(
                "Select channel to analyze",
                [meta[feat]['spend_col'] for feat in feat_cols],
                key='curve_channel'
            )
            
            feat = [f for f in feat_cols if meta[f]['spend_col'] == selected_channel][0]
            
            if feat not in model.params.index:
                st.error(f"⚠️ Coefficient not found for {selected_channel}")
                st.stop()
            
            try:
                beta = float(model.params.at[feat])
            except:
                beta_val = model.params.loc[feat]
                beta = float(beta_val.iloc[0] if hasattr(beta_val, 'iloc') else beta_val)
                
            # Get gamma-based Hill parameters (NEW)
            adstock_theta = meta[feat]['adstock_theta']
            alpha = meta[feat]['alpha']
            gamma = meta[feat]['gamma']
            x_min = meta[feat]['x_min']
            x_max = meta[feat]['x_max']
            
            historical_spend = test_df[selected_channel].values
            valid_spend = historical_spend[historical_spend > 0]
            
            if len(valid_spend) == 0:
                st.warning(f"⚠️ No positive spend for {selected_channel}")
                st.stop()
            
            max_spend = np.percentile(valid_spend, 95)
            spend_range = np.linspace(0, max_spend * 1.5, 200)
            
            # Apply adstock
            if adstock_theta < 1:
                adstocked = spend_range / (1 - adstock_theta)
            else:
                adstocked = spend_range
            
            # Apply Hill saturation with scaling
            saturated_raw = hill_transformation(adstocked, alpha, gamma)
            scaling_factor = meta[feat]['scaling_factor']
            saturated = saturated_raw * scaling_factor
            revenue = beta * saturated  # Scaled contribution
            
            # Calculate marginal ROAS (derivative accounts for scaling)
            if adstock_theta < 1:
                hill_deriv = hill_derivative(adstocked, alpha, gamma, x_min, x_max)
                # Marginal ROAS = β × scaling_factor × hill_derivative / (1 - θ)
                marginal_roas = beta * scaling_factor * hill_deriv / (1 - adstock_theta)
            else:
                marginal_roas = np.zeros_like(spend_range)
            
            # Calculate iROAS
            iroas = np.zeros_like(revenue)
            for i in range(1, len(revenue)):
                iroas[i] = revenue[i] / spend_range[i] if spend_range[i] > 0 else 0
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            axes[0, 0].plot(spend_range, revenue, color='steelblue', linewidth=2)
            axes[0, 0].axvline(historical_spend.mean(), color='red', linestyle='--', label='Current avg')
            axes[0, 0].set_title('Saturation Curve', fontsize=14, fontweight='bold')
            axes[0, 0].set_xlabel('Daily Spend')
            axes[0, 0].set_ylabel('Incremental Revenue')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            axes[0, 1].plot(spend_range, marginal_roas, color='coral', linewidth=2)
            axes[0, 1].axvline(historical_spend.mean(), color='red', linestyle='--', label='Current avg')
            axes[0, 1].axhline(y=1, color='green', linestyle='--', label='Break-even')
            axes[0, 1].set_title('Marginal ROAS', fontsize=14, fontweight='bold')
            axes[0, 1].set_xlabel('Daily Spend')
            axes[0, 1].set_ylabel('Marginal ROAS')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            axes[1, 0].plot(spend_range[1:], iroas[1:], color='purple', linewidth=2)
            axes[1, 0].axvline(historical_spend.mean(), color='red', linestyle='--', label='Current avg')
            axes[1, 0].axhline(y=1, color='green', linestyle='--', label='Break-even')
            axes[1, 0].set_title('Incremental ROAS', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Daily Spend')
            axes[1, 0].set_ylabel('iROAS')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            efficiency = revenue / spend_range
            efficiency[0] = 0
            axes[1, 1].plot(spend_range, efficiency, color='green', linewidth=2)
            axes[1, 1].axvline(historical_spend.mean(), color='red', linestyle='--', label='Current avg')
            axes[1, 1].set_title('Spend Efficiency', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Daily Spend')
            axes[1, 1].set_ylabel('Revenue / Spend')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.markdown("---")
            st.markdown("### 📊 Current Performance")
            
            current_spend = valid_spend.mean()
            current_idx = np.argmin(np.abs(spend_range - current_spend))
            
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            
            with metric_col1:
                st.metric("Current Avg Spend", f"${current_spend:,.0f}")
            with metric_col2:
                st.metric("Marginal ROAS", f"{marginal_roas[current_idx]:.2f}")
            with metric_col3:
                st.metric("iROAS", f"{iroas[current_idx]:.2f}")
            with metric_col4:
                if saturated[-1] > 0:
                    saturation_level = (saturated[current_idx] / saturated[-1]) * 100
                else:
                    saturation_level = 0
                st.metric("Saturation Level", f"{saturation_level:.1f}%")
        
        # Tab 4: Budget Optimization
        with result_tabs[3]:
            st.markdown("### Budget Allocation Optimizer (Scipy SLSQP)")
            
            st.info("""
            **Optimization Method:** Scipy SLSQP solver
            - Maximizes total revenue
            - Accounts for adstock + saturation
            - Preserves temporal dynamics
            """)
            
            current_budget = sum([test_df[meta[feat]['spend_col']].sum() for feat in feat_cols])
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                new_budget = st.slider(
                    "Total Budget",
                    min_value=int(current_budget * 0.5),
                    max_value=int(current_budget * 2),
                    value=int(current_budget),
                    step=int(current_budget * 0.05),
                    format="$%d"
                )
            
            with col2:
                budget_change = ((new_budget - current_budget) / current_budget) * 100
                st.metric("Budget Change", f"{budget_change:+.1f}%")
            
            if st.button("🚀 Run Optimization", type="primary", width='stretch'):
                with st.spinner("Running optimization..."):
                    try:
                        # Calculate baseline contribution
                        if 'const' in model.params.index:
                            try:
                                baseline_contrib = float(model.params.at['const']) * len(test_df)
                            except:
                                beta_val = model.params.loc['const']
                                baseline_contrib = float(beta_val.iloc[0] if hasattr(beta_val, 'iloc') else beta_val) * len(test_df)
                        else:
                            baseline_contrib = 0
                        
                        # Calculate seasonality contribution
                        seasonality_cols = [col for col in X_test.columns if 'dow_' in col or 'month_' in col]
                        seasonality_contrib = 0
                        for col in seasonality_cols:
                            if col in X_test.columns and col in model.params.index:
                                try:
                                    beta = float(model.params.at[col])
                                except:
                                    beta_val = model.params.loc[col]
                                    beta = float(beta_val.iloc[0] if hasattr(beta_val, 'iloc') else beta_val)
                                seasonality_contrib += (X_test[col].values * beta).sum()
                        
                        def mmm_objective(channel_totals):
                            total_revenue = baseline_contrib + seasonality_contrib
                            
                            for i, feat in enumerate(feat_cols):
                                if feat not in model.params.index:
                                    continue
                                
                                try:
                                    beta = float(model.params.at[feat])
                                except:
                                    beta_val = model.params.loc[feat]
                                    beta = float(beta_val.iloc[0] if hasattr(beta_val, 'iloc') else beta_val)
                                    
                                channel_name = meta[feat]['spend_col']
                                # Get transformation parameters
                                adstock_theta = meta[feat]['adstock_theta']
                                alpha = meta[feat]['alpha']
                                gamma = meta[feat]['gamma']
                                
                                current_total = test_df[channel_name].sum()
                                optimized_total = channel_totals[i]
                                
                                if current_total > 0:
                                    scale = optimized_total / current_total
                                else:
                                    scale = 0
                                
                                # Scale daily spend pattern
                                scaled_daily_spend = test_df[channel_name].values * scale
                                
                                # Apply transformations (NO standardization)
                                adstocked_spend = adstock_transformation(scaled_daily_spend, alpha=adstock_theta)
                                
                                # Apply Hill with FIXED inflection from training
                                # This ensures consistent saturation curve
                                x_min = meta[feat]['x_min']  # Fixed from training
                                x_max = meta[feat]['x_max']  # Fixed from training
                                inflexion = x_min * (1 - gamma) + x_max * gamma
                                inflexion = max(float(inflexion), 1e-9)
                                
                                # Manual Hill calculation with fixed inflection
                                adstocked_spend_clipped = np.maximum(adstocked_spend, 0.0)
                                x_alpha = np.power(adstocked_spend_clipped, alpha)
                                inflexion_alpha = np.power(inflexion, alpha)
                                saturated_raw = x_alpha / (x_alpha + inflexion_alpha)
                                
                                # Apply same scaling as training: multiply by scaling_factor (mean_spend)
                                # This ensures consistency with how features were created
                                scaling_factor = meta[feat]['scaling_factor']
                                saturated_spend = saturated_raw * scaling_factor
                                
                                # Calculate contribution
                                channel_revenue = np.sum(beta * saturated_spend)
                                total_revenue += channel_revenue
                            
                            return -total_revenue
                        
                        def budget_constraint(channel_totals):
                            return np.sum(channel_totals) - new_budget
                        
                        initial_totals = [test_df[meta[feat]['spend_col']].sum() for feat in feat_cols]
                        
                        # Add small perturbation to initial guess to help optimizer explore
                        # Even if budget unchanged, force optimizer to find true optimum
                        np.random.seed(42)  # Reproducibility
                        perturbation = np.random.uniform(0.95, 1.05, len(initial_totals))
                        perturbed_totals = np.array(initial_totals) * perturbation
                        # Rescale to match budget
                        perturbed_totals = perturbed_totals * (new_budget / np.sum(perturbed_totals))
                        
                        bounds = [(new_budget * 0.01, new_budget * 0.99) for _ in feat_cols]  # Each channel 1-99% of budget
                        
                        solution = minimize(
                            fun=mmm_objective,
                            x0=perturbed_totals,  # Start from perturbed allocation
                            bounds=bounds,
                            method="SLSQP",
                            constraints={'type': 'eq', 'fun': budget_constraint},
                            options={'maxiter': 1000, 'ftol': 1e-6, 'disp': False}  # Relaxed tolerance
                        )
                        
                        if solution.success:
                            st.success("✅ Optimization complete!")
                            
                            allocation_data = []
                            for i, feat in enumerate(feat_cols):
                                channel_name = meta[feat]['spend_col']
                                current_spend = test_df[channel_name].sum()
                                optimized_spend = solution.x[i]
                                
                                allocation_data.append({
                                    'Channel': channel_name.replace('_Cost', '').replace('_cost', ''),
                                    'Current Spend': current_spend,
                                    'Optimized Spend': optimized_spend,
                                    'Change': optimized_spend - current_spend,
                                    'Change %': ((optimized_spend - current_spend) / current_spend * 100) if current_spend > 0 else 0
                                })
                            
                            alloc_df = pd.DataFrame(allocation_data)
                            
                            st.markdown("---")
                            st.markdown("#### 📊 Optimal Allocation")
                            
                            st.dataframe(
                                alloc_df.style.format({
                                    'Current Spend': '{:,.0f}',
                                    'Optimized Spend': '{:,.0f}',
                                    'Change': '{:+,.0f}',
                                    'Change %': '{:+.1f}%'
                                }).background_gradient(subset=['Change %'], cmap='RdYlGn', vmin=-50, vmax=50),
                                width='stretch'
                            )
                            
                            st.markdown("---")
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                            
                            x = np.arange(len(alloc_df))
                            width = 0.35
                            
                            ax1.bar(x - width/2, alloc_df['Current Spend'], width, label='Current', color='steelblue')
                            ax1.bar(x + width/2, alloc_df['Optimized Spend'], width, label='Optimized', color='coral')
                            ax1.set_xlabel('Channel')
                            ax1.set_ylabel('Budget')
                            ax1.set_title('Current vs Optimized Budget', fontsize=14, fontweight='bold')
                            ax1.set_xticks(x)
                            ax1.set_xticklabels(alloc_df['Channel'], rotation=45, ha='right')
                            ax1.legend()
                            ax1.grid(axis='y', alpha=0.3)
                            
                            colors = ['green' if x > 0 else 'red' for x in alloc_df['Change %']]
                            ax2.barh(alloc_df['Channel'], alloc_df['Change %'], color=colors)
                            ax2.set_xlabel('Change (%)')
                            ax2.set_title('Budget Change', fontsize=14, fontweight='bold')
                            ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
                            ax2.grid(axis='x', alpha=0.3)
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                            st.markdown("---")
                            st.markdown("### 📈 Expected Impact")
                            
                            current_allocation = [test_df[meta[feat]['spend_col']].sum() for feat in feat_cols]
                            current_revenue_model = -mmm_objective(current_allocation)
                            optimized_revenue = -solution.fun
                            expected_lift = optimized_revenue - current_revenue_model
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Current Revenue (Model)", f"${current_revenue_model:,.0f}")
                            with col2:
                                st.metric("Optimized Revenue", f"${optimized_revenue:,.0f}", delta=f"${expected_lift:,.0f}")
                            with col3:
                                lift_pct = (expected_lift / current_revenue_model) * 100 if current_revenue_model > 0 else 0
                                st.metric("Expected Lift", f"{lift_pct:+.1f}%")
                            
                            with st.expander("📊 Model vs Actual"):
                                actual_revenue = y_test.sum()
                                prediction_error = ((current_revenue_model - actual_revenue) / actual_revenue * 100)
                                
                                st.write(f"**Actual Revenue:** ${actual_revenue:,.0f}")
                                st.write(f"**Model Prediction:** ${current_revenue_model:,.0f}")
                                st.write(f"**Prediction Error:** {prediction_error:+.1f}%")
                            
                            with st.expander("🔧 Optimization Details"):
                                st.write(f"**Status:** {solution.message}")
                                st.write(f"**Iterations:** {solution.nit}")
                                st.write(f"**Function Evals:** {solution.nfev}")
                        
                        else:
                            st.error(f"❌ Optimization failed: {solution.message}")
                    
                    except Exception as e:
                        st.error(f"Error: {e}")
                        import traceback
                        st.code(traceback.format_exc())
        
        # Tab 5: Model Summary + VIF
        with result_tabs[4]:
            st.markdown("### Model Summary + VIF Analysis")
            
            # NEW: VIF Analysis Section
            st.markdown("#### 🎯 VIF Analysis (Multicollinearity Detection)")
            
            st.info("""
            **VIF (Variance Inflation Factor)** measures multicollinearity:
            - **VIF < 5:** ✅ Low multicollinearity (good)
            - **VIF 5-10:** ⚠️ Moderate (acceptable)
            - **VIF > 10:** ❌ High (consider removing variable)
            - **VIF = inf:** ⚠️ Perfect multicollinearity (variable is constant or duplicate)
            - **VIF = None:** Dummy variables may show this (one category with no variance)
            """)
            
            try:
                vif_data = pd.DataFrame()
                vif_data["Feature"] = X_test.columns[1:]  # Exclude const
                
                # Calculate VIF with better handling of inf/None
                vif_values = []
                for i in range(1, X_test.shape[1]):
                    try:
                        vif = variance_inflation_factor(X_test.values, i)
                        # Handle inf and None
                        if not np.isfinite(vif):
                            vif_values.append(float('inf'))
                        else:
                            vif_values.append(vif)
                    except:
                        vif_values.append(None)
                
                vif_data["VIF"] = vif_values
                
                def vif_color(val):
                    if val is None or (isinstance(val, float) and np.isnan(val)):
                        return 'background-color: #ccffcc'  # Green for None
                    elif np.isinf(val):
                        return 'background-color: #ffcccc'  # Red for inf
                    elif val > 10:
                        return 'background-color: #ffcccc'  # Red
                    elif val > 5:
                        return 'background-color: #ffffcc'  # Yellow
                    return 'background-color: #ccffcc'  # Green
                
                # Format VIF values for display
                def format_vif(val):
                    if val is None:
                        return 'None'
                    elif np.isinf(val):
                        return 'inf'
                    else:
                        return f'{val:.2f}'
                
                vif_data['VIF_display'] = vif_data['VIF'].apply(format_vif)
                
                st.dataframe(
                    vif_data[['Feature', 'VIF_display']].rename(columns={'VIF_display': 'VIF'})
                        .style.apply(lambda x: [vif_color(v) if c == 'VIF' else '' 
                                                for c, v in zip(x.index, vif_data['VIF'])], axis=1),
                    width='stretch'
                )
                
                # VIF Summary
                high_vif = vif_data[vif_data['VIF'].apply(lambda x: x > 10 if x is not None and np.isfinite(x) else False)]
                inf_vif = vif_data[vif_data['VIF'].apply(lambda x: np.isinf(x) if x is not None else False)]
                
                if len(inf_vif) > 0:
                    st.error(f"❌ {len(inf_vif)} variable(s) with VIF = inf (perfect multicollinearity)!")
                    st.write("**Variables with infinite VIF (likely duplicates or constants):**")
                    st.dataframe(inf_vif[['Feature', 'VIF_display']].rename(columns={'VIF_display': 'VIF'}), width='stretch')
                
                if len(high_vif) > 0:
                    st.warning(f"⚠️ {len(high_vif)} variable(s) with VIF > 10 detected!")
                    st.dataframe(high_vif[['Feature', 'VIF_display']].rename(columns={'VIF_display': 'VIF'}), width='stretch')
                
                if len(high_vif) == 0 and len(inf_vif) == 0:
                    st.success("✅ No high multicollinearity detected")
                
            except Exception as e:
                st.error(f"Error calculating VIF: {e}")
            
            # Model Coefficients with Confidence Intervals
            st.markdown("---")
            st.markdown("#### 📊 Model Coefficients with 95% Confidence Intervals")
            
            # Get confidence intervals
            conf_intervals = model.conf_int(alpha=0.05)  # 95% CI
            
            coef_data = []
            for param in model.params.index:
                # Extract scalar values to avoid Series ambiguity - use .at[] accessor
                try:
                    p_val = float(model.pvalues.at[param])
                except:
                    p_val_temp = model.pvalues.loc[param]
                    p_val = float(p_val_temp.iloc[0] if hasattr(p_val_temp, 'iloc') else p_val_temp)
                
                try:
                    coef = float(model.params.at[param])
                except:
                    coef_temp = model.params.loc[param]
                    coef = float(coef_temp.iloc[0] if hasattr(coef_temp, 'iloc') else coef_temp)
                
                try:
                    std_err = float(model.bse.at[param])
                except:
                    se_temp = model.bse.loc[param]
                    std_err = float(se_temp.iloc[0] if hasattr(se_temp, 'iloc') else se_temp)
                
                try:
                    t_stat = float(model.tvalues.at[param])
                except:
                    t_temp = model.tvalues.loc[param]
                    t_stat = float(t_temp.iloc[0] if hasattr(t_temp, 'iloc') else t_temp)
                
                # Determine significance level
                if p_val < 0.001:
                    sig = '***'
                elif p_val < 0.01:
                    sig = '**'
                elif p_val < 0.05:
                    sig = '*'
                else:
                    sig = ''
                
                # Get confidence intervals safely
                try:
                    ci_lower = float(conf_intervals.at[param, 0])
                except:
                    try:
                        ci_lower = float(conf_intervals.loc[param].iloc[0])
                    except:
                        ci_lower = 0.0
                
                try:
                    ci_upper = float(conf_intervals.at[param, 1])
                except:
                    try:
                        ci_upper = float(conf_intervals.loc[param].iloc[1])
                    except:
                        ci_upper = 0.0
                
                ci_width = ci_upper - ci_lower
                
                coef_data.append({
                    'Variable': param,
                    'Coefficient': coef,
                    'Std Error': std_err,
                    'CI Lower (2.5%)': ci_lower,
                    'CI Upper (97.5%)': ci_upper,
                    'CI Width': ci_width,
                    'T-Statistic': t_stat,
                    'P-Value': p_val,
                    'Significant': sig
                })
            
            coef_df = pd.DataFrame(coef_data)
            
            st.dataframe(
                coef_df.style.format({
                    'Coefficient': '{:.4f}',
                    'Std Error': '{:.4f}',
                    'CI Lower (2.5%)': '{:.4f}',
                    'CI Upper (97.5%)': '{:.4f}',
                    'CI Width': '{:.4f}',
                    'T-Statistic': '{:.4f}',
                    'P-Value': '{:.4f}'
                }).background_gradient(subset=['Coefficient'], cmap='coolwarm', vmin=-1, vmax=1),
                width='stretch'
            )
            
            st.caption("Significance: *** p<0.001, ** p<0.01, * p<0.05")
            
            # Download button for coefficients table
            csv_coef = coef_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Coefficients with CIs",
                data=csv_coef,
                file_name=f"model_coefficients_CI_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                key="download_coef_ci"
            )
            
            # Confidence Interval Visualization
            st.markdown("---")
            st.markdown("#### 📊 Coefficient Estimates with 95% Confidence Intervals")
            
            # Filter to show only media channels and key variables (exclude seasonality dummies for clarity)
            key_vars = [v for v in coef_df['Variable'] if not any(x in v for x in ['dow_', 'month_']) or v == 'const']
            plot_df = coef_df[coef_df['Variable'].isin(key_vars)].copy()
            
            if len(plot_df) > 0:
                fig, ax = plt.subplots(figsize=(10, max(6, len(plot_df) * 0.4)))
                
                # Sort by coefficient value
                plot_df = plot_df.sort_values('Coefficient')
                
                y_pos = np.arange(len(plot_df))
                
                # Plot coefficient points
                ax.scatter(plot_df['Coefficient'], y_pos, s=100, c='steelblue', zorder=3, label='Coefficient')
                
                # Plot confidence interval error bars
                ci_lower = plot_df['CI Lower (2.5%)'].values
                ci_upper = plot_df['CI Upper (97.5%)'].values
                
                for i, (lower, upper, coef) in enumerate(zip(ci_lower, ci_upper, plot_df['Coefficient'].values)):
                    # Color code: green if CI doesn't include 0, orange if it does
                    color = 'green' if (lower > 0 or upper < 0) else 'orange'
                    ax.plot([lower, upper], [i, i], color=color, linewidth=2, zorder=2)
                
                # Add vertical line at 0
                ax.axvline(x=0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Zero')
                
                ax.set_yticks(y_pos)
                ax.set_yticklabels(plot_df['Variable'])
                ax.set_xlabel('Coefficient Value', fontsize=11, fontweight='bold')
                ax.set_title('Model Coefficients with 95% Confidence Intervals', fontsize=13, fontweight='bold')
                ax.grid(axis='x', alpha=0.3)
                ax.legend()
                
                plt.tight_layout()
                st.pyplot(fig)
                
                st.caption("🟢 Green bars: Statistically significant (CI doesn't include 0) | 🟠 Orange bars: Not significant (CI includes 0)")
            
            # Statistical note
            with st.expander("ℹ️ Understanding Confidence Intervals"):
                st.markdown("""
                **What are Confidence Intervals?**
                - A 95% CI means: "If we repeated this analysis 100 times, the true coefficient would fall within this range in 95 of those runs"
                
                **Interpretation:**
                - **Narrow CI** = More precise estimate (good!)
                - **Wide CI** = Less precise estimate (need more data or less noise)
                - **CI excludes 0** = Statistically significant effect
                - **CI includes 0** = Cannot confidently say effect is non-zero
                
                **Example:**
                - Coefficient: 2.5, CI: [1.2, 3.8] → Significant positive effect
                - Coefficient: 0.8, CI: [-0.5, 2.1] → Not significant (could be zero)
                """)
            
            # Model Statistics
            st.markdown("---")
            st.markdown("#### 📈 Model Statistics")
            
            stat_col1, stat_col2, stat_col3 = st.columns(3)
            
            with stat_col1:
                st.metric("R-squared", f"{model.rsquared:.4f}")
                st.metric("Adj. R-squared", f"{model.rsquared_adj:.4f}")
            
            with stat_col2:
                st.metric("F-statistic", f"{model.fvalue:.2f}")
                st.metric("Prob (F-stat)", f"{model.f_pvalue:.4e}")
            
            with stat_col3:
                st.metric("AIC", f"{model.aic:.2f}")
                st.metric("BIC", f"{model.bic:.2f}")
            
            # Model Diagnostics
            st.markdown("---")
            st.markdown("#### 🔍 Model Diagnostics")
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            residuals = y_test - y_test_pred
            
            axes[0, 0].scatter(y_test_pred, residuals, alpha=0.5)
            axes[0, 0].axhline(y=0, color='red', linestyle='--')
            axes[0, 0].set_xlabel('Fitted Values')
            axes[0, 0].set_ylabel('Residuals')
            axes[0, 0].set_title('Residuals vs Fitted', fontweight='bold')
            axes[0, 0].grid(True, alpha=0.3)
            
            stats.probplot(residuals, dist="norm", plot=axes[0, 1])
            axes[0, 1].set_title('Normal Q-Q Plot', fontweight='bold')
            axes[0, 1].grid(True, alpha=0.3)
            
            axes[1, 0].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
            axes[1, 0].set_xlabel('Residuals')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Distribution of Residuals', fontweight='bold')
            axes[1, 0].grid(True, alpha=0.3)
            
            axes[1, 1].scatter(y_test, y_test_pred, alpha=0.5)
            axes[1, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                           'r--', lw=2, label='Perfect')
            axes[1, 1].set_xlabel('Actual')
            axes[1, 1].set_ylabel('Predicted')
            axes[1, 1].set_title('Actual vs Predicted', fontweight='bold')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><b>Marketing Mix Modeling Platform</b></p>
    <p>Gamma-Based Hill Saturation | No Standardization | Control Variables | DECOMP.RSSD | VIF | 95% CI</p>
    <p>Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)
