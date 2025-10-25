# dashboard_improved.py
# Improved Uber Trips Dashboard with fair Prophet evaluation and stronger CNN/LSTM pipelines

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import timedelta, datetime
import os

# sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Prophet (optional)
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    Prophet = None
    PROPHET_AVAILABLE = False

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense, LSTM, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping

# reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Streamlit setup
st.set_page_config(page_title="Uber Trips Dashboard", layout="wide", page_icon="🚕")
st.markdown("""
<style>
    .main-header { font-size: 32px; color: #1E88E5; text-align: center; }
</style>
""", unsafe_allow_html=True)

REQUIRED_COLS = {'START_DATE', 'END_DATE', 'CATEGORY', 'START', 'STOP', 'MILES', 'PURPOSE'}

# ----------------- Helpers -----------------
def validate_columns(df):
    missing = REQUIRED_COLS - set(df.columns)
    return missing

@st.cache_data(show_spinner=False)
def load_and_prepare(path="Data.csv"):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    missing = validate_columns(df)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    # datetimes
    df['START_DATE'] = pd.to_datetime(df['START_DATE'], errors='coerce', format="%Y-%m-%d %H:%M")
    df['END_DATE'] = pd.to_datetime(df['END_DATE'], errors='coerce', format="%Y-%m-%d %H:%M")
    # derived
    df['Duration(min)'] = (df['END_DATE'] - df['START_DATE']).dt.total_seconds() / 60
    df['Date'] = df['START_DATE'].dt.date
    df['Hour'] = df['START_DATE'].dt.hour
    df['Weekday'] = df['START_DATE'].dt.day_name()
    df['PURPOSE'] = df['PURPOSE'].fillna('Unknown')
    return df

def series_from_target(filtered_df, target: str):
    if target == "Trip Count":
        daily = filtered_df.groupby('Date').size().reset_index(name='y')
    elif target == "Total Miles":
        daily = filtered_df.groupby('Date')['MILES'].sum().reset_index(name='y')
    elif target == "Avg Duration (min)":
        daily = filtered_df.groupby('Date')['Duration(min)'].mean().reset_index(name='y')
    else:
        raise ValueError("Unknown target")
    daily['ds'] = pd.to_datetime(daily['Date'])
    daily = daily[['ds','y']].sort_values('ds').reset_index(drop=True)
    return daily

def check_variability(values: np.ndarray):
    vals = np.array(values).astype(float)
    if vals.size == 0:
        return 0.0, False
    var = float(np.var(vals))
    mean = float(vals.mean()) if vals.mean() != 0 else 1.0
    sufficient = var > (0.001 * mean)
    return var, sufficient

def prepare_window_train_test(ts_values: np.ndarray, window: int, test_ratio: float = 0.2):
    """
    Prepare sequences for DL models. Fit scaler on train only.
    Returns: X_train, X_test, y_train, y_test, scaler
    """
    vals = np.array(ts_values).astype(float)
    if len(vals) <= window + 1:
        return None
    # create sequences before scaling (so split maintains time order)
    X_all, y_all = [], []
    for i in range(len(vals) - window):
        X_all.append(vals[i:i+window])
        y_all.append(vals[i+window])
    X_all = np.array(X_all)
    y_all = np.array(y_all)
    # train-test split in time order
    split_idx = int(len(X_all) * (1 - test_ratio))
    X_train_raw, X_test_raw = X_all[:split_idx], X_all[split_idx:]
    y_train_raw, y_test_raw = y_all[:split_idx], y_all[split_idx:]
    # fit scaler on train values
    scaler = MinMaxScaler()
    scaler.fit(y_train_raw.reshape(-1,1))
    # scale X and y using scaler (apply to values, not to features individually)
    def scale_set(X_raw):
        X_scaled = []
        for seq in X_raw:
            X_scaled.append(scaler.transform(seq.reshape(-1,1)).flatten())
        return np.array(X_scaled)
    X_train = scale_set(X_train_raw)
    X_test = scale_set(X_test_raw)
    y_train = scaler.transform(y_train_raw.reshape(-1,1)).flatten()
    y_test = scaler.transform(y_test_raw.reshape(-1,1)).flatten()
    # reshape for Keras (samples, window, 1)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    return X_train, X_test, y_train, y_test, scaler

def build_cnn(window:int):
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=(window,1)),
        Conv1D(64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def build_stacked_lstm(window:int):
    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True), input_shape=(window,1)),
        Dropout(0.3),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def evaluate_model_keras(model, X_test, y_test, scaler):
    if len(X_test) == 0:
        return np.nan, np.nan, np.nan, np.array([]), np.array([])
    preds_scaled = model.predict(X_test, verbose=0).flatten()
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1,1)).flatten()
    preds_inv = scaler.inverse_transform(preds_scaled.reshape(-1,1)).flatten()
    rmse_val = float(np.sqrt(mean_squared_error(y_test_inv, preds_inv)))
    mae_val = float(mean_absolute_error(y_test_inv, preds_inv))
    r2_val = float(r2_score(y_test_inv, preds_inv))
    return rmse_val, mae_val, r2_val, y_test_inv, preds_inv

def iterative_forecast_keras(model, last_window_raw, horizon, scaler, window):
    """
    last_window_raw: last raw values of length window (not scaled)
    scaler: fitted scaler from training (fitted on train target values)
    """
    # scale last window using scaler
    last_scaled = scaler.transform(np.array(last_window_raw).reshape(-1,1)).flatten()
    cur = last_scaled.copy()
    out_scaled = []
    for _ in range(horizon):
        p = model.predict(cur.reshape(1, window, 1), verbose=0)[0][0]
        out_scaled.append(p)
        cur = np.append(cur[1:], p)
    out = scaler.inverse_transform(np.array(out_scaled).reshape(-1,1)).flatten()
    return out

# ----------------- UI & Data load -----------------
st.sidebar.title("Settings")
data_path = st.sidebar.text_input("CSV path", "Data.csv")

try:
    df = load_and_prepare(data_path)
except Exception as e:
    st.sidebar.error(f"Failed to load dataset: {e}")
    st.stop()

min_date = df['START_DATE'].min().date()
max_date = df['START_DATE'].max().date()
date_range = st.sidebar.date_input("Filter by date range", [min_date, max_date])
hour_range = st.sidebar.slider("Hour range", 0, 23, (0,23))
detrend = st.sidebar.checkbox("Apply detrending (7-day rolling mean) before DL models", value=True)

filtered_df = df.copy()
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    filtered_df = filtered_df[
        (filtered_df['START_DATE'].dt.date >= date_range[0]) &
        (filtered_df['START_DATE'].dt.date <= date_range[1])
    ]
filtered_df = filtered_df[
    (filtered_df['Hour'] >= hour_range[0]) & (filtered_df['Hour'] <= hour_range[1])
]
if filtered_df.empty:
    st.warning("Filters returned no rows — showing full dataset.")
    filtered_df = df.copy()

# header metrics
st.markdown("<h1 class='main-header'>Uber Trips Dashboard </h1>", unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
c1.metric("Total Trips", len(filtered_df))
c2.metric("Total Miles (km)", f"{filtered_df['MILES'].sum():.2f}")
c3.metric("Avg Duration (min)", f"{filtered_df['Duration(min)'].mean():.2f}")

# Tabs
tab1, tab_forecast, tab_ml = st.tabs([
    "Data & EDA",
    "Forecasting (Prophet + CNN + LSTM)",
    "ML Models (LR, DT, RF)"
])

# ---------------- Tab 1: Data & EDA ----------------
with tab1:
    st.header("Data & Exploratory Analysis")
    st.dataframe(filtered_df.head(20))
    st.write("Missing values:")
    st.write(filtered_df.isnull().sum())

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Trips by Hour")
        trips_hour = filtered_df.groupby('Hour').size().reset_index(name='Trips')
        st.plotly_chart(px.bar(trips_hour, x='Hour', y='Trips', title="Trips by Hour"), use_container_width=True)

        st.subheader("Trips by Weekday")
        trips_week = filtered_df.groupby('Weekday').size().reindex(
            ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']).fillna(0).reset_index(name='Trips')
        st.plotly_chart(px.bar(trips_week, x='Weekday', y='Trips', title="Trips by Weekday"), use_container_width=True)

    with col2:
        st.subheader("Purpose distribution")
        if 'PURPOSE' in filtered_df.columns:
            purpose_counts = filtered_df['PURPOSE'].value_counts().reset_index()
            purpose_counts.columns = ['Purpose','Count']
            st.plotly_chart(px.bar(purpose_counts, x='Purpose', y='Count', title="Trips by Purpose"), use_container_width=True)
        else:
            st.info("PURPOSE column not found")

    st.subheader("Frequently traveled routes (top 20)")
    routes = filtered_df.groupby(['START','STOP']).size().reset_index(name='Trips').sort_values('Trips', ascending=False).head(20)
    if not routes.empty:
        st.dataframe(routes)
    else:
        st.info("No route data to show")

# ---------------- Tab Forecast: Prophet + CNN + LSTM ----------------
with tab_forecast:
    st.header("Forecast Comparison — Prophet vs CNN vs LSTM")

    # Maintain state across reruns
    if 'results_table' not in st.session_state:
        st.session_state.results_table = pd.DataFrame(columns=['Model', 'RMSE', 'MAE', 'R²'])
    if 'prophet_results' not in st.session_state:
        st.session_state.prophet_results = None
    if 'cnn_future' not in st.session_state:
        st.session_state.cnn_future = None
    if 'lstm_future' not in st.session_state:
        st.session_state.lstm_future = None
    if 'ensemble_future' not in st.session_state:
        st.session_state.ensemble_future = None

    # Select forecasting target
    target = st.selectbox("Select Target for Forecasting", ["Trip Count", "Total Miles", "Avg Duration (min)"])
    daily = series_from_target(filtered_df, target)

    st.line_chart(daily.set_index('ds')['y'])
    st.caption(f"Data points available: {len(daily)}")

    if len(daily) < 30:
        st.warning("Not enough data for reliable forecasting. Please expand your date range.")
        st.stop()

    # Deep model hyperparams
    window = st.number_input("Window Size (for Deep Learning Models)", 7, 60, 28)
    epochs = st.number_input("Training Epochs", 10, 300, 160)
    forecast_horizon = st.slider("Days to Forecast Ahead", 3, 30, 7)

    # --- Prophet: use train/test split to evaluate fairly ---
    if PROPHET_AVAILABLE:
        if st.button("Run Prophet Forecast (train/test)"):
            # split train/test in time order
            split_idx = int(len(daily) * 0.8)
            train_df = daily.iloc[:split_idx].reset_index(drop=True)
            test_df = daily.iloc[split_idx:].reset_index(drop=True)

            pm = Prophet(daily_seasonality=True)
            pm.fit(train_df)
            future = pm.make_future_dataframe(periods=len(test_df))
            forecast = pm.predict(future)

            # obtain out-of-sample forecast for test period
            forecast_test = forecast[['ds','yhat']].iloc[-len(test_df):].reset_index(drop=True)
            y_true = test_df['y'].values
            y_pred = forecast_test['yhat'].values

            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)

            # store and show
            st.session_state.prophet_results = forecast_test.tail(forecast_horizon).rename(columns={'yhat':'yhat'})
            # store evaluation arrays so we can compute ensemble metrics later
            st.session_state.prophet_eval = {
                'y_true': np.array(y_true).astype(float),
                'y_pred': np.array(y_pred).astype(float),
                'rmse': float(rmse),
                'mae': float(mae),
                'r2': float(r2)
            }
            st.session_state.results_table = pd.concat([
                st.session_state.results_table,
                pd.DataFrame([['Prophet', rmse, mae, r2]], columns=['Model', 'RMSE', 'MAE', 'R²'])
            ], ignore_index=True)

            st.success(f"Prophet trained. Test RMSE: {rmse:.3f}, R²: {r2:.3f}")

    else:
        st.warning("Prophet not installed. Run `pip install prophet` to enable.")

    # --- Prepare data for CNN and LSTM (with optional detrending) ---
    y_vals = daily['y'].values.copy()
    if detrend:
        rolling = pd.Series(y_vals).rolling(7, min_periods=1).mean().values
        y_model = (y_vals - rolling).astype(float)  # residual series for DL
        # keep last raw window for forecasting later (we will add back trend)
        last_window_raw = y_vals[-int(window):]
        last_trend = rolling[-int(window):]
    else:
        y_model = y_vals.astype(float)
        last_window_raw = y_vals[-int(window):]
        last_trend = np.zeros_like(last_window_raw)

    prepared = prepare_window_train_test(y_model, int(window), test_ratio=0.2)
    if prepared:
        X_train, X_test, y_train, y_test, scaler = prepared

        # build and train CNN
        if st.button("Train & Forecast CNN"):
            cnn = build_cnn(int(window))
            es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            history = cnn.fit(X_train, y_train, epochs=int(epochs), batch_size=16, validation_data=(X_test, y_test), callbacks=[es], verbose=0)
            rmse, mae, r2, y_t_inv, y_p_inv = evaluate_model_keras(cnn, X_test, y_test, scaler)
            # iterative forecast from last raw window: if detrend was applied, add trend back
            cnn_future_raw = iterative_forecast_keras(cnn, last_window_raw, int(forecast_horizon), scaler, int(window))
            if detrend:
                # reapply simple trend extrapolation: use last known trend value for each future step
                trend_last = float(pd.Series(y_vals).rolling(7, min_periods=1).mean().iloc[-1])
                cnn_future = (cnn_future_raw + trend_last).round(2)
            else:
                cnn_future = np.round(cnn_future_raw, 2)

            st.session_state.cnn_future = cnn_future
            # store evaluation arrays for ensemble evaluation (if needed)
            st.session_state.cnn_eval = {
                'y_true': np.array(y_t_inv).astype(float),
                'y_pred': np.array(y_p_inv).astype(float),
                'rmse': float(rmse),
                'mae': float(mae),
                'r2': float(r2)
            }
            st.session_state.results_table = pd.concat([
                st.session_state.results_table,
                pd.DataFrame([['CNN', rmse, mae, r2]], columns=['Model', 'RMSE', 'MAE', 'R²'])
            ], ignore_index=True)
            st.success(f"CNN trained. Test RMSE: {rmse:.3f}, R²: {r2:.3f}")

        # build and train LSTM
        if st.button("Train & Forecast LSTM"):
            lstm = build_stacked_lstm(int(window))
            es = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)
            history = lstm.fit(X_train, y_train, epochs=int(epochs), batch_size=16, validation_data=(X_test, y_test), callbacks=[es], verbose=0)
            rmse, mae, r2, y_t_inv, y_p_inv = evaluate_model_keras(lstm, X_test, y_test, scaler)
            lstm_future_raw = iterative_forecast_keras(lstm, last_window_raw, int(forecast_horizon), scaler, int(window))
            if detrend:
                trend_last = float(pd.Series(y_vals).rolling(7, min_periods=1).mean().iloc[-1])
                lstm_future = (lstm_future_raw + trend_last).round(2)
            else:
                lstm_future = np.round(lstm_future_raw, 2)

            st.session_state.lstm_future = lstm_future
            st.session_state.results_table = pd.concat([
                st.session_state.results_table,
                pd.DataFrame([['LSTM', rmse, mae, r2]], columns=['Model', 'RMSE', 'MAE', 'R²'])
            ], ignore_index=True)
            # store evaluation arrays for ensemble evaluation (if needed)
            st.session_state.lstm_eval = {
                'y_true': np.array(y_t_inv).astype(float),
                'y_pred': np.array(y_p_inv).astype(float),
                'rmse': float(rmse),
                'mae': float(mae),
                'r2': float(r2)
            }
            st.success(f"LSTM trained. Test RMSE: {rmse:.3f}, R²: {r2:.3f}")

    else:
        st.warning("Not enough sequential data for CNN/LSTM with the chosen window size.")

    # --- Ensemble (Prophet + LSTM) optional ---
    if st.button("Make Ensemble (Prophet + DL)"):
        model_list = []
        if st.session_state.prophet_results is not None:
            model_list.append(('Prophet', st.session_state.prophet_results['yhat'].values[-forecast_horizon:]))
        if st.session_state.lstm_future is not None:
            model_list.append(('LSTM', np.array(st.session_state.lstm_future)))
        if st.session_state.cnn_future is not None:
            model_list.append(('CNN', np.array(st.session_state.cnn_future)))

        if len(model_list) >= 2:
            # simple average ensemble (equal weights) for future forecasts
            preds = np.array([m[1] for m in model_list])
            ensemble = np.mean(preds, axis=0).round(2)
            st.session_state.ensemble_future = ensemble
            st.success("Ensemble created from available model forecasts.")

            # Try to compute ensemble metrics on test sets if evaluation predictions are available
            eval_sources = []
            for key in ('prophet_eval', 'lstm_eval', 'cnn_eval'):
                if key in st.session_state and st.session_state[key] is not None:
                    ev = st.session_state[key]
                    # ensure arrays are present and non-empty
                    if 'y_true' in ev and len(ev['y_true']) > 0 and 'y_pred' in ev and len(ev['y_pred']) > 0:
                        eval_sources.append(ev)

            if len(eval_sources) >= 2:
                # align to the shortest evaluation length (take last values)
                min_len = min([len(ev['y_true']) for ev in eval_sources])
                preds_matrix = np.array([ev['y_pred'][-min_len:] for ev in eval_sources])
                # use first eval source's y_true as reference
                y_true_ref = eval_sources[0]['y_true'][-min_len:]
                ensemble_pred_eval = preds_matrix.mean(axis=0)
                # compute metrics
                ens_rmse = float(np.sqrt(mean_squared_error(y_true_ref, ensemble_pred_eval)))
                ens_mae = float(mean_absolute_error(y_true_ref, ensemble_pred_eval))
                ens_r2 = float(r2_score(y_true_ref, ensemble_pred_eval))
                # store ensemble eval
                st.session_state.ensemble_eval = {
                    'y_true': np.array(y_true_ref).astype(float),
                    'y_pred': np.array(ensemble_pred_eval).astype(float),
                    'rmse': ens_rmse,
                    'mae': ens_mae,
                    'r2': ens_r2
                }
                # append numeric metrics to results table
                st.session_state.results_table = pd.concat([
                    st.session_state.results_table,
                    pd.DataFrame([['Ensemble', ens_rmse, ens_mae, ens_r2]], columns=['Model', 'RMSE', 'MAE', 'R²'])
                ], ignore_index=True)
            else:
                # no detailed eval arrays available; try a sensible fallback:
                # compute ensemble metrics as the mean of available numeric metrics
                model_names = [m[0] for m in model_list]
                df_res = st.session_state.results_table.copy()
                # select numeric rows for models that participated in the ensemble
                numeric = df_res[df_res['Model'].isin(model_names) & df_res['RMSE'].notnull()]
                if not numeric.empty:
                    ens_rmse = float(numeric['RMSE'].astype(float).mean())
                    ens_mae = float(numeric['MAE'].astype(float).mean()) if 'MAE' in numeric.columns and numeric['MAE'].notnull().any() else np.nan
                    ens_r2 = float(numeric['R²'].astype(float).mean()) if 'R²' in numeric.columns and numeric['R²'].notnull().any() else np.nan
                else:
                    ens_rmse = ens_mae = ens_r2 = np.nan

                st.session_state.results_table = pd.concat([
                    st.session_state.results_table,
                    pd.DataFrame([['Ensemble', ens_rmse, ens_mae, ens_r2]], columns=['Model', 'RMSE', 'MAE', 'R²'])
                ], ignore_index=True)
        else:
            st.warning("Need at least two model forecasts (Prophet + one DL) to build ensemble.")

    # --- Combined Forecast Visualization ---
    st.subheader("📈 Combined Forecast Comparison")
    future_dates = pd.date_range(start=daily['ds'].max() + timedelta(days=1), periods=int(forecast_horizon))
    combined = pd.DataFrame({"Date": future_dates})

    if st.session_state.prophet_results is not None:
        # For display align lengths if prophet output longer
        prophet_vals = st.session_state.prophet_results['yhat'].values
        combined["Prophet"] = prophet_vals[-int(forecast_horizon):]
    if st.session_state.cnn_future is not None:
        combined["CNN"] = np.array(st.session_state.cnn_future).round(2)
    if st.session_state.lstm_future is not None:
        combined["LSTM"] = np.array(st.session_state.lstm_future).round(2)
    if st.session_state.ensemble_future is not None:
        combined["Ensemble"] = np.array(st.session_state.ensemble_future).round(2)

    if len(combined.columns) > 1:
        melted = combined.melt(id_vars="Date", var_name="Model", value_name="Predicted")
        fig = px.line(melted, x="Date", y="Predicted", color="Model", markers=True,
                      title="Prophet vs CNN vs LSTM Forecast Comparison")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(combined.set_index("Date").round(2))
    else:
        st.info("Run at least one model to see their forecasts.")

    # --- Performance Comparison Table ---
    if not st.session_state.results_table.empty:
        df_perf = st.session_state.results_table.groupby("Model", as_index=False).mean().round(3)
        # highlight lowest RMSE (if numeric)
        if 'RMSE' in df_perf.columns and df_perf['RMSE'].notnull().any():
            best_model = df_perf.loc[df_perf['RMSE'].idxmin(), 'Model']
        else:
            best_model = None
        st.subheader("📊 Model Performance Comparison (on test sets)")
        st.plotly_chart(px.bar(df_perf, x="Model", y="RMSE", title="RMSE Comparison (lower is better)"), use_container_width=True)
        st.dataframe(df_perf)
        if best_model:
            st.success(f"🏆 Best Model: {best_model} (lowest RMSE)")
    else:
        st.info("Run models to populate the performance table.")

# ---------------- Tab ML: classical models ----------------
with tab_ml:
    st.header("Classical ML Models — predict Duration(min) from MILES & HOUR")
    ml_df = filtered_df.dropna(subset=['MILES','Hour','Duration(min)']).copy()
    if ml_df.empty or len(ml_df) < 10:
        st.warning("Not enough rows to train ML models. Adjust filters.")
    else:
        X = ml_df[['MILES','Hour']].values
        y = ml_df['Duration(min)'].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

        models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(random_state=SEED),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=SEED)
        }

        results = []
        for name, model in models.items():
            model.fit(X_train, y_train)
            yp = model.predict(X_test)
            rmse_val = float(np.sqrt(mean_squared_error(y_test, yp)))
            r2_val = float(r2_score(y_test, yp))
            results.append((name, rmse_val, r2_val))

        res_df = pd.DataFrame(results, columns=['Model','RMSE','R2']).sort_values('RMSE')
        st.subheader("Model comparison (lower RMSE better)")
        st.dataframe(res_df.style.format({"RMSE":"{:.2f}", "R2":"{:.2f}"}))

        best_model_name = res_df.iloc[0]['Model']
        st.success(f"Best model by RMSE: {best_model_name}")

        st.subheader("Predict duration for a new trip")
        miles_in = st.number_input("Miles (km value in MILES column)", value=5.0, min_value=0.0, step=0.1)
        hour_in = st.slider("Start Hour", 0, 23, 9)
        chosen = st.selectbox("Choose model", list(models.keys()))
        pred_val = models[chosen].predict(np.array([[miles_in, hour_in]]))[0]
        st.success(f"Predicted Duration ({chosen}): {pred_val:.2f} minutes")
