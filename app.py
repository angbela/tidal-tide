import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import io
import re
import contextlib
import random
import os

# Import tidal libraries
from utide import solve, reconstruct, ut_constants
try:
    import tensorflow as tf
    from rtide import RTide
except ImportError:
    st.error("RTide/TensorFlow libraries not found. Please install them using `pip install rtide tensorflow`.")
    RTide = None

from datetime import datetime, timedelta, date, time, timezone
from sklearn.metrics import mean_squared_error

def set_seed(seed=0):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if 'tf' in globals() or 'tf' in locals():
        tf.random.set_seed(seed)
        # Optional: Force single-threaded execution for absolute determinism 
        # (though it might slow down training)
        # os.environ['TF_DETERMINISTIC_OPS'] = '1'

# -----------------------------
# Streamlit config
# -----------------------------
st.set_page_config(page_title="Tidal Analysis (Unified)", layout="wide")

st.title("🌊 Unified Tidal Analysis")
st.markdown("Perform tidal analysis using either **Harmonic (UTide)** or **Automated Response (RTide)** methods.")

# -----------------------------
# Session state initialization
# -----------------------------
if "results" not in st.session_state:
    st.session_state.results = None

# -----------------------------
# Sidebar inputs
# -----------------------------
st.sidebar.header("Analysis Settings")

method = st.sidebar.radio(
    "Select Method",
    options=["UTide (Harmonic Analysis)", "RTide (Automated Response)"],
    index=0
)

st.sidebar.header("Input Parameters")

latitude = st.sidebar.number_input(
    "Latitude (deg)",
    value=-0.870403,
    format="%.6f"
)

longitude = st.sidebar.number_input(
    "Longitude (deg)",
    value=119.0,
    format="%.6f"
)

start_date = st.sidebar.date_input("Start date", value=date(2026, 1, 1))
start_time = st.sidebar.time_input("Start time", value=time(0, 0))

# RTide requires timezone-aware datetimes for astronomical calculations
if "RTide" in method:
    start_datetime = datetime.combine(start_date, start_time).replace(tzinfo=timezone.utc)
else:
    start_datetime = datetime.combine(start_date, start_time)

# UTide-specific constituents selection
if "UTide" in method:
    all_constituents = sorted([c.upper() for c in ut_constants.const.name])
    default_constituents = ['M2', 'S2', 'N2', 'K2', 'K1', 'O1', 'P1', 'M4', 'MS4']

    selected_constituents = st.sidebar.multiselect(
        "Select tidal constituents",
        options=all_constituents,
        default=[c for c in default_constituents if c in all_constituents]
    )

run_button = st.sidebar.button("Run Analysis")

# -----------------------------
# Main input area
# -----------------------------
st.subheader("Paste Hourly Elevation Data (cm)")

data_text = st.text_area(
    "One value per line, hourly interval",
    height=250,
    placeholder="123.4\n125.1\n128.0\n130.2"
)

# -----------------------------
# Progress capture class for RTide
# -----------------------------
class RTideProgress(io.StringIO):
    def __init__(self, placeholder):
        super().__init__()
        self.placeholder = placeholder
        self.current_epoch = 0
        self.total_epochs = 500

    def write(self, s):
        super().write(s)
        match = re.search(r'Epoch (\d+)/(\d+)', s)
        if match:
            self.current_epoch = int(match.group(1))
            self.total_epochs = int(match.group(2))
            self.placeholder.text(f"Training RTide... ({self.current_epoch}/{self.total_epochs})")

# -----------------------------
# Run analysis
# -----------------------------
if run_button:

    if not data_text.strip():
        st.error("Please paste elevation data.")
        st.stop()

    try:
        elev = pd.read_csv(
            io.StringIO(data_text),
            header=None,
            names=["elevation_cm"]
        )["elevation_cm"].astype(float).values
    except Exception as e:
        st.error(f"Error reading pasted data: {e}")
        st.stop()

    if "UTide" in method:
        # Generate hourly datetime array
        time_np = np.array(
            [start_datetime + timedelta(hours=i) for i in range(len(elev))],
            dtype="datetime64"
        )

        with st.spinner("Running UTide Harmonic Analysis..."):
            coef = solve(
                time_np,
                elev,
                lat=latitude,
                method="ols",
                conf_int="linear",
                constit=selected_constituents
            )
            tide_fit = reconstruct(time_np, coef)
            fit = tide_fit.h
            residual = elev - fit

            st.session_state.results = {
                "method": "UTide",
                "time": time_np,
                "observed": elev,
                "fit": fit,
                "residual": residual,
                "coef": coef,
                "constituents": selected_constituents
            }

    elif "RTide" in method:
        if RTide is None:
            st.error("RTide library is required for this method.")
            st.stop()

        # Generate hourly datetime array (timezone aware for RTide)
        time_np = pd.date_range(
            start=start_datetime,
            periods=len(elev),
            freq='h'
        )

        df = pd.DataFrame({'observations': elev}, index=time_np)
        
        try:
            progress_placeholder = st.empty()
            with st.spinner("Preparing RTide model..."):
                model = RTide(df, latitude, longitude)
                model.Prepare_Inputs(verbose=False)
                
                with contextlib.redirect_stdout(RTideProgress(progress_placeholder)):
                    model.Train(standard_epochs=500, verbose=True)
                
                progress_placeholder.success("RTide Training complete!")
                
                model.Predict(df)
                prediction_df = model.test_prediction_df
                fit = prediction_df['rtide'].values
                residual = elev - fit

                st.session_state.results = {
                    "method": "RTide",
                    "time": time_np,
                    "observed": elev,
                    "fit": fit,
                    "residual": residual,
                    "model": model
                }
        except Exception as e:
            st.error(f"Error during RTide analysis: {e}")
            st.stop()

# -----------------------------
# Plot & outputs (persistent)
# -----------------------------
if st.session_state.results is not None:
    res = st.session_state.results
    time_val = res["time"]
    elev = res["observed"]
    fit = res["fit"]
    residual = res["residual"]
    method_used = res["method"]

    # -----------------------------
    # Plot with residual
    # -----------------------------
    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(12, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]}
    )

    ax1.plot(time_val, elev, color="red", linewidth=2.4, label="Observed")
    ax1.plot(
        time_val, fit,
        color="blue",
        linestyle=(0, (5, 3)),
        linewidth=1.4,
        marker="o",
        markersize=5,
        markerfacecolor="white",
        markeredgecolor="black",
        markeredgewidth=1.2,
        markevery=24,
        label=f"{method_used} Fit"
    )

    ax1.set_ylabel("Water Level (cm)")
    ax1.set_title(f"Observed and {method_used}-Fitted Tide")
    ax1.legend(frameon=False, handlelength=3)
    ax1.grid(False)

    ax2.plot(time_val, residual, color="black", linestyle=":", linewidth=1.2)
    ax2.axhline(0, color="black", linestyle="--", linewidth=1.0)
    ax2.set_ylabel("Residual (cm)")
    ax2.set_xlabel("Time")
    ax2.grid(False)

    plt.tight_layout()
    st.pyplot(fig)

    # -----------------------------
    # Method-specific results
    # -----------------------------
    if method_used == "UTide":
        st.subheader("Harmonic Constituents")
        coef = res["coef"]
        names = [n.upper() for n in coef.name]
        amplitudes = coef.A
        phases = coef.g

        rows = []
        for c in res["constituents"]:
            if c in names:
                i = names.index(c)
                rows.append([c, amplitudes[i], phases[i]])
            else:
                rows.append([c, np.nan, np.nan])
        rows.append(["S0", coef.mean, 0.0])

        coef_df = pd.DataFrame(rows, columns=["Constituent", "Amplitude (cm)", "Phase (deg)"])
        st.dataframe(coef_df, use_container_width=True)

        # Datums for UTide
        MSL = coef.mean if hasattr(coef, 'mean') else np.mean(elev)
        A_M2 = amplitudes[names.index('M2')] if 'M2' in names else 0
        A_S2 = amplitudes[names.index('S2')] if 'S2' in names else 0
        A_K1 = amplitudes[names.index('K1')] if 'K1' in names else 0
        A_O1 = amplitudes[names.index('O1')] if 'O1' in names else 0
        Amp_spring = A_M2 + A_S2 + A_K1 + A_O1
        
        datums = {
            "HWS": MSL + Amp_spring,
            "MHWS": MSL + 0.707 * Amp_spring,
            "MHWL": MSL + 0.5 * (A_M2 + A_K1),
            "MSL": MSL,
            "MLWL": MSL - 0.5 * (A_M2 + A_K1),
            "MLWS": MSL - 0.707 * Amp_spring,
            "LWS": MSL - Amp_spring
        }
        datum_desc = [
            "Highest Water Springs", "Mean High Water Springs", "Mean High Water Level",
            "Mean Sea Level", "Mean Low Water Level", "Mean Low Water Springs", "Lowest Water Springs"
        ]

    else: # RTide
        st.subheader("RTide Model Summary")
        st.info("RTide uses an automated response method (ML-based) instead of fixed harmonic constituents.")
        coef_df = pd.DataFrame({
            "Constituent": ["Not applicable for RTide"],
            "Amplitude (cm)": [np.nan],
            "Phase (deg)": [np.nan]
        })
        
        # Datums for RTide (Statistical approximations)
        MSL = np.mean(elev)
        max_fit = np.nanmax(fit)
        min_fit = np.nanmin(fit)
        std_fit = np.nanstd(fit)
        
        datums = {
            "HWS": max_fit,
            "MHWS": MSL + 1.2 * std_fit,
            "MHWL": MSL + std_fit,
            "MSL": MSL,
            "MLWL": MSL - std_fit,
            "MLWS": MSL - 1.2 * std_fit,
            "LWS": min_fit
        }
        datum_desc = [
            "Highest Water Springs (Max Fit)", "Mean High Water Springs (Approx)", "Mean High Water Level (Approx)",
            "Mean Sea Level", "Mean Low Water Level (Approx)", "Mean Low Water Springs (Approx)", "Lowest Water Springs (Min Fit)"
        ]

    # -----------------------------
    # Tidal Datums table
    # -----------------------------
    st.subheader("Tidal Elevations")
    datum_df = pd.DataFrame({
        "Datum": list(datums.keys()),
        "Elevation (cm)": list(datums.values()),
        "Description": datum_desc
    })
    st.dataframe(datum_df, use_container_width=True)

    # -----------------------------
    # Model Accuracy
    # -----------------------------
    st.subheader("Model Accuracy")
    rmse = np.sqrt(mean_squared_error(elev, fit))
    tidal_range = np.nanmax(fit) - np.nanmin(fit)
    rmse_percent = (rmse / tidal_range) * 100 if tidal_range != 0 else np.nan

    st.metric("RMSE", f"{rmse:.2f} cm / {rmse_percent:.2f} %")
    st.metric("Tidal Range", f"{tidal_range:.2f} cm")

    # -----------------------------
    # Excel download
    # -----------------------------
    output_df = pd.DataFrame({
        "datetime": pd.to_datetime(time_val).tz_localize(None),
        "observed_cm": elev,
        "fit_cm": fit,
        "residual_cm": residual
    })
    excel_data = io.BytesIO()
    with pd.ExcelWriter(excel_data, engine="openpyxl") as writer:
        output_df.to_excel(writer, sheet_name="utide_tidal_timeseries", index=False)
        coef_df.to_excel(writer, sheet_name="harmonic_constituents", index=False)
        datum_df.to_excel(writer, sheet_name="tidal_elevations", index=False)

    st.subheader("Download Analysis Results")
    st.download_button(
        label="📥 Download Excel Workbook (3 sheets)",
        data=excel_data.getvalue(),
        file_name=f"{method_used.lower()}_tidal_timeseries.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

else:
    st.info("Paste elevation data, set parameters, then click **Run Analysis**.")
