import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components
import base64
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

def display_copyable_table(title, dataframe):
    title_column, copy_column = st.columns([8, 1])
    title_column.subheader(title)
    table_text = dataframe.to_csv(sep="\t", index=False, lineterminator="\n")
    encoded_table = base64.b64encode(table_text.encode("utf-8")).decode("ascii")
    copy_component = f"""
        <button id="copy-table" title="Copy" aria-label="Copy"
            style="height: 20px; padding: 0 12px; border: 1px;
            border-radius: 8px; cursor: pointer; font-size: 12px;">
            Copy table
        </button>
        <script>
            const tableBytes = Uint8Array.from(atob("{encoded_table}"), c => c.charCodeAt(0));
            const tableText = new TextDecoder().decode(tableBytes);
            const copyButton = document.getElementById("copy-table");
            let resetLabelTimeout;

            copyButton.addEventListener("click", async () => {{
                try {{
                    await navigator.clipboard.writeText(tableText);
                }} catch (error) {{
                    const textarea = document.createElement("textarea");
                    textarea.value = tableText;
                    textarea.style.position = "fixed";
                    textarea.style.left = "-9999px";
                    document.body.appendChild(textarea);
                    textarea.select();
                    const copied = document.execCommand("copy");
                    textarea.remove();
                    if (!copied) {{
                        copyButton.textContent = "Copy failed";
                        return;
                    }}
                }}
                copyButton.textContent = "Copied";
                clearTimeout(resetLabelTimeout);
                resetLabelTimeout = setTimeout(() => {{
                    copyButton.textContent = "Copy table";
                }}, 2000);
            }});
        </script>
    """
    with copy_column:
        components.html(copy_component, height=48)
    st.dataframe(dataframe, use_container_width=True)

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
if "location_name_error_visible" not in st.session_state:
    st.session_state.location_name_error_visible = False

# -----------------------------
# Sidebar inputs
# -----------------------------
st.sidebar.header("1. Analysis Setting")

method = st.sidebar.radio(
    "Select Method",
    options=["UTide (Harmonic Analysis)", "RTide (Automated Response)"],
    index=0
)

st.sidebar.header("2. Input Parameters")

elevation_unit = st.sidebar.radio(
    "Water Elevation Unit",
    options=["m", "cm"]
)

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

# -----------------------------
# Main input area
# -----------------------------
location_name = st.text_input("Location Name")
if location_name.strip():
    st.session_state.location_name_error_visible = False

location_name_error = st.empty()
if st.session_state.location_name_error_visible:
    location_name_error.error("Please enter a location name.")

st.subheader(f"Paste Hourly Elevation Data ({elevation_unit})")

data_text = st.text_area(
    "One value per line, hourly interval",
    height=200,
    placeholder=(
        "123.4\n125.1\n128.0\n130.2"
        if elevation_unit == "cm"
        else "1.23\n1.25\n1.28\n1.30"
    )
)

if st.session_state.results is None:
    st.info("Paste elevation data, set parameters, then click **Run Analysis**.")

_, run_button_column = st.columns([8, 1.5])
run_button = run_button_column.button(
    "Run Analysis",
    type="primary",
    use_container_width=True
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

    if not location_name.strip():
        st.session_state.location_name_error_visible = True
        location_name_error.error("Please enter a location name.")
        st.stop()

    if not data_text.strip():
        st.error("Please paste elevation data.")
        st.stop()

    try:
        elev = pd.read_csv(
            io.StringIO(data_text),
            header=None,
            names=["elevation"]
        )["elevation"].astype(float).values
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
                "constituents": selected_constituents,
                "unit": elevation_unit,
                "location": location_name.strip()
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
                    "model": model,
                    "unit": elevation_unit,
                    "location": location_name.strip()
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
    result_unit = res.get("unit", elevation_unit)
    result_location = res.get("location", location_name.strip()) or "location"

    st.title(f"{method_used} Tidal Analysis Result")

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

    ax1.set_ylabel(f"Water Level ({result_unit})")
    ax1.set_title(f"Observed and {method_used}-Fitted Tide")
    ax1.legend(frameon=False, handlelength=3)
    ax1.grid(False)

    ax2.plot(time_val, residual, color="black", linestyle=":", linewidth=1.2)
    ax2.axhline(0, color="black", linestyle="--", linewidth=1.0)
    ax2.set_ylabel(f"Residual ({result_unit})")
    ax2.set_xlabel("Time")
    ax2.grid(False)

    plt.tight_layout()
    st.pyplot(fig)

    # -----------------------------
    # Method-specific results
    # -----------------------------
    if method_used == "UTide":
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

        coef_df = pd.DataFrame(
            rows,
            columns=["Constituent", f"Amplitude ({result_unit})", "Phase (deg)"]
        )

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
            f"Amplitude ({result_unit})": [np.nan],
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
    if method_used == "UTide":
        display_copyable_table("Harmonic Constituents", coef_df)

    datum_df = pd.DataFrame({
        "Datum": list(datums.keys()),
        f"Elevation MSL datum ({result_unit})": list(datums.values()),
        f"Elevation LWS datum ({result_unit})": [elevation - datums["LWS"] for elevation in datums.values()],
        "Description": datum_desc
    })
    display_copyable_table("Tidal Elevations", datum_df)

    # -----------------------------
    # Model Accuracy
    # -----------------------------
    st.subheader("Model Accuracy")
    rmse = np.sqrt(mean_squared_error(elev, fit))
    tidal_range = datums["HWS"] - datums["LWS"]
    rmse_percent = (rmse / tidal_range) * 100 if tidal_range != 0 else np.nan

    st.metric("RMSE", f"{rmse:.2f} {result_unit} / {rmse_percent:.2f} %")
    st.metric("Tidal Range", f"{tidal_range:.2f} {result_unit}")

    # -----------------------------
    # Excel download
    # -----------------------------
    output_df = pd.DataFrame({
        "datetime": pd.to_datetime(time_val).tz_localize(None),
        f"observed_{result_unit}": elev,
        f"fit_{result_unit}": fit,
        f"residual_{result_unit}": residual
    })
    excel_data = io.BytesIO()
    with pd.ExcelWriter(excel_data, engine="openpyxl") as writer:
        output_df.to_excel(writer, sheet_name="utide_tidal_timeseries", index=False)
        coef_df.to_excel(writer, sheet_name="harmonic_constituents", index=False)
        datum_df.to_excel(writer, sheet_name="tidal_elevations", index=False)

    st.subheader("Download Analysis Results")
    safe_location = re.sub(
        r"[^\w.-]+",
        "-",
        result_location,
        flags=re.UNICODE
    ).strip("-.")
    safe_location = safe_location or "location"
    st.markdown(
        """
        <style>
        div[data-testid="stDownloadButton"] button {
            background-color: #28a745;
            border-color: #28a745;
            color: white;
        }
        div[data-testid="stDownloadButton"] button:hover {
            background-color: #218838;
            border-color: #1e7e34;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.download_button(
        label="📥 Download Excel Workbook (3 sheets)",
        data=excel_data.getvalue(),
        file_name=(
            f"{safe_location}-"
            f"{method_used.lower()}_tidal_timeseries.xlsx"
        ),
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
