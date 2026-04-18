import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import io
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Flow Mixing Analysis",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #0d1b2e 50%, #0a1520 100%);
    }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #001f3f 0%, #003366 50%, #002244 100%);
        border: 1px solid rgba(0, 150, 255, 0.3);
        border-radius: 16px;
        padding: 32px 40px;
        margin-bottom: 28px;
        box-shadow: 0 8px 32px rgba(0, 100, 255, 0.2), inset 0 1px 0 rgba(255,255,255,0.05);
    }

    .main-header h1 {
        color: #e8f4ff;
        font-size: 2.0rem;
        font-weight: 700;
        margin: 0 0 8px 0;
        letter-spacing: -0.5px;
    }

    .main-header .subtitle {
        color: #7fb3d3;
        font-size: 0.95rem;
        font-weight: 400;
        margin: 0;
    }

    .main-header .special-note {
        display: inline-block;
        background: rgba(255, 200, 0, 0.12);
        border: 1px solid rgba(255, 200, 0, 0.35);
        color: #ffd700;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
        margin-top: 14px;
        letter-spacing: 0.3px;
    }

    /* Cards */
    .card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.09);
        border-radius: 14px;
        padding: 24px;
        margin-bottom: 20px;
    }

    .card-title {
        color: #7fb3d3;
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-bottom: 18px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Input labels */
    label {
        color: #a8c8e8 !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
    }

    /* Text inputs */
    .stTextInput > div > div > input {
        background: rgba(0, 30, 60, 0.6) !important;
        border: 1px solid rgba(0, 120, 220, 0.35) !important;
        border-radius: 10px !important;
        color: #e8f4ff !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        text-align: center !important;
        padding: 12px 16px !important;
        transition: all 0.2s ease !important;
    }

    .stTextInput > div > div > input:focus {
        border-color: rgba(0, 160, 255, 0.7) !important;
        box-shadow: 0 0 0 3px rgba(0, 120, 255, 0.15) !important;
    }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #0066cc 0%, #0044aa 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 14px 32px !important;
        font-size: 1.0rem !important;
        font-weight: 600 !important;
        width: 100% !important;
        letter-spacing: 0.3px !important;
        box-shadow: 0 4px 20px rgba(0, 100, 255, 0.35) !important;
        transition: all 0.2s ease !important;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #0077ee 0%, #0055cc 100%) !important;
        box-shadow: 0 6px 28px rgba(0, 120, 255, 0.5) !important;
        transform: translateY(-1px) !important;
    }

    /* Download buttons */
    .stDownloadButton > button {
        background: rgba(0, 80, 160, 0.5) !important;
        color: #7fb3d3 !important;
        border: 1px solid rgba(0, 100, 200, 0.4) !important;
        border-radius: 10px !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
        width: 100% !important;
        padding: 10px 20px !important;
        transition: all 0.2s ease !important;
    }

    .stDownloadButton > button:hover {
        background: rgba(0, 100, 200, 0.7) !important;
        color: white !important;
        border-color: rgba(0, 150, 255, 0.6) !important;
    }

    /* Metrics */
    .metric-box {
        background: rgba(0, 40, 80, 0.5);
        border: 1px solid rgba(0, 100, 200, 0.25);
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 10px;
        text-align: center;
    }

    .metric-label {
        color: #6a9fc0;
        font-size: 0.76rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 4px;
    }

    .metric-value {
        color: #4db8ff;
        font-size: 1.4rem;
        font-weight: 700;
    }

    /* Status badges */
    .badge-success {
        display: inline-block;
        background: rgba(0, 180, 100, 0.15);
        border: 1px solid rgba(0, 200, 100, 0.4);
        color: #00e676;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
    }

    .badge-warning {
        display: inline-block;
        background: rgba(255, 160, 0, 0.12);
        border: 1px solid rgba(255, 180, 0, 0.35);
        color: #ffb300;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
    }

    /* Error boxes */
    .error-box {
        background: rgba(200, 0, 0, 0.12);
        border: 1px solid rgba(255, 80, 80, 0.4);
        border-radius: 10px;
        padding: 12px 16px;
        color: #ff6b6b;
        font-size: 0.9rem;
        margin-top: 6px;
    }

    /* Footer */
    .footer {
        margin-top: 40px;
        padding: 24px 32px;
        border-top: 1px solid rgba(255,255,255,0.07);
        text-align: center;
    }

    .footer-dev {
        color: #4a7fa0;
        font-size: 0.82rem;
    }

    .footer-dev span {
        color: #7fb3d3;
        font-weight: 600;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: rgba(5, 15, 30, 0.97) !important;
        border-right: 1px solid rgba(0, 80, 160, 0.25);
    }

    /* Alerts */
    .stAlert {
        border-radius: 10px !important;
    }

    /* Divider */
    hr {
        border-color: rgba(255,255,255,0.07) !important;
    }
</style>
""", unsafe_allow_html=True)


# ─── MODEL LOADING ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load or train the model. Returns (scaler_X, pca, best_model) or None."""
    model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            bundle = pickle.load(f)
        return bundle["scaler_X"], bundle["pca"], bundle["best_model"]
    return None


def predict_matrix(scaler_X, pca, best_model, i1, i2, i3, i4):
    """Run model prediction and return 15×15 numpy matrix."""
    x = pd.DataFrame([[i1, i2, i3, i4]], columns=['I1', 'I2', 'I3', 'I4'])
    x_scaled = scaler_X.transform(x)
    y_reduced_pred = best_model.predict(x_scaled)
    y_full = pca.inverse_transform(y_reduced_pred)
    matrix = y_full.reshape(15, 15)
    return matrix


# ─── INPUT VALIDATION ─────────────────────────────────────────────────────────
def validate_decimal(value_str, field_name):
    """Validate input is a legal decimal number. Returns (float, error_str)."""
    if value_str.strip() == "":
        return None, f"{field_name}: Field cannot be empty."
    try:
        val = float(value_str.strip())
        if np.isnan(val) or np.isinf(val):
            return None, f"{field_name}: Must be a finite number."
        if val < 0:
            return None, f"{field_name}: Value must be non-negative."
        return val, None
    except ValueError:
        return None, f"{field_name}: '{value_str.strip()}' is not a valid decimal number."


# ─── PLOT GENERATION ──────────────────────────────────────────────────────────
def generate_heatmap_fig(matrix, inputs):
    """Generate a high-quality heatmap figure."""
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor('#0d1b2e')
    ax.set_facecolor('#0d1b2e')

    vmin, vmax = matrix.min(), matrix.max()

    # jet colormap matches the reference image (blue→cyan→green→yellow→orange→red)
    # origin='lower' puts row 0 at bottom so both axes start from 0 at origin
    im = ax.imshow(
        matrix,
        cmap='jet',
        interpolation='nearest',
        aspect='equal',
        vmin=vmin,
        vmax=vmax,
        origin='lower'
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Concentration Value', color='#a8c8e8', fontsize=10, labelpad=10)
    cbar.ax.yaxis.set_tick_params(color='#a8c8e8', labelsize=8)
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#a8c8e8')
    cbar.outline.set_edgecolor((1, 1, 1, 0.15))

    # Both axes: 0 → 14 (aligned — same origin point at bottom-left)
    ticks = list(range(15))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([str(i) for i in ticks], color='#7fb3d3', fontsize=7.5)
    ax.set_yticklabels([str(i) for i in ticks], color='#7fb3d3', fontsize=7.5)

    for spine in ax.spines.values():
        spine.set_edgecolor((0/255, 100/255, 200/255, 0.3))

    ax.set_xlabel("Receiver", color='#a8c8e8', fontsize=11, fontweight='600', labelpad=10)
    ax.set_ylabel("Transmitter", color='#a8c8e8', fontsize=11, fontweight='600', labelpad=10)
    ax.tick_params(colors='#7fb3d3', length=4)

    title = (f"Tracer Concentration Distribution\n"
             f"I₁={inputs[0]:.4f}  I₂={inputs[1]:.4f}  I₃={inputs[2]:.4f}  I₄={inputs[3]:.4f}")
    ax.set_title(title, color='#c8e0f4', fontsize=11, fontweight='600', pad=14)

    plt.tight_layout()
    return fig


def generate_heatmap_png(matrix, inputs):
    fig = generate_heatmap_fig(matrix, inputs)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=180, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close(fig)
    return buf.getvalue()


def generate_excel_bytes(matrix, inputs):
    """Generate Excel file with the 15×15 matrix + metadata."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Matrix sheet
        df_matrix = pd.DataFrame(
            matrix,
            index=[f"Row {i+1}" for i in range(15)],
            columns=[f"Col {i+1}" for i in range(15)]
        )
        df_matrix.to_excel(writer, sheet_name='Concentration Matrix')

        # Metadata sheet
        meta = {
            'Parameter': ['Inlet 1 (I1)', 'Inlet 2 (I2)', 'Inlet 3 (I3)', 'Inlet 4 (I4)',
                          'Matrix Size', 'Min Value', 'Max Value', 'Mean Value', 'Std Dev',
                          'Application', 'Note'],
            'Value': [inputs[0], inputs[1], inputs[2], inputs[3],
                      '15 × 15', round(matrix.min(), 6), round(matrix.max(), 6),
                      round(matrix.mean(), 6), round(matrix.std(), 6),
                      'Flow Mixing Analysis – Steady State',
                      'Tracer is constant']
        }
        pd.DataFrame(meta).to_excel(writer, sheet_name='Analysis Info', index=False)

        # Style matrix sheet
        ws = writer.sheets['Concentration Matrix']
        from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
        from openpyxl.utils import get_column_letter

        header_fill = PatternFill("solid", fgColor="001f3f")
        header_font = Font(color="7fb3d3", bold=True, size=10)
        thin = Side(style='thin', color='003366')
        border = Border(left=thin, right=thin, top=thin, bottom=thin)

        for cell in ws[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal='center')
            cell.border = border

        for row in ws.iter_rows(min_row=2):
            for cell in row:
                if cell.column == 1:
                    cell.fill = header_fill
                    cell.font = header_font
                cell.alignment = Alignment(horizontal='center')
                cell.border = border
                if cell.column > 1 and cell.value is not None:
                    try:
                        v = float(cell.value)
                        norm = (v - matrix.min()) / max(matrix.max() - matrix.min(), 1e-9)
                        r = int(255)
                        g = int(255 * (1 - norm * 0.7))
                        b = int(255 * (1 - norm * 0.9))
                        cell.fill = PatternFill("solid", fgColor=f"{r:02X}{g:02X}{b:02X}")
                    except Exception:
                        pass

        for col in ws.columns:
            ws.column_dimensions[get_column_letter(col[0].column)].width = 10

    output.seek(0)
    return output.getvalue()


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 10px 0 20px 0; border-bottom: 1px solid rgba(255,255,255,0.08); margin-bottom: 20px;">
        <div style="color:#7fb3d3; font-size:0.7rem; text-transform:uppercase; letter-spacing:1.2px; margin-bottom:8px;">Application Info</div>
        <div style="color:#e8f4ff; font-weight:600; font-size:0.9rem;">Flow Mixing Analysis</div>
        <div style="color:#4a7fa0; font-size:0.78rem; margin-top:2px;">Steady State Prediction Engine</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="color:#6a9fc0; font-size:0.75rem; text-transform:uppercase; letter-spacing:1px; margin-bottom:10px;">Model Status</div>
    """, unsafe_allow_html=True)

    model_bundle = load_model()
    if model_bundle:
        st.markdown('<span class="badge-success">✓ Model Loaded</span>', unsafe_allow_html=True)
        st.markdown("""
        <div style="margin-top:12px; color:#4a7fa0; font-size:0.8rem; line-height:1.6;">
            <div>• Ridge Regression</div>
            <div>• PCA Dimensionality Reduction</div>
            <div>• StandardScaler Normalized</div>
            <div>• Output: 15×15 Matrix</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-warning">⚠ Model File Missing</span>', unsafe_allow_html=True)
        st.markdown("""
        <div class="error-box" style="margin-top:12px;">
            <b>model.pkl not found.</b><br><br>
            Place <code>model.pkl</code> in the app directory.<br><br>
            Generate it from your notebook with:<br>
            <code style="font-size:0.75rem;">
            import pickle<br>
            with open('model.pkl','wb') as f:<br>
            &nbsp;&nbsp;pickle.dump({'scaler_X': scaler_X, 'pca': pca, 'best_model': best_model}, f)
            </code>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="color:#6a9fc0; font-size:0.75rem; text-transform:uppercase; letter-spacing:1px; margin-bottom:10px;">How To Use</div>
    <div style="color:#4a7fa0; font-size:0.8rem; line-height:1.8;">
        1️⃣ Enter 4 inlet flow values<br>
        2️⃣ Click <b style="color:#7fb3d3">Run Analysis</b><br>
        3️⃣ View heatmap & matrix<br>
        4️⃣ Download results
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="color:#6a9fc0; font-size:0.75rem; text-transform:uppercase; letter-spacing:1px; margin-bottom:8px;">Input Guidelines</div>
    <div style="color:#4a7fa0; font-size:0.8rem; line-height:1.7;">
        • Enter decimal numbers only<br>
        • Values must be non-negative<br>
        • Example: <b style="color:#7fb3d3">0.05</b>, <b style="color:#7fb3d3">0.25</b>, <b style="color:#7fb3d3">1.0</b><br>
        • No text, symbols, or negatives
    </div>
    """, unsafe_allow_html=True)


# ─── MAIN CONTENT ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>⚛️ Flow Mixing Analysis for Steady State</h1>
    <p class="subtitle">Machine Learning Prediction Model &nbsp;•&nbsp; Nuclear Reactor Mixing &nbsp;•&nbsp; 15×15 Concentration Matrix</p>
    <div class="special-note">📌 Special Note: Tracer is constant — tracer number will be added later</div>
</div>
""", unsafe_allow_html=True)


# ─── INPUT SECTION ────────────────────────────────────────────────────────────
st.markdown("""
<div class="card">
    <div class="card-title">🔢 Inlet Flow Inputs</div>
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    i1_str = st.text_input("Inlet 1 (I₁)", value="", placeholder="e.g. 0.05",
                            help="Enter a non-negative decimal number for Inlet 1 flow rate")

with col2:
    i2_str = st.text_input("Inlet 2 (I₂)", value="", placeholder="e.g. 0.06",
                            help="Enter a non-negative decimal number for Inlet 2 flow rate")

with col3:
    i3_str = st.text_input("Inlet 3 (I₃)", value="", placeholder="e.g. 0.07",
                            help="Enter a non-negative decimal number for Inlet 3 flow rate")

with col4:
    i4_str = st.text_input("Inlet 4 (I₄)", value="", placeholder="e.g. 0.08",
                            help="Enter a non-negative decimal number for Inlet 4 flow rate")

# Validate on-the-fly
errors = []
vals = {}
for label, raw in [("Inlet 1", i1_str), ("Inlet 2", i2_str), ("Inlet 3", i3_str), ("Inlet 4", i4_str)]:
    if raw.strip():
        v, err = validate_decimal(raw, label)
        if err:
            errors.append(err)
        else:
            vals[label] = v

if errors:
    for e in errors:
        st.markdown(f'<div class="error-box">⚠ {e}</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

run_col, _ = st.columns([1, 3])
with run_col:
    run_btn = st.button("▶  Run Mixing Analysis", use_container_width=True)


# ─── PREDICTION & RESULTS ─────────────────────────────────────────────────────
if run_btn:
    # Full validation
    all_inputs = {}
    all_errors = []
    for label, raw in [("Inlet 1", i1_str), ("Inlet 2", i2_str),
                        ("Inlet 3", i3_str), ("Inlet 4", i4_str)]:
        v, err = validate_decimal(raw, label)
        if err:
            all_errors.append(err)
        else:
            all_inputs[label] = v

    if all_errors:
        st.error("**Please fix the following input errors before running:**")
        for e in all_errors:
            st.markdown(f"- {e}")
    elif model_bundle is None:
        st.error("**Model not loaded.** Please place `model.pkl` in the app directory and reload the page.")
    else:
        scaler_X, pca, best_model = model_bundle
        i1, i2, i3, i4 = (all_inputs["Inlet 1"], all_inputs["Inlet 2"],
                           all_inputs["Inlet 3"], all_inputs["Inlet 4"])
        inputs = [i1, i2, i3, i4]

        with st.spinner("Running prediction model…"):
            try:
                matrix = predict_matrix(scaler_X, pca, best_model, i1, i2, i3, i4)

                st.markdown("---")
                st.markdown("""
                <div style="color:#7fb3d3; font-size:0.78rem; text-transform:uppercase;
                            letter-spacing:1.2px; margin-bottom:20px;">
                    📊 Analysis Results
                </div>
                """, unsafe_allow_html=True)

                # Summary metrics
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.markdown(f"""<div class="metric-box">
                        <div class="metric-label">Min Value</div>
                        <div class="metric-value">{matrix.min():.4f}</div>
                    </div>""", unsafe_allow_html=True)
                with m2:
                    st.markdown(f"""<div class="metric-box">
                        <div class="metric-label">Max Value</div>
                        <div class="metric-value">{matrix.max():.4f}</div>
                    </div>""", unsafe_allow_html=True)
                with m3:
                    st.markdown(f"""<div class="metric-box">
                        <div class="metric-label">Mean Value</div>
                        <div class="metric-value">{matrix.mean():.4f}</div>
                    </div>""", unsafe_allow_html=True)
                with m4:
                    st.markdown(f"""<div class="metric-box">
                        <div class="metric-label">Std Deviation</div>
                        <div class="metric-value">{matrix.std():.4f}</div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Main results: heatmap + matrix
                left_col, right_col = st.columns([1.05, 1])

                with left_col:
                    st.markdown("""
                    <div style="color:#7fb3d3; font-size:0.78rem; text-transform:uppercase;
                                letter-spacing:1px; margin-bottom:12px;">🌡️ Concentration Heatmap</div>
                    """, unsafe_allow_html=True)
                    fig = generate_heatmap_fig(matrix, inputs)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

                with right_col:
                    st.markdown("""
                    <div style="color:#7fb3d3; font-size:0.78rem; text-transform:uppercase;
                                letter-spacing:1px; margin-bottom:12px;">📋 15×15 Concentration Matrix</div>
                    """, unsafe_allow_html=True)

                    df_display = pd.DataFrame(
                        np.round(matrix, 4),
                        index=[f"R{i+1:02d}" for i in range(15)],
                        columns=[f"C{i+1:02d}" for i in range(15)]
                    )

                    # Style the dataframe
                    def color_cells(val):
                        norm_val = (val - matrix.min()) / max(matrix.max() - matrix.min(), 1e-9)
                        r = int(255)
                        g = int(255 * (1 - norm_val * 0.75))
                        b = int(255 * (1 - norm_val * 0.9))
                        text = '#111' if norm_val < 0.5 else '#fff'
                        return f'background-color: rgb({r},{g},{b}); color: {text}; font-size: 10px;'

                    styled = df_display.style.applymap(color_cells).format("{:.4f}")
                    st.dataframe(styled, use_container_width=True, height=485)

                # ─── DOWNLOADS ────────────────────────────────────────────────
                st.markdown("---")
                st.markdown("""
                <div style="color:#7fb3d3; font-size:0.78rem; text-transform:uppercase;
                            letter-spacing:1px; margin-bottom:16px;">⬇️ Download Results</div>
                """, unsafe_allow_html=True)

                dl1, dl2, dl3 = st.columns(3)

                fname_suffix = f"I1_{i1}_I2_{i2}_I3_{i3}_I4_{i4}"

                with dl1:
                    png_bytes = generate_heatmap_png(matrix, inputs)
                    st.download_button(
                        label="🖼️  Download Heatmap (PNG)",
                        data=png_bytes,
                        file_name=f"heatmap_{fname_suffix}.png",
                        mime="image/png",
                        use_container_width=True
                    )

                with dl2:
                    excel_bytes = generate_excel_bytes(matrix, inputs)
                    st.download_button(
                        label="📊  Download Matrix (Excel)",
                        data=excel_bytes,
                        file_name=f"matrix_{fname_suffix}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )

                with dl3:
                    csv_df = pd.DataFrame(matrix,
                                          index=[f"Row_{i+1}" for i in range(15)],
                                          columns=[f"Col_{i+1}" for i in range(15)])
                    csv_bytes = csv_df.to_csv().encode('utf-8')
                    st.download_button(
                        label="📄  Download Matrix (CSV)",
                        data=csv_bytes,
                        file_name=f"matrix_{fname_suffix}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                st.success(f"✅ Analysis complete for I₁={i1:.4f}, I₂={i2:.4f}, I₃={i3:.4f}, I₄={i4:.4f}")

            except Exception as ex:
                st.error(f"**Prediction error:** {str(ex)}\n\nPlease verify your model.pkl is valid and inputs are in the expected range.")


# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <div class="footer-dev">
        Developed by <span>Gaurav Rai</span>
        &nbsp;&nbsp;|&nbsp;&nbsp;
        Under the guidance of <span>Dr. Hardik Kothadiya</span>
        &nbsp;&nbsp;|&nbsp;&nbsp;
        Flow Mixing Analysis · Steady State · Nuclear Reactor
    </div>
    <div style="color:#2a4a62; font-size:0.72rem; margin-top:6px;">
        Bare minimum data model &nbsp;•&nbsp; For research and analysis purposes only
    </div>
</div>
""", unsafe_allow_html=True)
