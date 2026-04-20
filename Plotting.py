import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

st.set_page_config(page_title="Material Analysis Plotter", layout="wide")

st.title("Material Characterization Plotter")

DEFAULT_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
]

# ---------------------------------------------------------
# 1. PARSER FUNCTIONS
# ---------------------------------------------------------
def parse_rheometer_txt(file_content):
    try:
        decoded_content = file_content.decode("utf-8")
    except UnicodeDecodeError:
        try:
            decoded_content = file_content.decode("cp932") 
        except UnicodeDecodeError:
            decoded_content = file_content.decode("shift_jis", errors="ignore")

    lines = decoded_content.splitlines()
    intervals = {}
    current_interval = 0
    data_lines = []
    is_data_section = False
    
    for line in lines:
        line = line.strip()
        if line.startswith("Meas. Pts."):
            if data_lines:  
                df = pd.DataFrame(data_lines, columns=["Meas. Pts", "Raw Viscosity", "Shear Rate", "Shear Stress", "Strain", "Interval Time", "Torque"])
                df = df.apply(pd.to_numeric, errors='coerce').dropna()
                intervals[f"Interval {current_interval}"] = df
                data_lines = []
            current_interval += 1
            is_data_section = True
            continue
        if is_data_section and line.startswith("["):
            continue
        if is_data_section and line == "":
            is_data_section = False
            if data_lines:
                df = pd.DataFrame(data_lines, columns=["Meas. Pts", "Raw Viscosity", "Shear Rate", "Shear Stress", "Strain", "Interval Time", "Torque"])
                df = df.apply(pd.to_numeric, errors='coerce').dropna()
                intervals[f"Interval {current_interval}"] = df
                data_lines = []
            continue
        if is_data_section:
            values = line.split('\t')
            if len(values) >= 7:
                values = [v.replace(',', '') for v in values]
                data_lines.append(values[:7])

    if data_lines:
        df = pd.DataFrame(data_lines, columns=["Meas. Pts", "Raw Viscosity", "Shear Rate", "Shear Stress", "Strain", "Interval Time", "Torque"])
        df = df.apply(pd.to_numeric, errors='coerce').dropna()
        intervals[f"Interval {current_interval}"] = df
    return intervals

def decode_file(file_content):
    try:
        return file_content.decode("utf-8")
    except UnicodeDecodeError:
        try:
            return file_content.decode("cp932")
        except UnicodeDecodeError:
            return file_content.decode("shift_jis", errors="ignore")


# ---------------------------------------------------------
# MAIN NAVIGATION
# ---------------------------------------------------------
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Select Analysis Module:", ["Rheology (Anton Paar)", "X-Ray Diffraction (XRD)"])
st.sidebar.divider()

# =========================================================
# MODULE 1: RHEOLOGY
# =========================================================
if app_mode == "Rheology (Anton Paar)":
    st.header("Rheometer Analysis")
    
    st.sidebar.header("1. Upload Data")
    uploaded_files = st.sidebar.file_uploader("Upload Anton Paar .txt files", type=["txt"], accept_multiple_files=True)

    if uploaded_files:
        try:
            all_parsed_intervals = {}
            for file in uploaded_files:
                file_name = file.name
                parsed_intervals = parse_rheometer_txt(file.getvalue())
                for inv_name, df in parsed_intervals.items():
                    unique_name = f"{file_name} | {inv_name}"
                    all_parsed_intervals[unique_name] = df
            
            st.sidebar.header("2. Data Selection")
            selected_intervals = st.sidebar.multiselect(
                "Select Measurement Intervals", 
                list(all_parsed_intervals.keys()), 
                default=list(all_parsed_intervals.keys())
            )
            
            st.sidebar.header("3. Customize Legends & Colors")
            custom_labels = {}
            custom_colors = {}
            for i, unique_inv in enumerate(selected_intervals):
                col1, col2 = st.sidebar.columns([3, 1])
                with col1:
                    custom_labels[unique_inv] = st.text_input(f"Rename:", value=unique_inv, key=f"name_{unique_inv}")
                with col2:
                    default_color = DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
                    custom_colors[unique_inv] = st.color_picker("Color", value=default_color, key=f"color_{unique_inv}")
            
            st.sidebar.header("4. Viscosity Units")
            visc_unit = st.sidebar.selectbox("Convert Viscosity to:", ["Pa·s", "mPa·s (cP)", "Poise (P)"])

            plot_df = pd.DataFrame()
            for unique_inv in selected_intervals:
                temp_df = all_parsed_intervals[unique_inv].copy()
                base_viscosity = temp_df["Shear Stress"] / temp_df["Shear Rate"]
                if visc_unit == "Pa·s":
                    temp_df["Converted Viscosity"] = base_viscosity
                elif visc_unit == "mPa·s (cP)":
                    temp_df["Converted Viscosity"] = base_viscosity * 1000
                elif visc_unit == "Poise (P)":
                    temp_df["Converted Viscosity"] = base_viscosity * 10
                temp_df["Interval"] = unique_inv
                plot_df = pd.concat([plot_df, temp_df])

            if not plot_df.empty:
                st.sidebar.header("5. Plot Controls")
                plot_mode = st.sidebar.radio("Plot Layout", ["Anton Paar Dual-Axis (Visc & Stress)", "Single Variable"])
                
                x_scale = st.sidebar.radio("X-axis Scale", ["Logarithmic", "Linear"], horizontal=True)
                y_scale = st.sidebar.radio("Y-axis Scale", ["Logarithmic", "Linear"], horizontal=True)
                
                x_type = "log" if x_scale == "Logarithmic" else "linear"
                y_type = "log" if y_scale == "Logarithmic" else "linear"
                x_exp_format = "power" if x_type == "log" else "none"
                x_dtick = 1 if x_type == "log" else None
                y_exp_format = "power" if y_type == "log" else "none"
                y_dtick = 1 if y_type == "log" else None

                st.sidebar.header("6. Filter Data Range (Shear Rate)")
                min_x = float(plot_df["Shear Rate"].min())
                max_x = float(plot_df["Shear Rate"].max())
                
                # Dynamically set the formatting of the input box based on the X-axis scale toggle
                dynamic_input_format = "%.3e" if x_scale == "Logarithmic" else "%.3f"
                
                col1, col2 = st.sidebar.columns(2)
                with col1:
                    user_min_x = st.number_input("Min Shear Rate", value=min_x, format=dynamic_input_format)
                with col2:
                    user_max_x = st.number_input("Max Shear Rate", value=max_x, format=dynamic_input_format)

                filtered_df = plot_df[(plot_df["Shear Rate"] >= user_min_x) & (plot_df["Shear Rate"] <= user_max_x)]
                st.caption("Hover over the top right of the graph and click the 'Camera' icon to download a high-resolution PNG for presentations.")
                
                if not filtered_df.empty:
                    if plot_mode == "Anton Paar Dual-Axis (Visc & Stress)":
                        fig = make_subplots(specs=[[{"secondary_y": True}]])
                        for unique_inv in selected_intervals:
                            interval_data = filtered_df[filtered_df["Interval"] == unique_inv]
                            display_name = custom_labels[unique_inv]
                            user_color = custom_colors[unique_inv]
                            
                            fig.add_trace(go.Scatter(x=interval_data["Shear Rate"], y=interval_data["Converted Viscosity"], name=f"{display_name} (Viscosity)", mode='lines+markers', marker_symbol='diamond', line=dict(color=user_color)), secondary_y=False)
                            fig.add_trace(go.Scatter(x=interval_data["Shear Rate"], y=interval_data["Shear Stress"], name=f"{display_name} (Stress)", mode='lines+markers', marker_symbol='square', line=dict(color=user_color, dash='dot')), secondary_y=True)
                        
                        fig.update_layout(plot_bgcolor='white', hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                        fig.update_xaxes(title_text="Shear Rate [1/s]", type=x_type, exponentformat=x_exp_format, dtick=x_dtick, showgrid=True, gridwidth=1, gridcolor='LightGray', ticks="inside", showline=True, linewidth=1, linecolor='black', mirror=True)
                        fig.update_yaxes(title_text=f"Viscosity [{visc_unit}]", type=y_type, exponentformat=y_exp_format, dtick=y_dtick, secondary_y=False, showgrid=True, gridwidth=1, gridcolor='LightGray', ticks="inside", showline=True, linewidth=1, linecolor='black')
                        fig.update_yaxes(title_text="Shear Stress [Pa]", type=y_type, exponentformat=y_exp_format, dtick=y_dtick, secondary_y=True, showgrid=False, ticks="inside", showline=True, linewidth=1, linecolor='black')
                    else:
                        available_cols = ["Shear Rate", "Shear Stress", "Converted Viscosity", "Strain", "Interval Time", "Torque"]
                        x_col = st.sidebar.selectbox("X-axis", available_cols, index=0) 
                        y_col = st.sidebar.selectbox("Y-axis", available_cols, index=2) 
                        
                        display_df = filtered_df.copy()
                        display_df["Interval"] = display_df["Interval"].map(custom_labels)
                        color_map = {custom_labels[inv]: custom_colors[inv] for inv in selected_intervals}
                        
                        fig = px.line(display_df, x=x_col, y=y_col, color="Interval", color_discrete_map=color_map, markers=True, symbol="Interval")
                        
                        def get_axis_label(col):
                            if col == "Converted Viscosity": return f"Viscosity [{visc_unit}]"
                            if col == "Shear Rate": return "Shear Rate [1/s]"
                            if col == "Shear Stress": return "Shear Stress [Pa]"
                            if col == "Strain": return "Strain [%]"
                            if col == "Interval Time": return "Time [s]"
                            if col == "Torque": return "Torque [mNm]"
                            return col
                        
                        fig.update_xaxes(title_text=get_axis_label(x_col), type=x_type, exponentformat=x_exp_format, dtick=x_dtick, ticks="inside", showline=True, linecolor='black', mirror=True)
                        fig.update_yaxes(title_text=get_axis_label(y_col), type=y_type, exponentformat=y_exp_format, dtick=y_dtick, ticks="inside", showline=True, linecolor='black', mirror=True)
                        fig.update_layout(plot_bgcolor='white')

                    export_config = {'toImageButtonOptions': {'format': 'png', 'filename': 'Rheology_Plot', 'height': 900, 'width': 1200, 'scale': 3}}
                    st.plotly_chart(fig, use_container_width=True, height=800, config=export_config)
                else:
                    st.warning("No data points exist within the selected ranges.")
            else:
                st.warning("Please select at least one interval to plot.")
        except Exception as e:
            st.error(f"Error processing files. Details: {e}")
    else:
        st.info("Awaiting file upload. You can drag and drop multiple .txt files here.")


# =========================================================
# MODULE 2: XRD & LOTGERING FACTOR
# =========================================================
elif app_mode == "X-Ray Diffraction (XRD)":
    st.header("X-Ray Diffraction (XRD) Analysis")
    
    st.sidebar.header("1. Upload Data")
    xrd_files = st.sidebar.file_uploader("Upload XRD Data (.csv or .txt)", type=["csv", "txt"], accept_multiple_files=True)
    
    if xrd_files:
        st.sidebar.header("2. File Settings")
        has_headers = st.sidebar.checkbox("File contains column headers (Check if YES)", value=False)
        st.sidebar.caption("Rigaku files often do NOT have headers. If unchecked, the app will auto-assign 2-Theta and Intensity to the first two columns.")
        skip_rows = st.sidebar.number_input("Rows to Skip (Header text length)", min_value=0, value=0, step=1)
        
        try:
            all_xrd_data = {}
            for file in xrd_files:
                file_name = file.name
                decoded_str = decode_file(file.getvalue())
                
                df = pd.read_csv(io.StringIO(decoded_str), skiprows=skip_rows, header=0 if has_headers else None, sep=None, engine='python')
                if not has_headers:
                    cols = list(df.columns)
                    if len(cols) >= 2:
                        df.rename(columns={cols[0]: "2-Theta", cols[1]: "Intensity"}, inplace=True)
                    else:
                        df.rename(columns={cols[0]: "2-Theta"}, inplace=True)
                all_xrd_data[file_name] = df
            
            st.sidebar.header("3. Data Selection")
            selected_xrd_files = st.sidebar.multiselect(
                "Select Files to Plot", 
                list(all_xrd_data.keys()), 
                default=list(all_xrd_data.keys())
            )
            
            st.sidebar.header("4. Customize Legends & Colors")
            custom_xrd_labels = {}
            custom_xrd_colors = {}
            for i, file_name in enumerate(selected_xrd_files):
                col1, col2 = st.sidebar.columns([3, 1])
                with col1:
                    custom_xrd_labels[file_name] = st.text_input(f"Rename:", value=file_name, key=f"xrd_name_{file_name}")
                with col2:
                    default_color = DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
                    custom_xrd_colors[file_name] = st.color_picker("Color", value=default_color, key=f"xrd_color_{file_name}")
                
            st.sidebar.header("5. Plot Mode")
            display_mode = st.sidebar.radio("Display Mode", ["Overlay", "Stacked"])

            if selected_xrd_files:
                first_file_df = all_xrd_data[selected_xrd_files[0]]
                col_options = list(first_file_df.columns)
                x_col = "2-Theta" if "2-Theta" in col_options else col_options[0]
                y_col = "Intensity" if "Intensity" in col_options else (col_options[1] if len(col_options) > 1 else col_options[0])

                plot_data = pd.DataFrame()
                for i, file_name in enumerate(selected_xrd_files):
                    temp_df = all_xrd_data[file_name].copy()
                    temp_df["Sample Name"] = custom_xrd_labels[file_name]
                    
                    if x_col in temp_df.columns and y_col in temp_df.columns:
                        temp_df = temp_df[[x_col, y_col, "Sample Name"]].dropna()
                        temp_df[x_col] = pd.to_numeric(temp_df[x_col], errors='coerce')
                        temp_df[y_col] = pd.to_numeric(temp_df[y_col], errors='coerce')
                        temp_df = temp_df.dropna()
                        
                        min_y = temp_df[y_col].min()
                        max_y = temp_df[y_col].max()
                        if max_y > min_y:
                            temp_df["Normalized Intensity"] = (temp_df[y_col] - min_y) / (max_y - min_y)
                        else:
                            temp_df["Normalized Intensity"] = 0
                            
                        if display_mode == "Stacked":
                            temp_df["Plot Intensity"] = temp_df["Normalized Intensity"] + i
                        else:
                            temp_df["Plot Intensity"] = temp_df["Normalized Intensity"]
                        plot_data = pd.concat([plot_data, temp_df])

                if not plot_data.empty:
                    st.write(f"### XRD Plot ({display_mode} View)")
                    
                    color_map = {custom_xrd_labels[f]: custom_xrd_colors[f] for f in selected_xrd_files}
                    fig = px.line(plot_data, x=x_col, y="Plot Intensity", color="Sample Name", color_discrete_map=color_map)
                    
                    fig.update_layout(plot_bgcolor='white', hovermode="x unified", legend=dict(title=None, orientation="v", yanchor="top", y=0.99, xanchor="right", x=0.99))
                    fig.update_xaxes(title_text="2θ (degrees)", showgrid=True, gridwidth=1, gridcolor='LightGray', ticks="inside", showline=True, linewidth=1, linecolor='black', mirror=True)
                    
                    if display_mode == "Stacked":
                        fig.update_yaxes(title_text="Intensity (a.u.)", showticklabels=False, showgrid=False, ticks="", showline=True, linewidth=1, linecolor='black', mirror=True)
                    else:
                        fig.update_yaxes(title_text="Normalized Intensity (a.u.)", showticklabels=True, showgrid=False, ticks="inside", showline=True, linewidth=1, linecolor='black', mirror=True)

                    export_config = {'toImageButtonOptions': {'format': 'png', 'filename': f'XRD_Plot_{display_mode}', 'height': 900, 'width': 1200, 'scale': 3}}
                    st.plotly_chart(fig, use_container_width=True, height=700, config=export_config)
                    
                    # ---------------------------------------------------------
                    # 🔬 INDEPENDENT PER-SAMPLE LOTGERING FACTOR
                    # ---------------------------------------------------------
                    with st.expander("🔬 Lotgering Factor Analysis (Orientation Calculation)", expanded=False):
                        st.write("Calculates orientation ($f$) based on JCPDS Ref Cards 2300448 (Al₂O₃) and 1526427 (ZrO₂). Select your target planes per sample to avoid overlap.")
                        
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            search_window = st.slider("Peak Search Window (± 2-Theta)", min_value=0.1, max_value=2.0, value=0.3, step=0.1)
                        
                        if 'ref_peaks_df' not in st.session_state:
                            st.session_state.ref_peaks_df = pd.DataFrame({
                                "Phase": ["Al2O3", "Al2O3", "Al2O3", "Al2O3", "Al2O3", "Al2O3", "Al2O3", "ZrO2", "ZrO2", "ZrO2", "ZrO2"],
                                "Plane (hkl)": ["(012)", "(104)", "(110)", "(006)", "(113)", "(024)", "(116)", "(101)", "(002)", "(110)", "(112)"],
                                "2-Theta (Approx)": [25.58, 35.15, 37.78, 41.68, 43.36, 52.55, 57.50, 30.08, 34.60, 35.30, 50.20],
                                "Ref Intensity (I_0)": [61.1, 100.0, 40.0, 5.0, 50.0, 30.0, 40.0, 100.0, 20.0, 20.0, 50.0],
                                "Is Preferred Plane?": [False, False, False, True, False, False, False, False, False, False, False]
                            })
                            
                        ref_peaks = st.data_editor(st.session_state.ref_peaks_df, num_rows="dynamic", use_container_width=True)
                        ref_peaks["Phase_Plane"] = ref_peaks["Phase"] + " " + ref_peaks["Plane (hkl)"]
                        all_planes = ref_peaks["Phase_Plane"].tolist()
                        
                        st.divider()
                        st.write("#### Per-Sample Peak Configuration")
                        st.write("To prevent mathematical errors, the app defaults to calculating Alumina only. If you wish to calculate Zirconia orientation, uncheck the Alumina planes and select the Zirconia planes.")
                        
                        results = []

                        for file_name in selected_xrd_files:
                            sample_name = custom_xrd_labels[file_name]
                            
                            with st.expander(f"⚙️ Configure: {sample_name}", expanded=False):
                                default_al2o3 = [p for p in all_planes if "Al2O3" in p]
                                
                                included_planes = st.multiselect(
                                    f"Include these planes for {sample_name}:",
                                    options=all_planes,
                                    default=default_al2o3,
                                    key=f"planes_{file_name}"
                                )
                                
                                sample_ref_peaks = ref_peaks[ref_peaks["Phase_Plane"].isin(included_planes)]
                                
                                if not sample_ref_peaks.empty:
                                    sum_i0_all = sample_ref_peaks["Ref Intensity (I_0)"].sum()
                                    sum_i0_pref = sample_ref_peaks[sample_ref_peaks["Is Preferred Plane?"] == True]["Ref Intensity (I_0)"].sum()
                                    p0 = sum_i0_pref / sum_i0_all if sum_i0_all > 0 else 0
                                    
                                    df = all_xrd_data[file_name]
                                    sum_i_all = 0
                                    sum_i_pref = 0
                                    
                                    extracted_intensities = {}
                                    
                                    for _, row in sample_ref_peaks.iterrows():
                                        target_2theta = row["2-Theta (Approx)"]
                                        is_pref = row["Is Preferred Plane?"]
                                        plane_name = row["Phase_Plane"]
                                        
                                        window = df[(df[x_col] >= target_2theta - search_window) & (df[x_col] <= target_2theta + search_window)]
                                        if not window.empty:
                                            peak_intensity = window[y_col].max() - window[y_col].min()
                                        else:
                                            peak_intensity = 0
                                            
                                        extracted_intensities[plane_name] = round(peak_intensity, 2)
                                        
                                        sum_i_all += peak_intensity
                                        if is_pref:
                                            sum_i_pref += peak_intensity
                                            
                                    p = sum_i_pref / sum_i_all if sum_i_all > 0 else 0
                                    f = (p - p0) / (1 - p0) if (1 - p0) != 0 else 0
                                    
                                    st.markdown(f"**Calculated Values:** P₀ = `{p0:.4f}` | P = `{p:.4f}` | Lotgering Factor ($f$) = `{f:.4f}`")
                                    st.write("Extracted Peak Intensities (Baseline Subtracted):")
                                    st.json(extracted_intensities)
                                    
                                    results.append({
                                        "Sample Name": sample_name,
                                        "Primary Phase Analysed": "Mixed" if ("Al2O3" in str(included_planes) and "ZrO2" in str(included_planes)) else ("Al2O3" if "Al2O3" in str(included_planes) else "ZrO2"),
                                        "P₀ (Ref)": round(p0, 4),
                                        "P (Sample)": round(p, 4),
                                        "Lotgering Factor (f)": round(f, 4)
                                    })
                                else:
                                    st.error("Please select at least one plane.")
                                    
                        if results:
                            st.write("#### Final Comparison Table")
                            st.dataframe(pd.DataFrame(results), use_container_width=True)

                else:
                    st.warning("No valid data to plot.")
            else:
                st.warning("Please select at least one file to plot.")
        except Exception as e:
            st.error(f"Error reading file structure. Details: {e}")
    else:
        st.info("Awaiting file upload. You can drag and drop multiple .csv or .txt files here.")
