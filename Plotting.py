import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

st.set_page_config(page_title="Material Characterization Plotter", layout="wide")

st.title("Material Characterization Plotter")

DEFAULT_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
]

# =========================================================
# HELPER FUNCTIONS
# =========================================================
@st.cache_data
def parse_rheometer_txt(file_content):
    decoded_content = decode_file(file_content)
    lines = decoded_content.splitlines()
    intervals, data_lines = {}, []
    current_interval = 0
    is_data_section = False
    headers = []
    
    for line in lines:
        line = line.strip()
        if line.startswith("Meas. Pts"):
            if data_lines and headers:  
                df = pd.DataFrame(data_lines, columns=headers)
                df = df.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all').dropna(how='any')
                if not df.empty:
                    intervals[f"Interval {current_interval}"] = df
                data_lines = []
            current_interval += 1
            is_data_section = True
            headers = [h.strip() for h in line.split('\t') if h.strip()]
            continue
        if is_data_section and line.startswith("["):
            continue
        if is_data_section and line == "":
            is_data_section = False
            if data_lines and headers:
                df = pd.DataFrame(data_lines, columns=headers)
                df = df.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all').dropna(how='any')
                if not df.empty:
                    intervals[f"Interval {current_interval}"] = df
                data_lines = []
            continue
        if is_data_section:
            values = [v.replace(',', '') for v in line.split('\t')]
            if len(values) >= len(headers) and len(headers) > 0:
                data_lines.append(values[:len(headers)])

    if data_lines and headers:
        df = pd.DataFrame(data_lines, columns=headers)
        df = df.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all').dropna(how='any')
        if not df.empty:
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

def strip_ka2_rachinger(two_theta, intensity, lambda1=1.540598, lambda2=1.544426, ratio=0.5):
    theta = np.radians(two_theta / 2.0)
    sin_theta_source = (lambda1 / lambda2) * np.sin(theta)
    theta_source = np.arcsin(np.clip(sin_theta_source, -1.0, 1.0))
    two_theta_source = np.degrees(2.0 * theta_source)
    
    sort_idx = np.argsort(two_theta)
    two_theta_sorted = two_theta[sort_idx]
    intensity_sorted = intensity[sort_idx]
    
    I_stripped_sorted = np.zeros_like(intensity_sorted)
    
    for i, t2 in enumerate(two_theta_sorted):
        t2_src = two_theta_source[sort_idx][i]
        if t2_src < two_theta_sorted[0]:
            I_stripped_sorted[i] = intensity_sorted[i]
        else:
            I_src = np.interp(t2_src, two_theta_sorted[:i], I_stripped_sorted[:i])
            I_stripped_sorted[i] = intensity_sorted[i] - (ratio * I_src)
            
    I_stripped_sorted = np.clip(I_stripped_sorted, 0, None)
    unsort_idx = np.argsort(sort_idx)
    return I_stripped_sorted[unsort_idx]

def parse_icdd_txt(file_bytes, filename):
    text = decode_file(file_bytes)
    lines = text.splitlines()
    
    phase_name = filename.split('.')[0]
    for line in lines[:20]:
        if line.startswith("Name:"):
            phase_name = line.split("Name:")[1].strip()
            break
        elif line.startswith("- Name:"):
            phase_name = line.split("- Name:")[1].strip()
            break
        elif line.startswith("Chemical Formula:"):
            phase_name = line.split("Chemical Formula:")[1].strip()
            break
            
    new_peaks = []
    is_data_section = False
    
    for line in lines:
        line = line.strip()
        lower_line = line.lower()
        
        # Detect headers for both COD and Rigaku PDXL2 TXT formats
        if not is_data_section and (("2-theta" in lower_line or "2theta" in lower_line) and ("intensity" in lower_line or " i " in lower_line or "(hkl)" in lower_line or "h |" in lower_line)):
            is_data_section = True
            continue
        
        if is_data_section:
            if line.startswith("---") or line == "":
                continue
                
            clean_line = line.replace("|", " ").replace(",", " ")
            parts = clean_line.split()
            
            try:
                if len(parts) >= 4:
                    # Rigaku PDXL2 format
                    if len(parts) >= 7 and parts[0].isdigit() and "." not in parts[0]:
                        two_theta = float(parts[1])
                        intensity = float(parts[3])
                        hkl = f"({parts[4]}{parts[5]}{parts[6]})"
                    # Standard COD format
                    else:
                        two_theta = float(parts[0])
                        intensity = float(parts[2])
                        hkl_raw = "".join(parts[3:]).replace("(", "").replace(")", "")
                        hkl = f"({hkl_raw})"
                        
                    is_pref = hkl.startswith("(00") or hkl.startswith("(00-")
                        
                    new_peaks.append({
                        "Phase": phase_name,
                        "Plane (hkl)": hkl, 
                        "2-Theta (Approx)": two_theta,
                        "Ref Intensity (I_0)": intensity,
                        "Show on Graph?": intensity >= 10.0,
                        "Is Preferred Plane?": is_pref
                    })
            except ValueError:
                continue
                
    return pd.DataFrame(new_peaks)


# =========================================================
# MAIN NAVIGATION
# =========================================================
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
        all_parsed_intervals = {}
        for file in uploaded_files:
            parsed_intervals = parse_rheometer_txt(file.getvalue())
            for inv_name, df in parsed_intervals.items():
                all_parsed_intervals[f"{file.name} | {inv_name}"] = df
        
        st.sidebar.header("2. Data Selection")
        selected_intervals = st.sidebar.multiselect("Select Measurement Intervals", list(all_parsed_intervals.keys()), default=list(all_parsed_intervals.keys()))
        
        st.sidebar.header("3. Customize Legends, Colors & Order")
        custom_labels, custom_colors, custom_orders = {}, {}, {}
        for i, unique_inv in enumerate(selected_intervals):
            col1, col2, col3 = st.sidebar.columns([1, 2, 1])
            custom_orders[unique_inv] = col1.number_input("Order", value=i, step=1, key=f"order_{unique_inv}")
            custom_labels[unique_inv] = col2.text_input("Rename:", value=unique_inv, key=f"name_{unique_inv}")
            custom_colors[unique_inv] = col3.color_picker("Color", value=DEFAULT_COLORS[i % len(DEFAULT_COLORS)], key=f"color_{unique_inv}")
        
        sorted_intervals = sorted(selected_intervals, key=lambda x: custom_orders[x])
        
        st.sidebar.header("4. Viscosity Units")
        visc_unit = st.sidebar.selectbox("Convert Viscosity to:", ["Pa·s", "mPa·s (cP)", "Poise (P)"])

        plot_df = pd.DataFrame()
        for unique_inv in sorted_intervals:
            temp_df = all_parsed_intervals[unique_inv].copy()
            
            if "Shear Stress" in temp_df.columns and "Shear Rate" in temp_df.columns:
                base_viscosity = temp_df["Shear Stress"] / temp_df["Shear Rate"]
            elif "Viscosity" in temp_df.columns:
                base_viscosity = temp_df["Viscosity"]
            else:
                base_viscosity = pd.Series(np.nan, index=temp_df.index)
            
            if visc_unit == "Pa·s": temp_df["Converted Viscosity"] = base_viscosity
            elif visc_unit == "mPa·s (cP)": temp_df["Converted Viscosity"] = base_viscosity * 1000
            elif visc_unit == "Poise (P)": temp_df["Converted Viscosity"] = base_viscosity * 10
                
            temp_df["Interval"] = unique_inv
            plot_df = pd.concat([plot_df, temp_df])

        if not plot_df.empty:
            st.sidebar.header("5. Plot Controls")
            plot_mode = st.sidebar.radio("Plot Layout", ["Anton Paar Dual-Axis (Visc & Stress)", "Single Variable"])
            
            x_scale = st.sidebar.radio("X-axis Scale", ["Logarithmic", "Linear"], horizontal=True)
            y_scale = st.sidebar.radio("Y-axis Scale", ["Logarithmic", "Linear"], horizontal=True)
            
            x_type, y_type = ("log", "log") if x_scale == "Logarithmic" and y_scale == "Logarithmic" else ("linear", "linear")
            if x_scale != y_scale:
                x_type = "log" if x_scale == "Logarithmic" else "linear"
                y_type = "log" if y_scale == "Logarithmic" else "linear"

            x_exp_format, x_dtick = ("power", 1) if x_type == "log" else ("none", None)
            y_exp_format, y_dtick = ("power", 1) if y_type == "log" else ("none", None)

            x_axis_col = "Shear Rate" if "Shear Rate" in plot_df.columns else plot_df.columns[0]
            st.sidebar.header(f"6. Filter Data Range ({x_axis_col})")
            
            col1, col2 = st.sidebar.columns(2)
            user_min_x = col1.number_input(f"Min", value=float(plot_df[x_axis_col].min()), format="%.3e")
            user_max_x = col2.number_input(f"Max", value=float(plot_df[x_axis_col].max()), format="%.3e")

            st.sidebar.header("7. Graph Dimensions")
            use_custom_size = st.sidebar.checkbox("Custom Graph Size", value=False)
            graph_width = st.sidebar.slider("Width (px)", 400, 1600, 800, 50) if use_custom_size else 1200
            graph_height = st.sidebar.slider("Height (px)", 400, 1600, 800, 50) if use_custom_size else 800

            filtered_df = plot_df[(plot_df[x_axis_col] >= user_min_x) & (plot_df[x_axis_col] <= user_max_x)]
            
            if not filtered_df.empty:
                st.caption("💡 **Tip:** You can click and drag the Legend box directly on the graph! Hover over the top right to download a high-res PNG.")
                
                if plot_mode == "Anton Paar Dual-Axis (Visc & Stress)":
                    if "Shear Rate" not in filtered_df.columns or "Shear Stress" not in filtered_df.columns:
                        st.error("Missing required columns ('Shear Rate' and 'Shear Stress') for the Dual-Axis layout. Please use 'Single Variable' mode.")
                    else:
                        fig = make_subplots(specs=[[{"secondary_y": True}]])
                        for inv in sorted_intervals:
                            data = filtered_df[filtered_df["Interval"] == inv]
                            name, color = custom_labels[inv], custom_colors[inv]
                            
                            fig.add_trace(go.Scatter(x=data["Shear Rate"], y=data["Converted Viscosity"], name=f"{name} (Viscosity)", mode='lines+markers', marker_symbol='diamond', line=dict(color=color)), secondary_y=False)
                            fig.add_trace(go.Scatter(x=data["Shear Rate"], y=data["Shear Stress"], name=f"{name} (Stress)", mode='lines+markers', marker_symbol='square', line=dict(color=color, dash='dot')), secondary_y=True)
                        
                        fig.update_layout(plot_bgcolor='white', hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                        fig.update_xaxes(title_text="Shear Rate [1/s]", type=x_type, exponentformat=x_exp_format, dtick=x_dtick, showgrid=True, gridwidth=1, gridcolor='LightGray', ticks="inside", showline=True, linewidth=1, linecolor='black', mirror=True)
                        fig.update_yaxes(title_text=f"Viscosity [{visc_unit}]", type=y_type, exponentformat=y_exp_format, dtick=y_dtick, secondary_y=False, showgrid=True, gridwidth=1, gridcolor='LightGray', ticks="inside", showline=True, linewidth=1, linecolor='black')
                        fig.update_yaxes(title_text="Shear Stress [Pa]", type=y_type, exponentformat=y_exp_format, dtick=y_dtick, secondary_y=True, showgrid=False, ticks="inside", showline=True, linewidth=1, linecolor='black')
                else:
                    available_cols = [c for c in plot_df.columns if c not in ["Interval", "Meas. Pts.", "Meas. Pts"]]
                    x_col = st.sidebar.selectbox("X-axis", available_cols, index=available_cols.index("Shear Rate") if "Shear Rate" in available_cols else 0) 
                    y_col = st.sidebar.selectbox("Y-axis", available_cols, index=available_cols.index("Converted Viscosity") if "Converted Viscosity" in available_cols else (1 if len(available_cols) > 1 else 0)) 
                    
                    display_df = filtered_df.copy()
                    display_df["Interval"] = display_df["Interval"].map(custom_labels)
                    color_map = {custom_labels[inv]: custom_colors[inv] for inv in sorted_intervals}
                    
                    fig = px.line(display_df, x=x_col, y=y_col, color="Interval", color_discrete_map=color_map, markers=True, symbol="Interval")
                    
                    def get_axis_label(col):
                        labels = {"Converted Viscosity": f"Viscosity [{visc_unit}]", "Shear Rate": "Shear Rate [1/s]", "Shear Stress": "Shear Stress [Pa]", "Strain": "Strain [%]", "Interval Time": "Time [s]", "Torque": "Torque [mNm]"}
                        return labels.get(col, col)
                    
                    fig.update_xaxes(title_text=get_axis_label(x_col), type=x_type, exponentformat=x_exp_format, dtick=x_dtick, ticks="inside", showline=True, linecolor='black', mirror=True)
                    fig.update_yaxes(title_text=get_axis_label(y_col), type=y_type, exponentformat=y_exp_format, dtick=y_dtick, ticks="inside", showline=True, linecolor='black', mirror=True)
                    fig.update_layout(plot_bgcolor='white', legend=dict(title=None))

                if use_custom_size: fig.update_layout(width=graph_width, height=graph_height)
                else: fig.update_layout(height=800)

                export_config = {
                    'toImageButtonOptions': {'format': 'png', 'filename': 'Rheology_Plot', 'height': graph_height if use_custom_size else 900, 'width': graph_width if use_custom_size else 1200, 'scale': 3},
                    'editable': True,
                    'edits': {'legendPosition': True, 'annotationPosition': True, 'titleText': False, 'axisTitleText': False}
                }
                st.plotly_chart(fig, use_container_width=not use_custom_size, config=export_config)
            else:
                st.warning("No data points exist within the selected ranges.")
    else:
        st.info("Awaiting file upload. You can drag and drop multiple .txt files here.")


# =========================================================
# MODULE 2: XRD, ANNOTATIONS & LOTGERING FACTOR
# =========================================================
elif app_mode == "X-Ray Diffraction (XRD)":
    st.header("X-Ray Diffraction (XRD) Analysis")
    st.sidebar.header("1. Upload Data")
    xrd_files = st.sidebar.file_uploader("Upload RAW XRD Data (.csv or .txt)", type=["csv", "txt"], accept_multiple_files=True)
    
    if xrd_files:
        st.sidebar.header("2. File Settings")
        has_headers = st.sidebar.checkbox("File contains column headers", value=False)
        skip_rows = st.sidebar.number_input("Rows to Skip (Header text length)", min_value=0, value=0, step=1)
        
        all_xrd_data = {}
        for file in xrd_files:
            decoded_str = decode_file(file.getvalue())
            
            try:
                df = pd.read_csv(io.StringIO(decoded_str), skiprows=skip_rows, header=0 if has_headers else None, sep=None, engine='python')
                if not has_headers:
                    cols = list(df.columns)
                    if len(cols) >= 2: df.rename(columns={cols[0]: "2-Theta", cols[1]: "Intensity"}, inplace=True)
                    else: df.rename(columns={cols[0]: "2-Theta"}, inplace=True)
                all_xrd_data[file.name] = df
            except pd.errors.ParserError:
                st.error(f"❌ Could not read `{file.name}`. This file looks like an ICDD reference file, not a raw XRD measurement.")
                continue 
        
        if not all_xrd_data:
            st.stop()
        
        st.sidebar.header("3. Data Selection")
        selected_xrd_files = st.sidebar.multiselect("Select Files to Plot", list(all_xrd_data.keys()), default=list(all_xrd_data.keys()))
        
        st.sidebar.header("4. Customize Legends, Colors & Order")
        custom_xrd_labels, custom_xrd_colors, custom_xrd_orders = {}, {}, {}
        for i, file_name in enumerate(selected_xrd_files):
            col1, col2, col3 = st.sidebar.columns([1, 2, 1])
            custom_xrd_orders[file_name] = col1.number_input("Order", value=i, step=1, key=f"xrd_order_{file_name}")
            custom_xrd_labels[file_name] = col2.text_input("Rename:", value=file_name, key=f"xrd_name_{file_name}")
            custom_xrd_colors[file_name] = col3.color_picker("Color", value=DEFAULT_COLORS[i % len(DEFAULT_COLORS)], key=f"xrd_color_{file_name}")
            
        sorted_xrd_files = sorted(selected_xrd_files, key=lambda x: custom_xrd_orders[x])
            
        st.sidebar.header("5. Plot Mode & Appearance")
        display_mode = st.sidebar.radio("Display Mode", ["Overlay", "Stacked"])
        strip_ka2 = st.sidebar.checkbox("Strip Cu Kα₂ Peaks (Rachinger Method)", value=False)
        line_opacity = st.sidebar.slider("Line Opacity", 0.1, 1.0, 1.0, 0.1)

        st.sidebar.header("6. Graph Dimensions")
        use_custom_size = st.sidebar.checkbox("Custom Graph Size", value=False, key="xrd_size_toggle")
        graph_width = st.sidebar.slider("Width (px)", 400, 1600, 800, 50, key="xrd_w") if use_custom_size else 1200
        graph_height = st.sidebar.slider("Height (px)", 400, 1600, 800, 50, key="xrd_h") if use_custom_size else 700

        # ---------------------------------------------------------
        # 📚 ICDD REFERENCE DATABASE & ANNOTATIONS
        # ---------------------------------------------------------
        with st.expander("📚 ICDD Reference Database & Annotations", expanded=True):
            
            if 'ref_peaks_df' not in st.session_state:
                st.session_state.ref_peaks_df = pd.DataFrame(columns=[
                    "Phase", "Plane (hkl)", "2-Theta (Approx)", "Ref Intensity (I_0)", "Show on Graph?", "Is Preferred Plane?"
                ])
                
            uploaded_icdd = st.file_uploader("Upload ICDD Data (.txt)", type=["txt"], accept_multiple_files=True, key="icdd_uploader", help="Upload COD/ICDD text files to extract theoretical peaks.")
            
            if st.button("Extract Peaks from .txt", type="primary") and uploaded_icdd:
                new_data_frames = []
                for txt_file in uploaded_icdd:
                    extracted_df = parse_icdd_txt(txt_file.getvalue(), txt_file.name)
                    if not extracted_df.empty:
                        new_data_frames.append(extracted_df)
                
                if new_data_frames:
                    combined_new = pd.concat(new_data_frames, ignore_index=True)
                    st.session_state.ref_peaks_df = pd.concat([st.session_state.ref_peaks_df, combined_new], ignore_index=True).drop_duplicates(subset=["Phase", "Plane (hkl)", "2-Theta (Approx)"])
                    st.success("Successfully imported peaks!")
                    st.rerun()

            st.write("##### Manage Reference Peaks")
            ref_peaks = st.data_editor(st.session_state.ref_peaks_df, num_rows="dynamic", use_container_width=True, key="ref_peaks_editor")
            st.session_state.ref_peaks_df = ref_peaks 
            
            if not ref_peaks.empty:
                ref_peaks["Phase_Plane"] = ref_peaks["Phase"] + " " + ref_peaks["Plane (hkl)"]
                all_planes = ref_peaks["Phase_Plane"].tolist()
                available_phases = ref_peaks["Phase"].unique().tolist()
            else:
                all_planes = []
                available_phases = []

            st.divider()
            st.write("##### Graph Annotation Controls")
            col1, col2 = st.columns(2)
            
            with col1:
                show_icdd_bottom = st.checkbox("Plot ICDD Reference at Bottom", value=True, help="Draws theoretical vertical stems for selected phases.")
                phases_to_plot = st.multiselect("Select Phases to Plot (Stem Graph):", available_phases, default=available_phases)
            with col2:
                show_annotations = st.checkbox("Show Miller Indices Annotations on Peaks", value=False, help="Pins (hkl) text on top of your raw data peaks.")
                phases_to_annotate = st.multiselect("Select Phases to Annotate:", available_phases, default=available_phases)


        if sorted_xrd_files:
            x_col = "2-Theta" if "2-Theta" in all_xrd_data[sorted_xrd_files[0]].columns else all_xrd_data[sorted_xrd_files[0]].columns[0]
            y_col = "Intensity" if "Intensity" in all_xrd_data[sorted_xrd_files[0]].columns else all_xrd_data[sorted_xrd_files[0]].columns[1]

            plot_data = pd.DataFrame()
            
            offset_base = 1 if (show_icdd_bottom and display_mode == "Stacked" and not ref_peaks.empty) else 0
            
            for i, file_name in enumerate(sorted_xrd_files):
                temp_df = all_xrd_data[file_name].copy()
                temp_df["Sample Name"] = custom_xrd_labels[file_name]
                
                if x_col in temp_df.columns and y_col in temp_df.columns:
                    temp_df = temp_df[[x_col, y_col, "Sample Name"]].dropna()
                    temp_df[x_col] = pd.to_numeric(temp_df[x_col], errors='coerce')
                    temp_df[y_col] = pd.to_numeric(temp_df[y_col], errors='coerce')
                    temp_df = temp_df.dropna()
                    
                    if strip_ka2:
                        temp_df[y_col] = strip_ka2_rachinger(temp_df[x_col].values, temp_df[y_col].values)
                    
                    min_y, max_y = temp_df[y_col].min(), temp_df[y_col].max()
                    temp_df["Normalized Intensity"] = (temp_df[y_col] - min_y) / (max_y - min_y) if max_y > min_y else 0
                    
                    stack_offset = offset_base + i if display_mode == "Stacked" else 0
                    temp_df["Plot Intensity"] = temp_df["Normalized Intensity"] + stack_offset
                    plot_data = pd.concat([plot_data, temp_df])

            if not plot_data.empty:
                st.caption("💡 **Tip:** You can click and drag the Floating Labels (and Miller Indices) directly on the graph!")
                
                plot_data[x_col] = pd.to_numeric(plot_data[x_col], errors='coerce')

                color_map = {custom_xrd_labels[f]: custom_xrd_colors[f] for f in sorted_xrd_files}
                fig = px.line(plot_data, x=x_col, y="Plot Intensity", color="Sample Name", color_discrete_map=color_map)
                fig.update_traces(opacity=line_opacity)
                
                # --- PLOT ICDD STEM GRAPH ---
                if show_icdd_bottom and not ref_peaks.empty and phases_to_plot:
                    is_shown = ref_peaks["Show on Graph?"].fillna(False).astype(bool)
                    
                    for idx, phase in enumerate(phases_to_plot):
                        is_selected_phase = ref_peaks["Phase"] == phase
                        ref_to_plot = ref_peaks[is_shown & is_selected_phase]
                        
                        if not ref_to_plot.empty:
                            x_ref = []
                            y_ref = []
                            for _, row in ref_to_plot.iterrows():
                                x_ref.extend([row["2-Theta (Approx)"], row["2-Theta (Approx)"], None])
                                y_val = row["Ref Intensity (I_0)"] / 100.0
                                y_ref.extend([0, y_val, None])
                                
                            phase_color = DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]
                            
                            fig.add_trace(go.Scatter(
                                x=x_ref, y=y_ref,
                                mode="lines",
                                line=dict(color=phase_color, width=2),
                                name=f"{phase} Ref",
                                hoverinfo="skip"
                            ))
                            
                            max_x_overall = plot_data[x_col].max()
                            fig.add_annotation(
                                x=max_x_overall,
                                y=0.5 - (idx * 0.15), 
                                text=f"{phase} ICDD",
                                font=dict(color=phase_color, size=14, family="Arial", weight="bold"),
                                showarrow=False,
                                xanchor="left",
                                yanchor="middle",
                                xshift=10
                            )

                fig.update_layout(showlegend=False)
                
                # --- RAW DATA FLOATING LABELS ---
                for file_name in sorted_xrd_files:
                    s_name = custom_xrd_labels[file_name]
                    c_color = custom_xrd_colors[file_name]
                    
                    s_data = plot_data[plot_data["Sample Name"] == s_name]
                    if not s_data.empty:
                        max_x = s_data[x_col].max()
                        end_y = s_data[s_data[x_col] == max_x]["Plot Intensity"].values[0]
                        
                        fig.add_annotation(
                            x=max_x,
                            y=end_y,
                            text=s_name,
                            font=dict(color=c_color, size=14, family="Arial"),
                            showarrow=False,
                            xanchor="left",
                            yanchor="middle",
                            xshift=10
                        )
                
                # --- ANNOTATIONS: Miller Indices ---
                if show_annotations and not ref_peaks.empty and phases_to_annotate:
                    is_shown = ref_peaks["Show on Graph?"].fillna(False).astype(bool)
                    is_annotated_phase = ref_peaks["Phase"].isin(phases_to_annotate)
                    
                    for _, row in ref_peaks[is_shown & is_annotated_phase].iterrows():
                        target_x = float(row["2-Theta (Approx)"])
                        window = plot_data[(plot_data[x_col] >= target_x - 0.3) & (plot_data[x_col] <= target_x + 0.3)]
                        peak_y = window["Plot Intensity"].max() if not window.empty else plot_data["Plot Intensity"].max()
                        
                        fig.add_shape(type="line", x0=target_x, x1=target_x, y0=0, y1=peak_y, line=dict(color="rgba(128,128,128,0.5)", width=1, dash="dot"))
                        
                        fig.add_annotation(
                            x=target_x, y=peak_y, text=f"{row['Phase']} {row['Plane (hkl)']}",
                            showarrow=True, arrowhead=0, arrowwidth=1, arrowcolor="rgba(128,128,128,0.8)",
                            ax=0, ay=-30, textangle=-90, xanchor="center", yanchor="bottom"
                        )
                
                fig.update_layout(plot_bgcolor='white', hovermode="x unified", margin=dict(r=150))
                fig.update_xaxes(title_text="2θ (degrees)", type="linear", showgrid=True, gridwidth=1, gridcolor='LightGray', ticks="inside", showline=True, linewidth=1, linecolor='black', mirror=True)
                
                if display_mode == "Stacked": fig.update_yaxes(title_text="Intensity (a.u.)", type="linear", showticklabels=False, showgrid=False, ticks="", showline=True, linewidth=1, linecolor='black', mirror=True)
                else: fig.update_yaxes(title_text="Normalized Intensity (a.u.)", type="linear", showticklabels=True, showgrid=False, ticks="inside", showline=True, linewidth=1, linecolor='black', mirror=True)

                if use_custom_size: fig.update_layout(width=graph_width, height=graph_height)
                else: fig.update_layout(height=graph_height)

                export_config = {
                    'toImageButtonOptions': {'format': 'png', 'filename': f'XRD_Plot_{display_mode}', 'height': graph_height if use_custom_size else 900, 'width': graph_width if use_custom_size else 1200, 'scale': 3},
                    'editable': True,
                    'edits': {'legendPosition': True, 'annotationPosition': True, 'titleText': False, 'axisTitleText': False}
                }
                st.plotly_chart(fig, use_container_width=not use_custom_size, config=export_config)
                
                # ---------------------------------------------------------
                # 🔬 LOTGERING FACTOR ANALYSIS
                # ---------------------------------------------------------
                with st.expander("🔬 Lotgering Factor Analysis (Orientation Calculation)", expanded=False):
                    search_window = st.slider("Peak Search Window (± 2-Theta)", 0.1, 2.0, 0.3, 0.1)
                    st.write("#### Per-Sample Peak Configuration")
                    
                    if not ref_peaks.empty:
                        results = []
                        for file_name in sorted_xrd_files:
                            sample_name = custom_xrd_labels[file_name]
                            
                            with st.expander(f"⚙️ Configure: {sample_name}", expanded=False):
                                default_phase = [p for p in all_planes if available_phases and available_phases[0] in p] if available_phases else []
                                included_planes = st.multiselect(f"Include these planes for {sample_name}:", options=all_planes, default=default_phase, key=f"planes_{file_name}")
                                sample_ref_peaks = ref_peaks[ref_peaks["Phase_Plane"].isin(included_planes)]
                                
                                if not sample_ref_peaks.empty:
                                    sum_i0_all = sample_ref_peaks["Ref Intensity (I_0)"].sum()
                                    sum_i0_pref = sample_ref_peaks[sample_ref_peaks["Is Preferred Plane?"] == True]["Ref Intensity (I_0)"].sum()
                                    p0 = sum_i0_pref / sum_i0_all if sum_i0_all > 0 else 0
                                    
                                    df = all_xrd_data[file_name]
                                    sum_i_all, sum_i_pref = 0, 0
                                    extracted_intensities = {}
                                    
                                    for _, row in sample_ref_peaks.iterrows():
                                        target_2theta, is_pref, plane_name = row["2-Theta (Approx)"], row["Is Preferred Plane?"], row["Phase_Plane"]
                                        window = df[(df[x_col] >= target_2theta - search_window) & (df[x_col] <= target_2theta + search_window)]
                                        peak_intensity = window[y_col].max() - window[y_col].min() if not window.empty else 0
                                            
                                        extracted_intensities[plane_name] = round(peak_intensity, 2)
                                        sum_i_all += peak_intensity
                                        if is_pref: sum_i_pref += peak_intensity
                                            
                                    p = sum_i_pref / sum_i_all if sum_i_all > 0 else 0
                                    f = (p - p0) / (1 - p0) if (1 - p0) != 0 else 0
                                    
                                    st.markdown(f"**Calculated Values:** P₀ = `{p0:.4f}` | P = `{p:.4f}` | Lotgering Factor ($f$) = `{f:.4f}`")
                                    st.write("Extracted Peak Intensities (Baseline Subtracted):")
                                    st.json(extracted_intensities)
                                    
                                    results.append({
                                        "Sample Name": sample_name,
                                        "Primary Phase Analysed": "Mixed" if len(set([p.split(' ')[0] for p in included_planes])) > 1 else (included_planes[0].split(' ')[0] if included_planes else "None"),
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
                        st.info("Add reference peaks to the database above to calculate the Lotgering Factor.")
    else:
        st.info("Awaiting file upload. You can drag and drop multiple .csv or .txt files here.")
