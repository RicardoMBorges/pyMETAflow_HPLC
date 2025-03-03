# data_processing_HPLC.py
# by Ricardo Moreira Borges (ricardo_mborges@ufrj.br; IPPN-Universidade Federal do Rio de Janeiro)
#    and Stefan Hermann Kuhn (stefan.kuhn@ut.ee; Universitas Tartuensis)
#
# Description:
#    This file provides functions for reading and processing HPLC data, including:
#      - .txt -> .csv conversion
#      - Combining CSV files
#      - Plotting chromatograms (overlapping, stacked)
#      - Filtering RT ranges
#      - Peak detection & visualization
#      - Data alignment (RAFFT, PAFFT, iCOshift)
#      - Normalization & Scaling methods
#      - PCA & PLS-DA
#      - Exporting data in MetaboAnalyst format
#
#    NO FUNCTIONS HAVE BEEN CHANGED from the original code.
#    Only organization & import grouping have been done.

# --------------------------------------------------------------------------
#                           IMPORT STATEMENTS
# --------------------------------------------------------------------------
import os
import glob
import csv
import chardet
import numpy as np
import pandas as pd
import unicodedata
import re
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import plotly.subplots as sp
import mpld3
import seaborn as sns

from io import StringIO
from scipy.signal import correlate, find_peaks
from scipy.stats import norm
from sklearn.model_selection import cross_val_score, GroupKFold
from sklearn.metrics import r2_score
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA

# iCOshift alignment
from pyicoshift import Icoshift

# --------------------------------------------------------------------------
#                    1) TEXT -> CSV Conversion Functions
# --------------------------------------------------------------------------
def process_txt_files(input_folder):
    """
    Reads all .txt files in input_folder, extracts the table that starts at the line containing
    'R.Time (min)' and 'Intensity' (tab-separated), converts numeric decimals from comma to dot,
    and saves the table as a CSV file in a subfolder called 'data'.
    """
    output_folder = os.path.join(input_folder, 'data')
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.txt'):
            file_path = os.path.join(input_folder, filename)
            with open(file_path, 'r') as file:
                lines = file.readlines()
            
            header_index = None
            for idx, line in enumerate(lines):
                if "R.Time (min)" in line and "Intensity" in line:
                    header_index = idx
                    break
            
            if header_index is None:
                print(f"Header not found in {filename}. Skipping file.")
                continue
            
            table_data = "".join(lines[header_index:])
            
            try:
                df = pd.read_csv(StringIO(table_data), sep='\t', decimal=',')
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                continue
            
            output_filename = os.path.splitext(filename)[0] + '.csv'
            output_path = os.path.join(output_folder, output_filename)
            df.to_csv(output_path, index=False)

# --------------------------------------------------------------------------
#                    2) CSV Combination & Chromatogram Plotting
# --------------------------------------------------------------------------
def combine_csv_files(data_folder, output_csv=None):
    """
    Combines CSV files in the specified folder by taking the first column as the common axis
    and using the second column from each file as separate data columns.
    """
    combined_df = None
    csv_files = [f for f in os.listdir(data_folder) if f.lower().endswith('.csv')]
    if not csv_files:
        print("No CSV files found in the folder.")
        return None
    
    for file in csv_files:
        file_path = os.path.join(data_folder, file)
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Error reading {file}: {e}")
            continue
        
        if df.shape[1] < 2:
            print(f"File {file} has less than 2 columns. Skipping.")
            continue
        
        df = df.iloc[:, :2]
        base_name = os.path.splitext(file)[0]
        df.columns = ['RT(min)', base_name]
        
        if combined_df is None:
            combined_df = df
        else:
            combined_df = pd.merge(combined_df, df, on='RT(min)', how='outer')
    
    if combined_df is not None:
        combined_df.sort_values(by='RT(min)', inplace=True)
        combined_df.reset_index(drop=True, inplace=True)
        
        if output_csv:
            combined_df.to_csv(output_csv, index=False)
            print(f"Combined CSV saved to {output_csv}")
    else:
        print("No data to combine.")
    
    return combined_df

def create_chromatogram_plot(dataframe, 
                             x_axis_col='RT(min)', 
                             start_column=1, 
                             end_column=25, 
                             title='Chromatogram Overlapping',
                             xaxis_title='RT (min)',
                             yaxis_title='Intensity',
                             legend_title='Samples',
                             output_dir='images', 
                             output_file='chromatogram_overlapping.html',
                             show_fig=False):
    """
    Creates a Plotly figure of overlapping chromatograms from the given DataFrame,
    using the specified range of columns (by index) for the sample traces.
    """
    fig = go.Figure()
    for column in dataframe.columns[start_column:end_column + 1]:
        fig.add_trace(go.Scatter(x=dataframe[x_axis_col],
                                 y=dataframe[column],
                                 mode='lines',
                                 name=column))
    
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        legend_title=legend_title,
        hovermode='closest'
    )
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_file)
    fig.write_html(output_path)
    print(f"Plot saved as: {output_path}")
    
    if show_fig:
        fig.show()

def create_stacked_chromatogram_plot(dataframe, 
                                     x_axis_col='RT(min)', 
                                     start_column=1, 
                                     end_column=25, 
                                     gap=10000, 
                                     title='Stacked Chromatograms with Larger Gaps',
                                     xaxis_title='RT (min)',
                                     yaxis_title='Intensity (Stacked)',
                                     legend_title='Samples',
                                     output_dir='images', 
                                     output_file='stacked_chromatograms_larger_gap.html',
                                     show_fig=False):
    """
    Creates a stacked chromatogram Plotly figure by offsetting each trace vertically by a specified gap.
    """
    fig = go.Figure()
    for i, column in enumerate(dataframe.columns[start_column:end_column + 1]):
        fig.add_trace(go.Scatter(
            x=dataframe[x_axis_col],
            y=dataframe[column] + i * gap,
            mode='lines',
            name=column
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        legend_title=legend_title,
        hovermode='closest'
    )
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_file)
    fig.write_html(output_path)
    print(f"Plot saved as: {output_path}")
    
    if show_fig:
        fig.show()

def create_vertical_multiplot(dataframes, titles,
                              x_axis_col='RT(min)', 
                              start_column=1, end_column=None, 
                              xaxis_title='RT(min)', 
                              yaxis_title='Intensity', 
                              legend_title='Samples',
                              output_dir='images', 
                              output_file='aligned_nmr_multiplot.html',
                              show_fig=True):
    """
    Creates a vertical multiplot (one subplot per DataFrame) comparing aligned LC data.
    Uses Scattergl for improved performance with large datasets.
    """
    n = len(dataframes)
    # Create subplots: one column, n rows, shared x-axis.
    fig = sp.make_subplots(rows=n, cols=1, shared_xaxes=True,
                           vertical_spacing=0.05,
                           subplot_titles=titles)
    
    for i, df in enumerate(dataframes, start=1):
        x = df[x_axis_col].values
        if end_column is None:
            intensity_cols = df.columns[start_column:]
        else:
            intensity_cols = df.columns[start_column:end_column+1]
        for col in intensity_cols:
            fig.add_trace(
                go.Scattergl(x=x, y=df[col].values, mode='lines', name=col),
                row=i, col=1
            )
        fig.update_yaxes(title_text=yaxis_title, row=i, col=1)
    
    fig.update_layout(
        title="Comparison of Aligned LC data",
        xaxis_title=xaxis_title,
        legend_title=legend_title,
        height=500 * n,
        margin=dict(t=100)
    )
    
    os.makedirs(output_dir, exist_ok=True)
    fig.write_html(os.path.join(output_dir, output_file))
    
    if show_fig:
        fig.show()
        
    return fig

# --------------------------------------------------------------------------
#                    3) Removing Regions Functions
# --------------------------------------------------------------------------
def filter_rt_range(data, start_rt, end_rt, axis_column=None):
    """
    Filters the DataFrame to keep only rows where the values in the axis column are between start_rt and end_rt.
    """
    if axis_column is None:
        axis_column = data.columns[0]
    filtered_data = data[(data[axis_column] >= start_rt) & (data[axis_column] <= end_rt)]
    return filtered_data

def substitute_region_with_zeros(data, region_start, region_end, axis_column=None):
    """
    Sets values to zero for all rows where the value in the axis column is between region_start and region_end.
    """
    if axis_column is None:
        axis_column = data.columns[0]
    modified_data = data.copy()
    mask = (modified_data[axis_column] >= region_start) & (modified_data[axis_column] <= region_end)
    for col in modified_data.columns:
        if col != axis_column:
            modified_data.loc[mask, col] = 0
    return modified_data

# --------------------------------------------------------------------------
#                    4) Peak Analysis & Visualization
# --------------------------------------------------------------------------
def analyze_and_visualize_peaks(dataframe, 
                                x_axis_col='RT(min)', 
                                start_column=1, 
                                end_column=25, 
                                peak_height=5000, 
                                peaks_csv_path='images/peaks_count.csv', 
                                visualization_dir='images/peaks_visualization',
                                show_fig=False):
    """
    Detects peaks in the specified sample columns of a DataFrame, saves a CSV file with the peak counts,
    and creates HTML visualizations for each sample showing the chromatogram with detected peaks.
    """
    os.makedirs(os.path.dirname(peaks_csv_path), exist_ok=True)
    os.makedirs(visualization_dir, exist_ok=True)
    
    peaks_count = {}
    for column in dataframe.columns[start_column:end_column + 1]:
        intensity_values = dataframe[column].values
        peaks, _ = find_peaks(intensity_values, height=peak_height)
        peaks_count[column] = len(peaks)
    
    peaks_count_df = pd.DataFrame(list(peaks_count.items()), columns=['Sample', 'Number of Peaks'])
    peaks_count_df.to_csv(peaks_csv_path, index=False)
    print(f"Peak counts saved to: {peaks_csv_path}")
    
    for column in dataframe.columns[start_column:end_column + 1]:
        intensity_values = dataframe[column].values
        rt_values = dataframe[x_axis_col].values
        peaks, _ = find_peaks(intensity_values, height=peak_height)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=rt_values,
            y=intensity_values,
            mode='lines',
            name=column,
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=rt_values[peaks],
            y=intensity_values[peaks],
            mode='markers',
            name='Detected Peaks',
            marker=dict(color='red', size=8, symbol='circle-open')
        ))
        fig.update_layout(
            title=f'Chromatogram with Detected Peaks for {column}',
            xaxis_title=x_axis_col,
            yaxis_title='Intensity',
            legend_title='Legend',
            hovermode='closest'
        )
        
        output_file = os.path.join(visualization_dir, f'{column}_peaks.html')
        fig.write_html(output_file)
        
        if show_fig:
            fig.show()
    
    return peaks_count_df

# --------------------------------------------------------------------------
#                    5) Single-Fraction Chromatogram Plot
# --------------------------------------------------------------------------
def create_fraction_chromatogram_plot(dataframe, 
                                      selected_rt, 
                                      x_axis_col='RT(min)', 
                                      output_dir='images', 
                                      show_fig=False):
    """
    Creates a line plot of intensities across samples (fractions) at the retention time (RT) closest to a specified value.
    """
    closest_rt = dataframe[x_axis_col].iloc[(dataframe[x_axis_col] - selected_rt).abs().argsort()[:1]].values[0]
    selected_row = dataframe[dataframe[x_axis_col] == closest_rt]
    
    sample_names = dataframe.columns[1:]
    intensities = selected_row.iloc[0, 1:].values

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sample_names,
        y=intensities,
        mode='lines+markers',
        name=f'RT = {closest_rt} min'
    ))
    fig.update_layout(
        title=f'Intensity Across Samples at Closest RT = {closest_rt} min to Selected RT = {selected_rt} min',
        xaxis_title='Samples',
        yaxis_title='Intensity',
        legend_title='Retention Time',
        hovermode='closest'
    )
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'intensity_across_samples_{selected_rt}min.html')
    fig.write_html(output_file)
    print(f"Plot saved as: {output_file}")
    
    if show_fig:
        fig.show()

# --------------------------------------------------------------------------
#                    6) Data Referencing & Alignment
# --------------------------------------------------------------------------
def ref_spectra_to_df(df, thresh=0.01, offsetppm=None, interactive=True, testThreshold=False, xlim=(-0.7, 0.7)):
    """
    Reference (align) HPLC Chromatograms contained in a DataFrame by shifting the RT(min) axis,
    and re-interpolates each spectrum onto a common, referenced RT(min) axis.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the first column named "RT(min)" and subsequent columns as sample intensities.
    thresh : float, optional
        Fraction of maximum intensity used to determine the peak threshold.
    offsetppm : float or None, optional
        If provided, the candidate peak closest to this value is used as the reference for the first spectrum.
    interactive : bool, optional
        If True and offsetppm is None, allows interactive selection via ginput.
    testThreshold : bool, optional
        If True, plots candidate peaks for threshold testing and returns.
    xlim : tuple, optional
        Tuple (xmin, xmax) defining the RT(min) range to search for the reference peak.
    
    Returns
    -------
    referenced_df : pd.DataFrame
        DataFrame with the first column "RT(min)" (the new common axis) and re-interpolated spectra.
    offsets : dict
        Dictionary of offsets applied for each spectrum.
    """
    axis_col = "RT(min)"
    if df.columns[0] != axis_col:
        raise ValueError(f"The first column must be named '{axis_col}'")
    
    ppm_orig = df[axis_col].values
    sample_cols = list(df.columns[1:])
    
    offsets = {}
    sample1 = sample_cols[0]
    intensity1 = df[sample1].values
    thresh_val = thresh * np.max(intensity1)
    
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(intensity1, height=thresh_val)
    candidate_mask = (ppm_orig[peaks] >= xlim[0]) & (ppm_orig[peaks] <= xlim[1])
    candidate_peaks = peaks[candidate_mask]
    
    offset1 = 0.0
    if len(candidate_peaks) == 0:
        print(f"No candidate peaks found in {sample1} within {xlim}. No referencing applied for this spectrum.")
    else:
        if testThreshold:
            plt.figure()
            plt.plot(ppm_orig, intensity1, 'k-', label=sample1)
            plt.plot(ppm_orig[candidate_peaks], intensity1[candidate_peaks], 'ro', label='Candidates')
            plt.xlim(xlim)
            plt.gca().invert_xaxis()
            plt.title(f"Candidate Peaks for {sample1} (Threshold Test)")
            plt.xlabel("RT(min)")
            plt.ylabel("Intensity")
            plt.legend()
            plt.show()
            return None, None
        
        if offsetppm is not None:
            diff = np.abs(ppm_orig[candidate_peaks] - offsetppm)
            idx = np.argmin(diff)
            offset1 = ppm_orig[candidate_peaks][idx]
        else:
            if interactive:
                plt.figure()
                plt.plot(ppm_orig, intensity1, 'k-')
                plt.plot(ppm_orig[candidate_peaks], intensity1[candidate_peaks], 'ro')
                plt.xlim(xlim)
                plt.gca().invert_xaxis()
                plt.title(f"Click near the reference peak for {sample1}")
                plt.xlabel("RT(min)")
                plt.ylabel("Intensity")
                pts = plt.ginput(1, timeout=-1)
                plt.close()
                if pts:
                    click_x = pts[0][0]
                    diff = np.abs(ppm_orig[candidate_peaks] - click_x)
                    idx = np.argmin(diff)
                    offset1 = ppm_orig[candidate_peaks][idx]
                else:
                    print("No selection made; defaulting to no shift.")
                    offset1 = 0.0
            else:
                heights = intensity1[candidate_peaks]
                idx = np.argmax(heights)
                offset1 = ppm_orig[candidate_peaks][idx]
    
    offsets[sample1] = offset1
    new_axis = ppm_orig - offset1
    print(f"Sample '{sample1}' referenced using offset {offset1:.4f} ppm.")
    
    ref_intensity = {}
    for sample in sample_cols:
        intensity = df[sample].values
        thresh_val = thresh * np.max(intensity)
        peaks, _ = find_peaks(intensity, height=thresh_val)
        if len(peaks) == 0:
            print(f"No peaks found in {sample}; no referencing applied.")
            offset_i = 0.0
        else:
            candidate_ppms = ppm_orig[peaks]
            idx = np.argmin(np.abs(candidate_ppms))
            offset_i = candidate_ppms[idx]
        offsets[sample] = offset_i
        shifted_axis = ppm_orig - offset_i
        intensity_interp = np.interp(new_axis, shifted_axis, intensity)
        ref_intensity[sample] = intensity_interp
    
    referenced_df = pd.DataFrame({axis_col: new_axis})
    for sample in sample_cols:
        referenced_df[sample] = ref_intensity[sample]
    
    return referenced_df, offsets


# RAFFT & PAFFT alignment
def recur_align(data, reference, shift, lookahead):
    """
    Recursive alignment for RAFFT on a single 1D data.
    """
    if len(data) < 10:
        return data
    lag = fft_corr(data, reference, shift)
    if lag == 0 and lookahead <= 0:
        return data
    else:
        if lag == 0:
            lookahead -= 1
        if abs(lag) < len(data):
            aligned = move_seg(data, lag)
        else:
            aligned = data.copy()
        mid = find_mid(aligned)
        first_seg = recur_align(aligned[:mid], reference[:mid], shift, lookahead)
        second_seg = recur_align(aligned[mid:], reference[mid:], shift, lookahead)
        return np.concatenate([first_seg, second_seg])

def fft_corr(data, target, shift):
    M = len(target)
    diff = 1e6
    for i in range(1, 21):
        curdiff = (2**i) - M
        if curdiff > 0 and curdiff < diff:
            diff = curdiff
    diff = int(diff)
    target_pad = np.pad(target, (0, diff), mode='constant')
    spectrum_pad = np.pad(data, (0, diff), mode='constant')
    M_new = len(target_pad)
    X = np.fft.fft(target_pad)
    Y = np.fft.fft(spectrum_pad)
    R = (X * np.conjugate(Y)) / M_new
    rev = np.fft.ifft(R)
    vals = np.real(rev)
    maxi = -1
    maxpos = 0
    shift = min(shift, M_new)
    for i in range(shift):
        if vals[i] > maxi:
            maxi = vals[i]
            maxpos = i
        if vals[M_new - i - 1] > maxi:
            maxi = vals[M_new - i - 1]
            maxpos = M_new - i - 1
    if maxi < 0.1:
        return 0
    if maxpos > len(vals) / 2:
        lag = maxpos - len(vals) - 1
    else:
        lag = maxpos - 1
    return lag

def move_seg(seg, lag):
    if lag == 0 or lag >= len(seg):
        return seg
    if lag > 0:
        ins = np.full(lag, seg[0])
        return np.concatenate([ins, seg[:-lag]])
    else:
        lag_abs = abs(lag)
        ins = np.full(lag_abs, seg[-1])
        return np.concatenate([seg[lag_abs:], ins])

def find_mid(spec):
    M = int(np.ceil(len(spec) / 2))
    offset = int(np.floor(M / 4))
    start = max(M - offset, 0)
    end = min(M + offset, len(spec))
    spec_segment = spec[start:end]
    I = np.argmin(spec_segment)
    mid = I + start
    return mid

def align_RAFFT_df(data, reference_idx=0, shift_RT=None, lookahead=1):
    """
    Align spectra using RAFFT from a DataFrame, where the shift is specified in RT(min).
    """
    axis = data.iloc[:, 0].values
    intensities = data.iloc[:, 1:].values

    n_points = len(axis)
    if intensities.shape[0] == n_points:
        intensities = intensities.T

    n_spectra = intensities.shape[0]
    if reference_idx < 0 or reference_idx >= n_spectra:
        raise ValueError(f"Reference index must be between 0 and {n_spectra-1}.")

    reference_spectrum = intensities[reference_idx, :]
    if shift_RT is not None:
        shift = int(round(shift_RT / (np.abs(axis[1] - axis[0]))))
    else:
        shift = len(reference_spectrum)

    aligned_intensities = np.zeros_like(intensities)
    for i in range(n_spectra):
        aligned_intensities[i, :] = recur_align(intensities[i, :], reference_spectrum, shift, lookahead)

    aligned_intensities = aligned_intensities.T
    aligned_df = pd.DataFrame(np.column_stack((axis, aligned_intensities)), columns=data.columns)
    return aligned_df

def align_PAFFT_df(data, segSize_RT, reference_idx=0, shift_RT=None):
    """
    Align spectra using PAFFT from a DataFrame, where segSize is specified in RT(min).
    """
    axis = data.iloc[:, 0].values
    intensities = data.iloc[:, 1:].values

    n_points = len(axis)
    if intensities.shape[0] == n_points:
        intensities = intensities.T

    n_spectra = intensities.shape[0]
    if reference_idx < 0 or reference_idx >= n_spectra:
        raise ValueError(f"Reference index must be between 0 and {n_spectra-1}.")

    reference_spectrum = intensities[reference_idx, :]
    dppm = np.abs(axis[1] - axis[0])

    if shift_RT is not None:
        shift = int(round(shift_RT / (np.abs(axis[1] - axis[0]))))
    else:
        shift = len(reference_spectrum)

    segSize = int(round(segSize_RT / (np.abs(axis[1] - axis[0]))))

    aligned_intensities = np.zeros_like(intensities)
    for i in range(n_spectra):
        aligned_intensities[i, :] = PAFFT(intensities[i, :], reference_spectrum, segSize, shift)

    aligned_intensities = aligned_intensities.T
    aligned_df = pd.DataFrame(np.column_stack((axis, aligned_intensities)), columns=data.columns)
    return aligned_df

def PAFFT(spectrum, reference, segSize, shift):
    n_points = len(spectrum)
    aligned_segments = []
    startpos = 0
    while startpos < n_points:
        endpos = startpos + segSize * 2
        if endpos >= n_points:
            samseg = spectrum[startpos:]
            refseg = reference[startpos:]
        else:
            samseg = spectrum[startpos + segSize: endpos - 1]
            refseg = reference[startpos + segSize: endpos - 1]
            minpos = find_min(samseg, refseg)
            endpos = startpos + minpos + segSize
            samseg = spectrum[startpos:endpos]
            refseg = reference[startpos:endpos]
        lag = fft_corr(samseg, refseg, shift)
        moved = move_seg(samseg, lag)
        aligned_segments.append(moved)
        startpos = endpos + 1
    aligned_full = np.concatenate(aligned_segments)
    if len(aligned_full) < n_points:
        aligned_full = np.pad(aligned_full, (0, n_points - len(aligned_full)), mode='edge')
    else:
        aligned_full = aligned_full[:n_points]
    return aligned_full

def find_min(samseg, refseg):
    Cs = np.sort(samseg)
    Is = np.argsort(samseg)
    Cr = np.sort(refseg)
    Ir = np.argsort(refseg)
    n_limit = max(1, int(round(len(Cs) / 20)))
    for i in range(n_limit):
        for j in range(n_limit):
            if Ir[j] == Is[i]:
                return Is[i]
    return Is[0]

# iCOshift alignment
def align_samples_using_icoshift(df, n_intervals=50, target='maxcorr'):
    """
    Aligns spectral data using iCOshift.
    """
    ppm = df.iloc[:, 0].to_numpy()
    Xnp = df.iloc[:, 1:].to_numpy()
    if np.isnan(Xnp).all():
        raise ValueError("Spectral data contains only NaNs!")
    
    if ppm.shape[0] != Xnp.shape[0]:
        raise ValueError(f"Mismatch: ppm has {ppm.shape[0]} values, but Xnp has {Xnp.shape[0]} rows.")

    Xnp = Xnp.T
    Xnp = np.nan_to_num(Xnp)

    fix_int_size = Icoshift()
    fix_int_size.signals = Xnp
    fix_int_size.unit_vector = ppm
    fix_int_size.inter = ('n_intervals', n_intervals)
    fix_int_size.target = target
    fix_int_size.run()

    aligned_df = pd.DataFrame(fix_int_size.result.T, columns=df.columns[1:])
    aligned_df.insert(0, df.columns[0], ppm)
    return aligned_df

# --------------------------------------------------------------------------
#                    7) CENTERING, NORMALIZATION & SCALING FUNCTIONS
# --------------------------------------------------------------------------
def log_transform(df, constant=1):
    return np.log10(df + constant)

def sqrt_transform(df):
    return np.sqrt(df.where(df >= 0))

def cbrt_transform(df):
    return np.cbrt(df)

def min_max_normalize(df):
    for column in df.columns:
        if column != 'RT(min)':
            min_val = df[column].min()
            max_val = df[column].max()
            df[column] = (df[column] - min_val) / (max_val - min_val)
    return df

def z_score_normalize(df, exclude_columns=None):
    df = df.copy()
    exclude_columns = exclude_columns if exclude_columns else []
    for column in df.columns:
        # Always exclude 'RT(min)' from normalization
        if column not in exclude_columns and column != 'RT(min)':
            mean_val = df[column].mean()
            std_val = df[column].std()
            if std_val != 0:
                df[column] = (df[column] - mean_val) / std_val
            else:
                df[column] = df[column] - mean_val
    return df

def normalize_by_control(df, control_column, exclude_columns=None):
    df = df.copy()
    exclude_columns = exclude_columns if exclude_columns else []
    control = df[control_column]
    for column in df.columns:
        # Exclude the control column, any columns in exclude_columns, and 'RT(min)'
        if column != control_column and column not in exclude_columns and column != 'RT(min)':
            df[column] = df[column] / control
    return df

def pqn_normalize(df, reference=None, exclude_columns=None):
    df = df.copy()
    exclude_columns = exclude_columns if exclude_columns else []
    # Exclude 'RT(min)' from the DataFrame for normalization purposes
    cols = [col for col in df.columns if col != 'RT(min)' and col not in exclude_columns]
    df_numeric = df[cols]
    
    if reference is None:
        reference = df_numeric.median(axis=1)
    
    quotients = df_numeric.divide(reference, axis=0)
    median_quotients = quotients.median(axis=0)
    df_norm = df_numeric.divide(median_quotients, axis=1)
    
    # Put back the excluded columns unchanged.
    for col in df.columns:
        if col not in df_norm.columns:
            df_norm[col] = df[col]
    
    # Optional: reorder to match original order.
    df_norm = df_norm[df.columns]
    return df_norm

def std_dev_normalize(df, exclude_columns=None):
    df = df.copy()
    exclude_columns = exclude_columns if exclude_columns else []
    for column in df.columns:
        if column not in exclude_columns and column != 'RT(min)':
            std_val = df[column].std()
            if std_val != 0:
                df[column] = df[column] / std_val
    return df

def median_normalize(df, target_median=1.0, exclude_columns=None):
    df = df.copy()
    exclude_columns = exclude_columns if exclude_columns else []
    for column in df.columns:
        if column not in exclude_columns and column != 'RT(min)':
            median_val = df[column].median()
            if median_val != 0:
                df[column] = (df[column] / median_val) * target_median
    return df

def quantile_normalize(df, exclude_columns=None):
    """
    Perform quantile normalization on the DataFrame, excluding columns specified in exclude_columns,
    and always excluding 'RT(min)' and 'Chemical Shift (ppm)'.
    
    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - exclude_columns (list, optional): Additional columns to exclude from normalization.
    
    Returns:
    - pd.DataFrame: Quantile normalized DataFrame.
    """
    df = df.copy()
    # Ensure these columns are always excluded
    if exclude_columns is None:
        exclude_columns = ["RT(min)", "Chemical Shift (ppm)"]
    else:
        exclude_columns = list(set(exclude_columns + ["RT(min)", "Chemical Shift (ppm)"]))
    
    # Select only numeric columns for normalization
    norm_cols = [col for col in df.columns if col not in exclude_columns]
    df_numeric = df[norm_cols]

    # Get the shape of the numeric DataFrame
    sorted_df = pd.DataFrame(
        np.sort(df_numeric.values, axis=0),
        index=df_numeric.index,
        columns=df_numeric.columns
    )
    
    # Compute the mean of each row (i.e., rank across samples)
    rank_means = sorted_df.mean(axis=1)
    
    # Create a new DataFrame to store normalized values
    df_normalized = df_numeric.copy()
    
    # For each column, map the sorted ranks to the average values
    for col in df_numeric.columns:
        # Get the order of values (indices that would sort the column)
        order = df_numeric[col].argsort()
        # Create a Series with the rank means corresponding to the sorted order
        normalized_vals = pd.Series(rank_means.values, index=order)
        # Reorder the values to the original order of the data
        df_normalized[col] = normalized_vals.sort_index().values

    # Merge back the excluded columns unchanged
    for col in exclude_columns:
        df_normalized[col] = df[col]
    
    # Reorder columns to match the original DataFrame
    df_normalized = df_normalized[df.columns]
    
    return df_normalized



import numpy as np
import pandas as pd

def min_max_scale(df, new_min=0, new_max=1, exclude_columns=None):
    """
    Scale the DataFrame using min-max scaling to the range [new_min, new_max],
    excluding any columns specified in exclude_columns.
    """
    df = df.copy()
    if exclude_columns is None:
        exclude_columns = ["RT(min)"]
    for column in df.columns:
        if column not in exclude_columns:
            min_val = df[column].min()
            max_val = df[column].max()
            if max_val != min_val:
                df[column] = (df[column] - min_val) / (max_val - min_val) * (new_max - new_min) + new_min
    return df

def standard_scale(df, exclude_columns=None):
    """
    Standard scale the DataFrame (Z-score scaling) for all columns not in exclude_columns.
    """
    df = df.copy()
    if exclude_columns is None:
        exclude_columns = ["RT(min)"]
    for column in df.columns:
        if column not in exclude_columns:
            mean_val = df[column].mean()
            std_val = df[column].std()
            if std_val != 0:
                df[column] = (df[column] - mean_val) / std_val
            else:
                df[column] = df[column] - mean_val
    return df

def robust_scale(df, exclude_columns=None):
    """
    Apply robust scaling using the median and interquartile range for columns not in exclude_columns.
    """
    df = df.copy()
    if exclude_columns is None:
        exclude_columns = ["RT(min)"]
    for column in df.columns:
        if column not in exclude_columns:
            median_val = df[column].median()
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            if IQR != 0:
                df[column] = (df[column] - median_val) / IQR
    return df

def mean_center(df, exclude_columns=None):
    """
    Mean center each column by subtracting its mean for columns not in exclude_columns.
    """
    df = df.copy()
    if exclude_columns is None:
        exclude_columns = ["RT(min)"]
    for column in df.columns:
        if column not in exclude_columns:
            df[column] = df[column] - df[column].mean()
    return df

def auto_scale(df, exclude_columns=None):
    """
    Auto scale the DataFrame by mean-centering and dividing by the standard deviation for columns not in exclude_columns.
    """
    df = df.copy()
    if exclude_columns is None:
        exclude_columns = ["RT(min)"]
    for column in df.columns:
        if column not in exclude_columns:
            std_val = df[column].std()
            if std_val != 0:
                df[column] = (df[column] - df[column].mean()) / std_val
            else:
                df[column] = df[column] - df[column].mean()
    return df

def pareto_scale(df, exclude_columns=None):
    """
    Pareto scale the DataFrame by mean-centering and dividing by the square root of the standard deviation,
    for columns not in exclude_columns.
    """
    df = df.copy()
    if exclude_columns is None:
        exclude_columns = ["RT(min)"]
    for column in df.columns:
        if column not in exclude_columns:
            std_val = df[column].std()
            if std_val != 0:
                df[column] = (df[column] - df[column].mean()) / np.sqrt(std_val)
            else:
                df[column] = df[column] - df[column].mean()
    return df

def range_scale(df, exclude_columns=None):
    """
    Range scale the DataFrame by mean-centering and dividing by the range (max - min) for columns not in exclude_columns.
    """
    df = df.copy()
    if exclude_columns is None:
        exclude_columns = ["RT(min)"]
    for column in df.columns:
        if column not in exclude_columns:
            rng = df[column].max() - df[column].min()
            if rng != 0:
                df[column] = (df[column] - df[column].mean()) / rng
    return df


def plot_histogram_with_distribution(data, output_dir='images', file_name='histogram_with_distribution_curve.html', log_scale=False, x_range=None):
    """
    Plots a horizontal histogram of data with an overlaid normal distribution curve.
    
    Parameters:
    - data (pd.DataFrame): DataFrame with normalized data (already normalized), with one column for each sample.
    - output_dir (str): Directory to save the HTML file.
    - file_name (str): Name of the HTML file to save the plot.
    - log_scale (bool): If True, apply a logarithmic scale to the x-axis.
    - x_range (tuple): Tuple specifying the x-axis range as (min, max).
    
    Returns:
    - fig (go.Figure): Plotly figure object of the histogram with distribution curve.
    """
    # Melt the DataFrame for easier plotting
    melted_df = data.melt(value_name="Normalized Value")
    
    # Create the output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate mean and standard deviation for normal distribution
    mean_val = melted_df["Normalized Value"].mean()
    std_val = melted_df["Normalized Value"].std()
    
    # Generate x values for the normal curve
    x_values = np.linspace(melted_df["Normalized Value"].min(), melted_df["Normalized Value"].max(), 100)
    normal_curve = norm.pdf(x_values, mean_val, std_val)
    
    # Create the Plotly figure
    fig = go.Figure()

    # Add the histogram
    fig.add_trace(go.Histogram(
        x=melted_df["Normalized Value"], 
        histnorm='probability density', 
        opacity=0.6,
        name='Normalized Data'
    ))

    # Add the normal distribution curve
    fig.add_trace(go.Scatter(
        x=x_values,
        y=normal_curve,
        mode='lines',
        line=dict(color='blue', width=2),
        name='Normal Distribution Curve'
    ))

    # Update layout
    fig.update_layout(
        title='Histogram with Normal Distribution Curve',
        xaxis_title='Normalized Value',
        yaxis_title='Density',
        showlegend=True
    )

    # Apply log scale to the x-axis if specified
    if log_scale:
        fig.update_xaxes(type="log")

    # Set x-axis range if specified
    if x_range:
        fig.update_xaxes(range=x_range)

    # Display the plot in the notebook
    #fig.show()

    # Export the plot as an HTML file
    fig.write_html(os.path.join(output_dir, file_name))
    
    return fig

# --------------------------------------------------------------------------
#                    8) STOCSY FUNCTIONS
# --------------------------------------------------------------------------
def STOCSY_LC(target,X,ppm):
    """
    Function designed to calculate covariance/correlation and plots its color coded projection of NMR spectrum
    Adapted for LC, but logic is similar.
    """
    import numpy as np
    from scipy import stats
    import matplotlib.pyplot as plt
    from matplotlib import collections as mc
    import pylab as pl
    import math
    import os
    
    if type(target) == float:
        idx = np.abs(ppm - target).idxmin()
        target_vect = X.iloc[idx]
    else:
        target_vect = target
    
    corr = (stats.zscore(target_vect.T,ddof=1)@stats.zscore(X.T,ddof=1))/((X.T.shape[0])-1)
    covar = (target_vect-(target_vect.mean()))@(X.T-(np.tile(X.T.mean(),(X.T.shape[0],1))))/((X.T.shape[0])-1)
    
    x = np.linspace(0, len(covar), len(covar))
    y = covar
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    fig, axs = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(16,4))
    norm = plt.Normalize(corr.min(), corr.max())
    from matplotlib import collections as mc
    lc = mc.LineCollection(segments, cmap='jet', norm=norm)
    lc.set_array(corr)
    lc.set_linewidth(2)
    line = axs.add_collection(lc)
    fig.colorbar(line, ax=axs)
    axs.set_xlim(x.min(), x.max())
    axs.set_ylim(y.min(), y.max())
    # For LC, we do not invert the x-axis
    minppm = min(ppm)
    maxppm = max(ppm)
    ticksx = []
    tickslabels = []
    if maxppm<30:
       ticks = np.linspace(int(math.ceil(minppm)), int(maxppm), int(maxppm)-math.ceil(minppm)+1)
    else:
       ticks = np.linspace(int(math.ceil(minppm / 10.0)) * 10, (int(math.ceil(maxppm / 10.0)) * 10)-10, int(math.ceil(maxppm / 10.0))-int(math.ceil(minppm / 10.0)))
    currenttick=0
    for p in ppm:
       if currenttick<len(ticks) and p>ticks[currenttick]:
           position=int((p-minppm)/(maxppm-minppm)*max(x))
           if position<len(x):
               ticksx.append(x[position])
               tickslabels.append(ticks[currenttick])
           currenttick+=1
    plt.xticks(ticksx,tickslabels, fontsize=10)
    axs.set_xlabel('RT (min)', fontsize=12)
    axs.set_ylabel(f"Covariance with \n signal at {target:.2f} min", fontsize=12)
    axs.set_title(f'STOCSY from signal at {target:.2f} min', fontsize=14)
    
    text = axs.text(1, 1, '')
    lnx = plt.plot([60,60], [0,1.5], color='black', linewidth=0.3)
    lny = plt.plot([0,100], [1.5,1.5], color='black', linewidth=0.3)
    lnx[0].set_linestyle('None')
    lny[0].set_linestyle('None')
    
    def hover(event):
        if event.inaxes == axs:
            maxcoord=axs.transData.transform((x[0], 0))[0]
            mincoord=axs.transData.transform((x[len(x)-1], 0))[0]
            val=((maxcoord-mincoord)-(event.x-mincoord))/(maxcoord-mincoord)*(maxppm-minppm)+minppm
            c=covar[int(((maxcoord-mincoord)-(event.x-mincoord))/(maxcoord-mincoord)*len(covar))]
            r=corr[int(((maxcoord-mincoord)-(event.x-mincoord))/(maxcoord-mincoord)*len(corr))]
            text.set_visible(True)
            text.set_position((event.xdata, event.ydata))
            text.set_text('{:.2f}'.format(val)+" min, covariance: "+'{:.6f}'.format(c)+", correlation: "+'{:.2f}'.format(r))
            lnx[0].set_data([event.xdata, event.xdata], [-1, 1])
            lnx[0].set_linestyle('--')
            lny[0].set_data([x[0],x[len(x)-1]], [c,c])
            lny[0].set_linestyle('--')
        else:
            text.set_visible(False)
            lnx[0].set_linestyle('None')
            lny[0].set_linestyle('None')
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)    
    pl.show()
    
    if not os.path.exists('images'):
        os.mkdir('images')
    plt.savefig(f"images/stocsy_from_{target}.pdf", transparent=True, dpi=300)
    
    return corr, covar, fig

# --------------------------------------------------------------------------
#                    9) Data-Processing Report
# --------------------------------------------------------------------------
def print_data_processing_report(start_rt, end_rt, samples_to_remove,
                                 aligned_method, normalization_method,
                                 scale_method):
    """
    Print a concise data-processing report for LC analysis.
    """
    samples_str = ", ".join(samples_to_remove) if samples_to_remove else "None"
    report = f"""
Data-Processing Report
----------------------
1. Unwanted outer RT regions removed: {start_rt}–{end_rt} min
2. Removed samples: {samples_str}
3. Alignment method: {aligned_method}
4. Normalization method: {normalization_method}
5. Scaling method: {scale_method}
    """
    print(report.strip())

# --------------------------------------------------------------------------
#                    10) PCA & PLS-DA, VIP FUNCTIONS
# --------------------------------------------------------------------------
def pca_plot(normalized_df,
             df_metadata,
             sample_id_col='HPLC_filename',
             classification_col='ATTRIBUTE_classification',
             n_components=2,
             palette='hls',
             figsize=(8,6),
             show_fig=True):
    """
    Perform PCA on a normalized dataset and plot the first two principal components
    colored by group classification from the metadata.
    """
    X = normalized_df.transpose()
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(X)
    
    pc_labels = [f"PC{i+1}" for i in range(n_components)]
    pca_df = pd.DataFrame(data=principal_components,
                          columns=pc_labels,
                          index=X.index)
    
    pca_df = pca_df.merge(
        df_metadata[[sample_id_col, classification_col]],
        left_index=True,
        right_on=sample_id_col,
        how='left'
    )
    
    plt.figure(figsize=figsize)
    sns.scatterplot(
        data=pca_df,
        x='PC1', y='PC2',
        hue=classification_col,
        palette=palette,
        s=100
    )
    plt.title("PCA Analysis Colored by Group Classification")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="Group")
    
    if show_fig:
        plt.show()
    else:
        plt.close()
    
    return pca, pca_df

def perform_pca_analysis(data, pc_x=1, pc_y=2, n_components=None, variance_threshold=90,
                         metadata=None, color_column="ATTRIBUTE_group", sample_id_col="ATTRIBUTE_localsampleid",
                         output_dir='images', score_plot_filename=None, ev_plot_filename=None,
                         show_fig=True):
    """
    Performs PCA on the given data and produces interactive HTML plots for:
      - PCA Score Plot (PC{pc_x} vs. PC{pc_y})
      - Explained Variance Plot
    """
    if (score_plot_filename is not None) or (ev_plot_filename is not None):
        os.makedirs(output_dir, exist_ok=True)
    
    X = data.transpose()
    
    max_comp = min(X.shape)
    pca_full = PCA(n_components=max_comp)
    pca_full.fit(X)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_) * 100
    n_comp = np.argmax(cum_var >= variance_threshold) + 1
    n_comp = max(n_comp, pc_x, pc_y)
    
    if n_components is not None:
        n_comp = n_components
    
    pca_model = PCA(n_components=n_comp)
    principal_components = pca_model.fit_transform(X)
    
    comp_labels = [f"PC{i+1}" for i in range(n_comp)]
    scores_df = pd.DataFrame(principal_components, columns=comp_labels, index=X.index)
    
    if metadata is not None:
        scores_df = scores_df.merge(metadata[[sample_id_col, color_column]],
                                    left_index=True, right_on=sample_id_col, how='left')
    
    fig_scores = px.scatter(
        scores_df, 
        x=f"PC{pc_x}", 
        y=f"PC{pc_y}",
        color=color_column if metadata is not None else None,
        title="PCA Score Plot",
        labels={f"PC{pc_x}": f"Principal Component {pc_x}", 
                f"PC{pc_y}": f"Principal Component {pc_y}"}
    )
    if score_plot_filename is not None:
        score_plot_file = os.path.join(output_dir, score_plot_filename)
        fig_scores.write_html(score_plot_file)
    if show_fig:
        fig_scores.show()
    
    explained_variance = pca_model.explained_variance_ratio_ * 100
    ev_df = pd.DataFrame({"Component": comp_labels, "Explained Variance (%)": explained_variance})
    fig_ev = px.bar(
        ev_df, 
        x="Component", 
        y="Explained Variance (%)",
        title="Explained Variance by Principal Components",
        text=ev_df["Explained Variance (%)"].apply(lambda x: f"{x:.1f}%")
    )
    fig_ev.update_traces(textposition='outside')
    if ev_plot_filename is not None:
        ev_plot_file = os.path.join(output_dir, ev_plot_filename)
        fig_ev.write_html(ev_plot_file)
    if show_fig:
        fig_ev.show()
    
    return pca_model, scores_df, explained_variance

def plot_pca_loadings(data, pca_model, PC_choose=1, x_axis_col='RT(min)', 
                      output_dir='images', output_file=None, save_fig=True, show_fig=True):
    """
    Creates an interactive Plotly plot of PCA loadings for a chosen principal component,
    with options to save and/or display the plot.
    """
    if output_file is None:
        output_file = f'PCA_PC{PC_choose}_Loadings.html'
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data[x_axis_col],
        y=pca_model.components_[PC_choose - 1],
        mode='lines',
        name=f'PC{PC_choose} Loading'
    ))
    fig.update_layout(
        title=f'PC{PC_choose} Loading Plot',
        xaxis_title=x_axis_col,
        yaxis_title='Loading Value',
        legend_title='Component'
    )
    
    if save_fig:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_file)
        fig.write_html(output_path)
        print(f"Plot saved as: {output_path}")
    
    if show_fig:
        fig.show()
    
    return fig

def plot_pca_scores(scores_df, pc_x, pc_y, explained_variance):
    """
    Create an interactive scatter plot for specified PCA components.
    """
    fig = px.scatter(scores_df, x=f'PC{pc_x}', y=f'PC{pc_y}', text=scores_df.index, title=f'PCA Score Plot: PC{pc_x} vs PC{pc_y}')
    fig.update_layout(
        xaxis_title=f'PC{pc_x} ({explained_variance[pc_x-1]:.2f}%)',
        yaxis_title=f'PC{pc_y} ({explained_variance[pc_y-1]:.2f}%)'
    )
    fig.update_traces(marker=dict(size=7), selector=dict(mode='markers+text'))
    fig.show()


import os
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import OneHotEncoder

def perform_pls_da(
    data,
    metadata,
    group_col="ATTRIBUTE_classification",
    sample_id_col="HPLC_filename",
    n_components=2,
    output_dir="images",
    score_plot_filename=None,
    show_fig=True
):
    """
    Perform PLS-DA (Partial Least Squares Discriminant Analysis) for LC data.
    
    The input 'data' should be a normalized feature matrix with samples as columns and 
    features as rows. For LC data, if a column "RT(min)" is present (representing the 
    retention time in minutes), it is automatically dropped.
    
    The function:
      1. Drops the "RT(min)" column if present.
      2. Transposes the data so that rows represent samples and columns represent features.
      3. Aligns the metadata with the data using a sample ID column.
      4. One-hot encodes the group labels (in group_col).
      5. Fits a PLSRegression model (PLS-DA style) using n_components.
      6. Extracts latent variable scores and merges them with metadata.
      7. Creates an interactive scatter plot (using Plotly Express) of the first two latent variables,
         colored by the specified group.
    
    Parameters
    ----------
    data : pd.DataFrame
        Normalized feature matrix with samples as columns and features as rows.
        (If the DataFrame includes a column "RT(min)", it will be dropped.)
    metadata : pd.DataFrame
        DataFrame containing sample metadata. Must include the sample_id_col (matching data’s columns)
        and a grouping column (group_col) for classification.
    group_col : str
        Column in metadata that contains the class/group labels (default is "ATTRIBUTE_classification").
    sample_id_col : str
        Column in metadata that matches the sample IDs in data’s columns (default is "HPLC_filename").
    n_components : int
        Number of latent variables (components) to use in the PLS-DA model. Default is 2.
    output_dir : str
        Directory where the interactive HTML plot will be saved if score_plot_filename is provided.
    score_plot_filename : str or None
        If provided, the score plot is saved as an HTML file with this name in output_dir.
    show_fig : bool
        If True, the interactive Plotly figure is displayed.
    
    Returns
    -------
    pls_model : PLSRegression
        The fitted PLS regression (PLS-DA) model.
    scores_df : pd.DataFrame
        DataFrame containing the latent variable scores for each sample, merged with group labels.
    """
    # If the data contains an "RT(min)" column (retention time), drop it.
    if "RT(min)" in data.columns:
        data = data.drop(columns=["RT(min)"])
    
    # 1. Transpose data so that rows = samples, columns = features
    X = data.transpose()  # shape: [n_samples, n_features]
    
    # 2. Align the metadata: create a DataFrame of sample IDs from X and merge with metadata.
    sample_index_df = pd.DataFrame({sample_id_col: X.index})
    merged_df = sample_index_df.merge(metadata, on=sample_id_col, how='left')
    
    # 3. One-hot encode the group labels for PLS-DA.
    groups_values = merged_df[group_col].astype(str).values.reshape(-1, 1)
    encoder = OneHotEncoder(sparse_output=False)
    Y = encoder.fit_transform(groups_values)  # shape: [n_samples, n_classes]
    
    # 4. Fit the PLS regression model (PLS-DA style) with the specified number of components.
    pls_model = PLSRegression(n_components=n_components)
    pls_model.fit(X, Y)
    
    # 5. Extract x-scores (latent variable scores for each sample)
    x_scores = pls_model.x_scores_
    lv_cols = [f"LV{i+1}" for i in range(n_components)]
    scores_df = pd.DataFrame(x_scores, columns=lv_cols, index=X.index)
    
    # (A) Reset index so that sample IDs become a column
    scores_df = scores_df.reset_index().rename(columns={"index": sample_id_col})
    # (B) Merge the scores with the group labels from the merged metadata.
    scores_df = scores_df.merge(
        merged_df[[sample_id_col, group_col]],
        on=sample_id_col,
        how='left'
    )
    
    # 6. Create an interactive scatter plot using Plotly Express.
    if n_components >= 2:
        fig = px.scatter(
            scores_df,
            x="LV1",
            y="LV2",
            color=group_col,
            hover_data=[sample_id_col],
            title="PLS-DA Score Plot (LV1 vs. LV2)",
            labels={"LV1": "Latent Variable 1", "LV2": "Latent Variable 2"}
        )
        if score_plot_filename is not None:
            html_path = os.path.join(output_dir, score_plot_filename)
            os.makedirs(output_dir, exist_ok=True)
            fig.write_html(html_path)
        if show_fig:
            fig.show()
    else:
        print(f"n_components={n_components} < 2, so no 2D score plot was generated.")
    
    return pls_model, scores_df

def evaluate_plsda_components(X, y, groups=None, n_splits=5, 
                              save_fig=True, show_fig=True,
                              output_dir='images', 
                              output_file='PLSDA_Q2_R2_Scores.png',
                              figure_size=(10, 5)):
    """
    Evaluate PLS-DA performance over a range of components using cross-validation.
    
    If y is non-numeric (i.e. contains class labels), it is one-hot encoded before evaluation.
    For each number of components (from 1 to the maximum allowed by both the full data and the smallest
    training set among the folds), the function computes:
      - Q² score: the mean cross-validated R² score (using GroupKFold)
      - R² score: the coefficient of determination on the full dataset.
    
    Both scores are then overlaid in a single plot.
    
    The plot can be saved as a PNG image or as an interactive HTML file:
      - If output_file ends with ".html", the figure is saved as HTML (using mpld3).
      - Otherwise, it is saved as a PNG image.
    
    Parameters
    ----------
    X : pd.DataFrame or np.array
        Feature matrix. For LC data, if X includes a column "RT(min)", that column is dropped.
        The matrix must have samples as rows and features as columns.
    y : array-like or pd.Series
        Target variable containing group/class labels.
        If non-numeric, y will be one-hot encoded.
    groups : array-like or None
        Group labels for samples for GroupKFold. If None and X has an index, X.index is used.
    n_splits : int
        Number of splits for GroupKFold cross-validation (default: 5).
    save_fig : bool
        If True, save the combined plot as an image file.
    show_fig : bool
        If True, display the plot interactively.
    output_dir : str
        Directory where the plot will be saved.
    output_file : str
        Filename for the combined Q²/R² scores plot. If it ends with ".html", the plot is saved as HTML.
    figure_size : tuple
        Figure size for the plot.
    
    Returns
    -------
    q2_scores : np.array
        Array of Q² scores for each number of components.
    r2_scores : np.array
        Array of R² scores for each number of components.
    """
    # If X is a DataFrame and contains "RT(min)", drop that column.
    if isinstance(X, pd.DataFrame) and "RT(min)" in X.columns:
        X = X.drop(columns=["RT(min)"])
    
    # Use X.index as groups if groups not provided.
    if groups is None:
        try:
            groups = X.index
        except AttributeError:
            raise ValueError("Groups not provided and X has no index attribute. Please supply groups explicitly.")
    
    # If y is a pandas Series, reindex it to match X's row order.
    if isinstance(y, pd.Series):
        y = y.reindex(X.index)
    
    # If y is non-numeric, one-hot encode it.
    y = np.array(y)
    if not np.issubdtype(y.dtype, np.number):
        encoder = OneHotEncoder(sparse_output=False)
        y = encoder.fit_transform(y.reshape(-1, 1))
    
    # Create a GroupKFold cross-validator.
    group_kfold = GroupKFold(n_splits=n_splits)
    
    # Determine maximum number of components:
    max_comp_full = min(X.shape[0] - 1, X.shape[1])
    fold_train_sizes = []
    for train_idx, _ in group_kfold.split(X, y, groups=groups):
        fold_train_sizes.append(len(train_idx))
    min_train_size = min(fold_train_sizes)
    max_comp_cv = min(min_train_size - 1, X.shape[1])
    
    num_components = min(max_comp_full, max_comp_cv)
    if num_components < max_comp_full:
        print(f"Limiting components to {num_components} based on the smallest training fold (size={min_train_size}).")
    
    # Initialize arrays to store scores.
    q2_scores = np.zeros(num_components)
    r2_scores = np.zeros(num_components)
    
    # Loop over component numbers.
    for i in range(1, num_components + 1):
        plsr = PLSRegression(n_components=i)
        q2 = cross_val_score(plsr, X, y, cv=group_kfold, groups=groups, scoring='r2')
        q2_scores[i - 1] = np.mean(q2)
        
        plsr.fit(X, y)
        y_pred = plsr.predict(X)
        r2_scores[i - 1] = r2_score(y, y_pred)
    
    # Create combined plot for Q² and R² scores.
    plt.figure(figsize=figure_size)
    comps = range(1, num_components + 1)
    plt.plot(comps, q2_scores, marker='o', label='Q² Scores')
    plt.plot(comps, r2_scores, marker='o', label='R² Scores', color='orange')
    plt.xlabel('Number of Components')
    plt.ylabel('Score')
    plt.title('PLS-DA Q² and R² Scores')
    plt.legend()
    
    if save_fig:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_file)
        if output_file.lower().endswith('.html'):
            # Save interactive HTML using mpld3.
            mpld3.save_html(plt.gcf(), output_path)
            print(f"Combined Q²/R² Scores plot saved as HTML: {output_path}")
        else:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Combined Q²/R² Scores plot saved as: {output_path}")
    
    if show_fig:
        plt.show()
    else:
        plt.close()
    
    return q2_scores, r2_scores

def calculate_vip(pls_model):
    """
    Calculate VIP (Variable Importance in Projection) scores for a fitted PLS model.
    Assumes a single response variable.
    
    Parameters
    ----------
    pls_model : fitted PLSRegression model (from sklearn.cross_decomposition)
    
    Returns
    -------
    vip : np.array of shape (n_predictors,)
        VIP scores for each predictor.
    """
    t = pls_model.x_scores_  # latent scores, shape (n_samples, n_components)
    w = pls_model.x_weights_ # weights, shape (n_predictors, n_components)
    q = pls_model.y_loadings_  # y loadings, shape (n_components, 1) for single response
    A = w.shape[1]
    p = w.shape[0]
    
    # Calculate the explained sum of squares for each component.
    ssy = np.zeros(A)
    for a in range(A):
        ssy[a] = np.sum(t[:, a]**2) * (q[a, 0] ** 2)
    total_ssy = np.sum(ssy)
    
    vip = np.zeros(p)
    for j in range(p):
        sum_term = 0.0
        for a in range(A):
            norm_w_a = np.linalg.norm(w[:, a])
            sum_term += ((w[j, a] / norm_w_a) ** 2) * ssy[a]
        vip[j] = np.sqrt(p * sum_term / total_ssy)
    return vip

def plot_pls_loadings_lc_html(pls_model, rt, comp=0, vip_threshold=1.0, 
                              output_filename=None, show_fig=True, figure_size=(800, 600)):
    """
    Plot the loadings for a chosen component from a fitted PLS-DA model against the retention time (RT in minutes).
    
    The loadings are plotted as a continuous line and overlaid with square markers that are color-coded 
    based on their VIP scores:
      - Red squares for VIP >= vip_threshold and positive loading.
      - Blue squares for VIP >= vip_threshold and negative loading.
      - Gray squares for VIP < vip_threshold.
    
    The resulting interactive plot is saved as an HTML file if output_filename is provided.
    
    Parameters
    ----------
    pls_model : fitted PLSRegression model
        The PLS-DA model from which to extract loadings.
    rt : array-like, shape (n_predictors,)
        Retention time values (in minutes) corresponding to the predictors.
    comp : int, optional (default=0)
        The component index to plot (0 corresponds to the first component).
    vip_threshold : float, optional (default=1.0)
        VIP threshold; predictors with VIP >= vip_threshold will be highlighted.
    output_filename : str or None, optional (default=None)
        If provided, the interactive plot is saved as this HTML file.
    show_fig : bool, optional (default=True)
        If True, display the interactive plot.
    figure_size : tuple, optional (default=(800,600))
        Size of the figure in pixels.
    
    Returns
    -------
    fig : plotly.graph_objects.Figure
        The interactive Plotly figure.
    """
    # Get the loadings for the selected component.
    loadings = pls_model.x_loadings_[:, comp]
    vip = calculate_vip(pls_model)
    
    # Create a DataFrame for convenience.
    df = pd.DataFrame({'RT': rt, 'Loadings': loadings, 'VIP': vip})
    
    # Create a Plotly figure.
    fig = go.Figure()
    
    # Add a line trace for the continuous loadings.
    fig.add_trace(go.Scatter(
        x=df['RT'],
        y=df['Loadings'],
        mode='lines',
        line=dict(color='black', width=2),
        name='Loadings Trace'
    ))
    
    # Prepare marker attributes based on VIP threshold.
    marker_colors = []
    marker_sizes = []
    for idx, row in df.iterrows():
        if row['VIP'] >= vip_threshold:
            marker_colors.append('red' if row['Loadings'] >= 0 else 'blue')
            marker_sizes.append(12)
        else:
            marker_colors.append('gray')
            marker_sizes.append(6)
    
    # Add markers as a scatter trace.
    fig.add_trace(go.Scatter(
        x=df['RT'],
        y=df['Loadings'],
        mode='markers',
        marker=dict(
            symbol='circle',
            size=marker_sizes,
            color=marker_colors,
            line=dict(width=1, color='black')
        ),
        name='Loadings'
    ))
    
    # Create dummy traces for a custom legend.
    dummy_traces = [
        go.Scatter(x=[None], y=[None], mode='markers',
                   marker=dict(symbol='circle', size=4, color='red', line=dict(width=1, color='black')),
                   name=f'VIP ≥ {vip_threshold} (positive)'),
        go.Scatter(x=[None], y=[None], mode='markers',
                   marker=dict(symbol='circle', size=4, color='blue', line=dict(width=1, color='black')),
                   name=f'VIP ≥ {vip_threshold} (negative)'),
        go.Scatter(x=[None], y=[None], mode='markers',
                   marker=dict(symbol='circle', size=2, color='gray', line=dict(width=1, color='black')),
                   name=f'VIP < {vip_threshold}'),
        go.Scatter(x=[None], y=[None], mode='lines',
                   line=dict(color='black', width=2),
                   name='Loadings Trace')
    ]
    for dummy in dummy_traces:
        fig.add_trace(dummy)
    
    # Update layout.
    fig.update_layout(
        title=f'PLS-DA Loadings (Component {comp+1})<br>VIP Threshold = {vip_threshold}',
        xaxis_title='RT (min)',
        yaxis_title=f'Loadings (Component {comp+1})',
        width=figure_size[0],
        height=figure_size[1],
        legend_title='Legend'
    )
    
    # Save as HTML if output_filename is provided.
    if output_filename is not None:
        os.makedirs(os.path.dirname(output_filename) or '.', exist_ok=True)
        fig.write_html(output_filename)
        print(f"Loadings plot saved as HTML: {output_filename}")
    
    if show_fig:
        fig.show()
    
    return fig

import numpy as np
import pandas as pd
import plotly.graph_objects as go

def plot_pls_loadings_components(plsr, rt, comp=0, vip_threshold=1.0, output_filename=None, show_fig=True, figure_size=(800,600)):
    """
    Create an interactive Plotly HTML plot for the loadings of a specified component from a fitted PLS-DA model.
    For LC data, the x-axis represents the retention time (RT in minutes).
    
    The function plots a continuous line (the loadings trace) and overlays square markers:
      - Red squares if VIP >= vip_threshold and loading is positive.
      - Blue squares if VIP >= vip_threshold and loading is negative.
      - Gray squares if VIP < vip_threshold.
    
    Parameters:
      plsr : fitted PLSRegression model
          The model must be fitted, so that attributes like x_loadings_ are available.
      rt : array-like, shape (n_predictors,)
          Retention time values (in minutes) corresponding to each predictor.
      comp : int, optional (default=0)
          Component index to plot (0 corresponds to Component 1).
      vip_threshold : float, optional (default=1.0)
          VIP threshold to decide the marker color.
      output_filename : str or None, optional (default=None)
          If provided, the interactive plot is saved as an HTML file with this name.
      show_fig : bool, optional (default=True)
          If True, display the plot interactively.
      figure_size : tuple, optional (default=(800,600))
          Size of the figure in pixels.
      
    Returns:
      fig : plotly.graph_objects.Figure
          The interactive Plotly figure.
    """
    # Ensure the model is fitted by trying to access x_loadings_
    try:
        loadings = plsr.x_loadings_[:, comp]
    except AttributeError:
        print("Error: The PLSRegression model does not have the attribute 'x_loadings_'. "
              "Please ensure the model is fitted (e.g., by calling plsr.fit(X, Y)).")
        return None
    
    # Define a helper function to calculate VIP scores.
    def calculate_vip(pls_model):
        t = pls_model.x_scores_
        w = pls_model.x_weights_
        q = pls_model.y_loadings_
        A = w.shape[1]
        p = w.shape[0]
        ssy = np.zeros(A)
        for a in range(A):
            ssy[a] = np.sum(t[:, a]**2) * (q[a, 0] ** 2)
        total_ssy = np.sum(ssy)
        vip = np.zeros(p)
        for j in range(p):
            sum_term = 0.0
            for a in range(A):
                norm_w_a = np.linalg.norm(w[:, a])
                sum_term += ((w[j, a] / norm_w_a) ** 2) * ssy[a]
            vip[j] = np.sqrt(p * sum_term / total_ssy)
        return vip
    
    vip_scores = calculate_vip(plsr)
    
    # Create a DataFrame for plotting convenience.
    df = pd.DataFrame({
        'RT': rt,
        'Loadings': loadings,
        'VIP': vip_scores
    })
    
    # Determine marker colors and sizes.
    marker_colors = []
    marker_sizes = []
    for idx, row in df.iterrows():
        if row['VIP'] >= vip_threshold:
            marker_colors.append('red' if row['Loadings'] >= 0 else 'blue')
            marker_sizes.append(12)
        else:
            marker_colors.append('gray')
            marker_sizes.append(6)
    
    # Create an interactive Plotly figure.
    fig = go.Figure()
    
    # Add line trace for continuous loadings.
    fig.add_trace(go.Scatter(
        x=df['RT'],
        y=df['Loadings'],
        mode='lines',
        line=dict(color='black', width=2),
        name='Loadings Trace'
    ))
    
    # Add marker trace.
    fig.add_trace(go.Scatter(
        x=df['RT'],
        y=df['Loadings'],
        mode='markers',
        marker=dict(
            symbol='square',
            size=marker_sizes,
            color=marker_colors,
            line=dict(width=1, color='black')
        ),
        name='Loadings'
    ))
    
    # Add dummy traces for custom legend.
    dummy_traces = [
        go.Scatter(x=[None], y=[None], mode='markers',
                   marker=dict(symbol='square', size=12, color='red', line=dict(width=1, color='black')),
                   name=f'VIP ≥ {vip_threshold} (positive)'),
        go.Scatter(x=[None], y=[None], mode='markers',
                   marker=dict(symbol='square', size=12, color='blue', line=dict(width=1, color='black')),
                   name=f'VIP ≥ {vip_threshold} (negative)'),
        go.Scatter(x=[None], y=[None], mode='markers',
                   marker=dict(symbol='square', size=6, color='gray', line=dict(width=1, color='black')),
                   name=f'VIP < {vip_threshold}'),
        go.Scatter(x=[None], y=[None], mode='lines',
                   line=dict(color='black', width=2),
                   name='Loadings Trace')
    ]
    for dummy in dummy_traces:
        fig.add_trace(dummy)
    
    # Update layout.
    fig.update_layout(
        title=f'PLS-DA Loadings (Component {comp+1})<br>VIP Threshold = {vip_threshold}',
        xaxis_title='RT (min)',
        yaxis_title=f'Loadings (Component {comp+1})',
        width=figure_size[0],
        height=figure_size[1],
        legend_title='Legend'
    )
    
    # Save as HTML if output_filename is provided.
    if output_filename is not None:
        os.makedirs(os.path.dirname(output_filename) or '.', exist_ok=True)
        fig.write_html(output_filename)
        print(f"Loadings plot saved as HTML: {output_filename}")
    
    if show_fig:
        fig.show()
    
    return fig

def plot_pls_loadings(data, pls_model, PC_choose=1, x_axis_col='RT(min)', 
                      output_dir='images', output_file=None, save_fig=True, show_fig=True):
    """
    Creates an interactive Plotly plot of PLS-DA loadings for a chosen component,
    with options to save and/or display the plot.

    Parameters:
        data : pd.DataFrame
            DataFrame containing the x-axis values (e.g., "RT(min)").
        pls_model : PLSRegression
            The fitted PLSRegression model (with attribute x_loadings_).
        PC_choose : int
            The component number to plot loadings for (1-indexed; default is 1).
        x_axis_col : str
            Name of the column in data to use for the x-axis (default is 'RT(min)').
        output_dir : str
            Directory where the plot HTML file will be saved.
        output_file : str or None
            If provided, the plot is saved as an HTML file with this name.
            If None, a default filename is generated.
        save_fig : bool
            If True, saves the figure as an HTML file.
        show_fig : bool
            If True, displays the interactive plot.
    
    Returns:
        fig : plotly.graph_objects.Figure
            The interactive Plotly figure.
    """
    # Set a default output filename if not provided.
    if output_file is None:
        output_file = f'PLS_PC{PC_choose}_Loadings.html'
    
    # Create the Plotly figure.
    fig = go.Figure()
    
    # Add a line trace using the PLS model's x_loadings_
    fig.add_trace(go.Scatter(
        x=data[x_axis_col],
        y=pls_model.x_loadings_[:, PC_choose - 1],
        mode='lines',
        name=f'PLS Component {PC_choose} Loading'
    ))
    
    # Update layout with titles and labels.
    fig.update_layout(
        title=f'PLS Component {PC_choose} Loading Plot',
        xaxis_title=x_axis_col,
        yaxis_title='Loading Value',
        legend_title='Component'
    )
    
    # Save the plot if requested.
    if save_fig:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_file)
        fig.write_html(output_path)
        print(f"Plot saved as: {output_path}")
    
    # Optionally display the plot.
    if show_fig:
        fig.show()
    
    return fig

def calculate_vip_scores(pls_model, X):
    t = pls_model.x_scores_
    w = pls_model.x_weights_
    q = pls_model.y_loadings_
    p, h = w.shape
    vip = np.zeros((p,))
    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    total_s = np.sum(s)

    for i in range(p):
        weight = np.array([(w[i, j] / np.linalg.norm(w[:, j]))**2 for j in range(h)])
        vip[i] = np.sqrt(p * (s.T @ weight) / total_s)
    return vip



# --------------------------------------------------------------------------
#                    11) Data-Export
# --------------------------------------------------------------------------
def sanitize_string(s):
    """
    Normalize a string by removing diacritics and keeping only
    English letters, numbers, and underscores.
    """
    s = str(s)
    # Normalize to NFKD form and encode to ASCII, ignoring errors
    s = unicodedata.normalize('NFKD', s)
    s = s.encode('ASCII', 'ignore').decode('utf-8')
    # Remove any character not a letter, number, or underscore
    s = re.sub(r'[^A-Za-z0-9_]', '', s)
    return s

def export_metaboanalyst_lc(normalized_df, df_metadata,
                            sample_id_col="HPLC_filename",
                            class_col="ATTRIBUTE_classification",
                            output_file="metaboanalyst_input.csv"):
    """
    Export LC data to a CSV suitable for MetaboAnalyst with the following format:
      - First row: column headers (e.g., "RT(min)" plus sample IDs)
      - Second row: blank under "RT(min)", then each sample's classification
      - Remaining rows: the actual data from normalized_df

    The function filters the metadata to include only samples present in the data.
    It also sanitizes sample IDs and classification values so that only English letters,
    numbers, and underscores remain.

    Parameters
    ----------
    normalized_df : pd.DataFrame
        LC data with the first column "RT(min)" and subsequent columns as sample intensities.
        The column names after the first one must correspond to sample IDs.
    df_metadata : pd.DataFrame
        Metadata DataFrame containing sample information. Must include:
          - sample_id_col: the column with sample IDs.
          - class_col: the column with class/group information for each sample.
    sample_id_col : str, optional
        The column in df_metadata that contains sample IDs.
    class_col : str, optional
        The column in df_metadata that contains the sample classification.
    output_file : str, optional
        The output CSV filename.

    Returns
    -------
    new_df : pd.DataFrame
        The new DataFrame (with only valid sample columns) that is saved to CSV.
    """
    # --- Step 1: Sanitize column headers in normalized_df ---
    # Assume the first column is "RT(min)" and keep it unchanged.
    orig_col_names = list(normalized_df.columns)
    # For sample columns (all except the first), sanitize their names.
    sanitized_sample_cols = [sanitize_string(s) for s in orig_col_names[1:]]
    sanitized_col_names = [orig_col_names[0]] + sanitized_sample_cols
    normalized_df = normalized_df.copy()
    normalized_df.columns = sanitized_col_names

    # --- Step 2: Identify sample columns (all except the first "RT(min)") ---
    sample_cols = sanitized_col_names[1:]
    
    # --- Step 3: Filter and sanitize metadata ---
    # Create a copy of the metadata and sanitize the sample IDs.
    meta = df_metadata.copy()
    meta[sample_id_col] = meta[sample_id_col].apply(sanitize_string)
    # Also sanitize the classification values.
    meta[class_col] = meta[class_col].apply(sanitize_string)
    # Set the index using the sanitized sample IDs.
    meta_indexed = meta.set_index(sample_id_col)
    # Reindex metadata to include only sample IDs from normalized_df.
    classification_series = meta_indexed.reindex(sample_cols)[class_col]
    
    # Optionally, drop any sample columns with missing metadata.
    valid_sample_cols = classification_series.dropna().index.tolist()
    if len(valid_sample_cols) < len(sample_cols):
        missing = set(sample_cols) - set(valid_sample_cols)
        print(f"Warning: The following sample IDs are missing metadata and will be excluded: {missing}")
    
    # --- Step 4: Build the classification row ---
    # First cell is blank (for "RT(min)"), then classification values for valid sample columns.
    classification_row = [""]
    classification_row.extend(classification_series.loc[valid_sample_cols].values)
    
    # --- Step 5: Build a new DataFrame using only "RT(min)" and valid sample columns ---
    new_df = normalized_df[['RT(min)'] + valid_sample_cols].copy()
    
    # --- Step 6: Write the CSV with the classification row after the header ---
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    with open(output_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        # Write header row.
        writer.writerow(new_df.columns.tolist())
        # Write classification row.
        writer.writerow(classification_row)
        # Write data rows.
        for i in range(len(new_df)):
            writer.writerow(new_df.iloc[i].values)
    
    print(f"MetaboAnalyst input CSV saved as: {output_file}")
    return new_df



# --------------------------------------------------------------------------
#                    12) Data-Processing Report
# --------------------------------------------------------------------------
def print_data_processing_report(start_rt, end_rt, samples_to_remove,
                                 aligned_method, normalization_method,
                                 scale_method):
    """
    Print a concise data-processing report for LC analysis.
    """
    samples_str = ", ".join(samples_to_remove) if samples_to_remove else "None"
    report = f"""
Data-Processing Report
----------------------
1. Unwanted outer RT regions removed: {start_rt}–{end_rt} min
2. Removed samples: {samples_str}
3. Alignment method: {aligned_method}
4. Normalization method: {normalization_method}
5. Scaling method: {scale_method}
    """
    print(report.strip())

# --------------------------------------------------------------------------
#                       End of data_processing_HPLC.py
# --------------------------------------------------------------------------
