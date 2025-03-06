# pyMETAflow for LC-UV Data

A comprehensive workflow for processing, analyzing, and comparing LC‑UV metabolomics data. This repository includes scripts and notebooks that transform raw LC data into meaningful biological insights through a series of preprocessing, alignment, normalization, multivariate analysis, and data fusion steps.
Overview

The pyMETAflow workflow covers the following key steps:

    Data Import and Preprocessing
        Converts raw .txt files to CSV.
        Combines individual CSV files into a single dataset.
        Filters retention time ranges to remove unwanted regions.
        Excludes specific samples (e.g., standards or interfering compounds) based on metadata.

    Chromatogram Visualization and Peak Detection
        Creates overlapping and stacked chromatogram plots using Plotly.
        Detects peaks within the chromatograms and visualizes the results.

    Data Alignment
        Corrects for retention time variations between runs.
        Implements multiple alignment methods (iCOshift, RAFFT, PAFFT) and provides visual comparisons.

    Normalization and Scaling
        Applies normalization techniques such as Z-score, PQN, or control-based normalization.
        Provides scaling methods including mean-centering, auto-scaling, Pareto scaling, and min-max scaling.

    Multivariate Statistical Analysis
        Performs unsupervised (PCA) and supervised (PLS-DA) analyses.
        Generates score and loading plots and calculates VIP scores for variable importance.

    Data Fusion and STOCSY Analysis
        Merges LC‑UV data with bioactivity data to provide a holistic view of sample composition.
        Uses STOCSY to reveal correlations between spectral features.

    Data Export
        Exports processed data in formats suitable for further analysis (e.g., MetaboAnalyst).

#### Installation


#### Required Packages

    Python 3.7+
    numpy
    pandas
    matplotlib
    seaborn
    plotly
    scikit-learn
    scipy
    pyicoshift

## Usage

The repository includes the main processing script (data_processing_HPLC.py) and a Jupyter Notebook demonstrating the workflow (notebooks/pyMETAflow_LC_UV.ipynb). Below are some usage examples:
Data Import and Preprocessing

import data_processing_LC as dp

### Convert .txt files to CSV and combine them
input_folder = r'path/to/your/data'
dp.process_txt_files(input_folder)
data_folder = os.path.join(input_folder, 'data')
combined_df = dp.combine_csv_files(data_folder, output_csv=os.path.join(data_folder, 'combined.csv'))

## Visualization and Peak Detection

### Create overlapping chromatogram plot
dp.create_chromatogram_plot(combined_df, x_axis_col='RT(min)', start_column=1, end_column=25)

### Detect and visualize peaks
peaks_df = dp.analyze_and_visualize_peaks(
    combined_df,
    x_axis_col='RT(min)',
    start_column=1,
    end_column=25,
    peak_height=50000,
    peaks_csv_path='images/peaks_count.csv',
    visualization_dir='images/peaks_visualization'
)

## Data Alignment

### Align data using iCOshift, RAFFT, or PAFFT
aligned_icoshift = dp.align_samples_using_icoshift(combined_df)
aligned_RAFFT = dp.align_RAFFT_df(combined_df, reference_idx=1, shift_RT=0.3, lookahead=1)
aligned_PAFFT = dp.align_PAFFT_df(combined_df, segSize_RT=0.4, reference_idx=0, shift_RT=0.4)

## Normalization and Scaling

### Normalize data using Z-score or PQN normalization
normalized_df = dp.z_score_normalize(aligned_icoshift, exclude_columns="RT(min)")
pqn_normalized_df = dp.pqn_normalize(aligned_icoshift, exclude_columns="RT(min)")

### Scale data using auto-scaling
scaled_df = dp.auto_scale(normalized_df, exclude_columns=["RT(min)"])

## Multivariate Analysis

### Perform PCA
pca_model, pca_scores_df = dp.pca_plot(
    normalized_df=normalized_df,
    df_metadata=pd.read_csv('Metadata.csv', sep="\t"),
    sample_id_col='HPLC_filename',
    classification_col='ATTRIBUTE_classification',
    n_components=2
)

### Perform PLS-DA
pls_model, scores_df = dp.perform_pls_da(
    data=normalized_df,
    metadata=pd.read_csv('Metadata.csv', sep="\t"),
    group_col="ATTRIBUTE_classification",
    sample_id_col="HPLC_filename",
    n_components=2
)

## Data Export

### Export data in MetaboAnalyst format
metabo_df = dp.export_metaboanalyst_lc(
    normalized_df=normalized_df,
    df_metadata=pd.read_csv('Metadata.csv', sep="\t"),
    sample_id_col="HPLC_filename",
    class_col="ATTRIBUTE_classification",
    output_file="data/MetaboAnalyst_input.csv"
)

## Repository Structure

pyMETAflow/
├── data_processing_HPLC.py     # Main processing script
├── README.md                   # This file
├── requirements.txt            # Python package dependencies
└── notebooks/                  # Jupyter notebooks demonstrating the workflow
    └── pyMETAflow_LC_UV.ipynb

## License

This project is licensed under the MIT License. See the LICENSE file for details.
Acknowledgments

    Authors: Ricardo Moreira Borges and Stefan Hermann Kuhn
    Contributions, feedback, and suggestions are welcome.
