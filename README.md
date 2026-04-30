# Music-EEG TRF Analysis Pipeline

This repository implements a pipeline for analysing EEG responses to musical stimuli using Temporal Response Functions (TRFs). The pipeline includes: preparing musical stimuli from MIDI files, extracting acoustic features from the audio (converted WAV files), computing forward (single + multi feature encoding) and backward (decoding) TRF models, aggregating results into a single CSV, checking for unreliable boosting fits, performing statistical analyses, and generating visualisations. The analysis focuses on modelling the relationship between acoustic features of music and EEG to compare brain activity patterns between musicians and non-musicians across five frequency bands and two feature-extraction pipelines (raw and processed).

## Structure

The analysis consists of nine components:

1. **Stimulus Preparation:** Converting MIDI files to WAV format (`midi_to_wav.ipynb`)
2. **Feature Extraction (raw pipeline):** Extracting raw acoustic features from WAV files (`extract_features.ipynb`)
3. **Feature Extraction (processed pipeline):** Extracting, processing and standardising acoustic features from WAV files (`improved_extract_features.ipynb`)
4. **TRF Computation (raw pipeline):** Computing encoding and decoding TRFs using raw features (`TRF_pipeline.py`)
5. **TRF Computation (processed pipeline):** Computing encoding and decoding TRFs using processed features (`improved_TRF_pipeline.py`)
6. **Result Aggregation:** Collecting results from all pickle files (from TRF pipelines) into one CSV (`make_csv.ipynb`)
7. **Data Quality Check:** Verifying the aggregated CSV for unreliable boosting fits (`checking_nans.ipynb`)
8. **Statistical Analysis:** Comparing performance across groups, bands, features, directions and pipelines (`statistical_analysis.ipynb`)
9. **Visualisation:** Producing all figures shown in the thesis (`visualisations.ipynb`)

The `All results` directory contains intermediate outputs (directory of WAV files, feature pickles, the results CSV) and the directory of final visualisation PDFs. The only intermediate outputs not in this repository are large TRF pickle files; they are stored in Google Drive (see *Data Availability* for more information).

## Data Requirements

**EEG Data:**
- Source: `diliBach_4dryad_CND`
- Dataset stored as `.mat` files (`dataSub1.mat` through `dataSub20.mat`)
- 20 subjects (10 non-musicians: Sub1–Sub10, 10 musicians: Sub11–Sub20)
- 30 trials per subject (10 unique Bach pieces, each presented 3 times)
- 64-channel EEG recordings

**Audio Stimuli:**
- Source: `diliBach_midi_4dryad`
- 10 Bach pieces in MIDI format (`audio1.mid` through `audio10.mid`)
- Converted to WAV format (`diliBach_wav_4dryad`) for acoustic feature extraction

The EEG data and MIDI stimuli should be downloaded from the [Dryad repository (doi:10.5061/dryad.g1jwstqmh)](https://doi.org/10.5061/dryad.g1jwstqmh) cited in the references and placed inside the same directory as the rest of the scripts. For setup details, see the *How to Run* section below.

## Dependencies

```
python       3.13.2
numpy        2.3.1
scipy        1.16.0
pandas       2.3.1
matplotlib   3.10.3
mne          1.10.0
eelbrain     0.40.4
librosa      0.11.0
soundfile    0.13.1
pretty_midi  0.2.10
pingouin     0.6.1
statsmodels  0.14.6
```

```bash
# Core scientific computing
pip install numpy scipy pandas matplotlib

# EEG analysis
pip install mne eelbrain

# Audio processing
pip install librosa pretty_midi soundfile

# Statistics
pip install pingouin statsmodels

# Data handling (built-in python)
# os, csv, pickle, pathlib, itertools, sys, time
```

## Pipeline Components

### 1. MIDI to WAV Conversion (`midi_to_wav.ipynb`)

Converts 10 symbolic MIDI files into 10 audio waveforms suitable for acoustic feature extraction.

**Key Steps:**
- Audio synthesis using PrettyMIDI's built-in synthesiser
- Leading-silence trimming based on amplitude threshold
- Amplitude normalisation to a consistent gain

**Output:** 10 WAV files (`1.wav` through `10.wav`) saved in `diliBach_wav_4dryad`

### 2. Feature Extraction — Raw Pipeline (`extract_features.ipynb`)

Extracts raw acoustic features from each WAV file at 100 Hz.

**Features extracted:**
- Envelope (eelbrain's `wav.envelope()`)
- Onsets (positive time-derivative of the envelope)
- Pitch (F0, via librosa `pyin`, NaN-interpolated through unvoiced frames)
- Spectral centroid (librosa)
- Mel spectrogram (8 mel bands, librosa)

**Output:** `song_features.pkl` containing all raw features for all 10 songs

### 3. Feature Extraction — Processed Pipeline (`improved_extract_features.ipynb`)

Extracts acoustic features from each WAV file at 100 Hz, with several improvements over the raw pipeline.

**Improvements:**
- Onsets are low-pass filtered with a Butterworth filter to retain only prominent peaks
- Pitch is recomputed as the absolute time-derivative of F0 (note-change spikes) instead of the raw F0 contour
- MFCC3 (4th MFCC coefficient, capturing spectral fine structure) replaces the mel spectrogram whose high dimensionality made it computationally expensive and also yielded poorer results than the other features
- Standardisation (z-scoring) is applied across all 10 songs, so each feature has zero mean and unit variance

**Output:** `improved_song_features.pkl` containing all processed features for all 10 songs

### 4. TRF Pipeline for Raw Features (`TRF_pipeline.py`)

Performs encoding and decoding TRF analyses for the raw features (using `song_features.pkl`), run per subject across all 5 frequency bands.

**Analysis Steps:**

**EEG Preprocessing:**
- Load subject data from `.mat` file (from `diliBach_4dryad_CND`)
- Scale and resample EEG
- Set up 64-channel montage from `chanlocs`
- Bandpass filter to the target band (delta (1-4 Hz), theta (4-8 Hz), alpha (8-12 Hz), beta (12-30 Hz), wideband (1-30 Hz))
- Concatenate trials, mark trial onsets in a stimulus channel, build MNE Raw object, build eelbrain events

**TRF Modelling:**
- **Forward (encoding) models:** predict EEG from acoustic features; all 31 non-empty subsets of the 5 features are tested as predictor sets (singles, pairs, triples, quads, full quintuple)
- **Backward (decoding) models:** predict each acoustic feature individually from EEG (5 single-feature decoders)
- Pearson correlations (r's) between predicted and true signals are stored for every trial
- For decoded features, per-trial predictions are also stored

**Output:** 100 pickle files (`Sub{n}_{band}.pkl`) containing all TRF models, per-trial r-values, and (for decoded features only) per-trial predicted and true features, saved to `not improved results` (these results are too large for GitHub, see *Data Availability* for access)

### 5. TRF Pipeline for Processed Features (`improved_TRF_pipeline.py`)

Identical in structure to the raw pipeline, but uses the processed features (`improved_song_features.pkl`) and writes 100 pickle files to `results` (these results are also too large for GitHub, see *Data Availability* for access).

### 6. Result Aggregation (`make_csv.ipynb`)

Goes through every pickle file across both pipelines, all 20 subjects, all 5 bands, all 31 encoding combinations, and all 5 decoding features, and assembles data in one CSV file (`all_trf_results.csv`) with 7,200 rows. Each row contains: subject, group, band, direction (encoding or decoding), pipeline, combination name, number of features (for decoding always 1, for encoding up to 5), mean r, mean proportion explained (used for future research, not for thesis), boosting fit time, and individual per-trial r and proportion-explained values (used for future research, not for thesis) for all 30 trials.

**Output:** one CSV file (`all_trf_results.csv`) with all TRF results

### 7. Data Quality Check (`checking_nans.ipynb`)

Checks for NaN entries in the CSV file (caused by boosting models that stopped immediately and produced an all-zero kernel), identifies the affected subject/band/feature combinations, verifies via the original pickle files that the underlying TRF kernels are flat, and flags those rows for exclusion from statistical analysis and visualisations.

There is no output. The subject/band/feature combinations with NaN entries are excluded in cell 1 of both `statistical_analysis.ipynb` and `visualisations.ipynb`.

### 8. Statistical Analysis (`statistical_analysis.ipynb`)

Performs the full statistical analysis of encoding and decoding results across all subjects, bands, pipelines, features and directions.

**Analysis Components:**
- **Group comparisons:** two-way ANOVAs (group * band) per feature and pipeline, for both encoding and decoding
- **Combination-size analyses:** one-way ANOVAs on number of features per band and pipeline
- **Pipeline comparisons:** two-way ANOVAs (pipeline * band) per direction, on single features and on combinations
- **Feature contribution:** linear regression with binary feature indicators

There is no output. The results are discussed in the thesis.

### 9. Visualisations (`visualisations.ipynb`)

Generates all PDF figures shown in the thesis.

**Figure Types:**
- Waveform, acoustic features, EEG and TRF examples
- Best, median and worst decoding reconstructions per feature
- Group and pipeline performance boxplots
- Combination size and runtime analyses plots
- Musicians vs non-musicians TRF butterfly plots and topomaps per band and feature

**Output:** 23 PDF files with all visualisations

## How to Run

1. Create a working directory `DATA_ROOT`. The pipeline assumes everything lives in this single directory.
2. Clone or download this repository into `DATA_ROOT`. All outputs from running the scripts are stored in the `All results` directory of the repository; if you are not running the pipeline from scratch, drag the files out of `All results` into `DATA_ROOT` so the scripts can find them.
3. Download the Bach dataset and place both Bach directories (`diliBach_4dryad_CND` and `diliBach_midi_4dryad`) inside `DATA_ROOT`, alongside all the scripts.
4. Update `DATA_ROOT` at the top of every script (as stated in the comments) to point to that same directory.
5. Convert all MIDI files to WAV format using `midi_to_wav.ipynb` which produces `diliBach_wav_4dryad` directory with all WAV files.
6. Extract raw features using `extract_features.ipynb` which produces `song_features.pkl`.
7. Extract processed features using `improved_extract_features.ipynb` which produces `improved_song_features.pkl`.
8. Run the TRF pipeline for raw features per subject using `TRF_pipeline.py`. To run from the command line use the form (as stated in the comments) e.g., `python TRF_pipeline.py Sub1` and to run inside an IDE, hardcode `SUBJECT = 'Sub1'` instead of reading from `sys.argv` (as stated in the comments). Repeat for all 20 subjects. This produces `not improved results` directory with 100 pickle files (see *Data Availability*). 
9. Run the TRF pipeline for processed features the same way using `improved_TRF_pipeline.py`. Repeat for all 20 subjects. This produces `results` directory with 100 pickle files (see *Data Availability*).
10. Aggregate all pickle results into one CSV using `make_csv.ipynb` which produces `all_trf_results.csv`.
11. Verify data quality using `checking_nans.ipynb`.
12. Run statistical analyses using `statistical_analysis.ipynb`.
13. Generate visualisations using `visualisations.ipynb` which produces 23 PDF files.

## Data Availability

The pickle files produced by the TRF pipelines (one per subject * band) are not committed to this repository due to their size. They are available on Google Drive instead:

- **TRF pipeline for raw features** (100 files in `not improved results`): [Google Drive](https://drive.google.com/drive/folders/1lrsEbL7bi_VuoHowq53tGFW8_tVbqNLr)
- **TRF pipeline for processed features** (100 files in `results`): [Google Drive](https://drive.google.com/drive/folders/1fVWHDM-4HAreBTAF4yEswM5RvZf6Caog)

All other outputs generated and every script used for the thesis are uploaded directly to this repository.

## Key Findings

All findings are presented in the thesis. The full document will be uploaded to this repository once it has been marked.

## References

The dataset is cited in the thesis. Several aspects of the TRF pipeline were adapted from the publicly available *TRF for Alice EEG Dataset* pipeline (also cited in the thesis). Every line in the scripts following this pipeline is annotated with the comment `# "TRF for Alice EEG Dataset"`. Preliminary results from this project, previously published at the AES AIMLA 2025 conference, are noted in the thesis. All citations can be found in the *References* section of the thesis.
