# This script implements the TRF encoding and decoding pipeline using processed acoustic features for one subject across all 5 frequency bands
# All lines where "TRF for Alice EEG Dataset" pipeline (cited in the thesis) is followed are commented with: "TRF for Alice EEG Dataset"


from itertools import combinations
from pathlib import Path
import os
import pickle
import time
import sys
import numpy as np
from scipy.io import loadmat
from scipy.signal import resample
import mne
from mne.channels import make_dig_montage
import eelbrain

# Define paths that will be used throughout
DATA_ROOT = Path('/Users/zorkabozilovic/Desktop/PART1')
EEG_DIR = DATA_ROOT / 'diliBach_4dryad_CND'
OUTPUT_DIR = DATA_ROOT / 'results'
OUTPUT_DIR.mkdir(exist_ok=True)

# Subject is passed as a command-line argument so the script can be run per subject in parallel
# Example run: python improved_TRF_pipeline.py Sub1
# Or hardcode for IDE runs (example: SUBJECT = 'Sub1')
SUBJECT = sys.argv[1]

# TRF parameters
DECODE_TMIN = -0.600
DECODE_TMAX = 0.200
ENCODE_TMIN = -0.150
ENCODE_TMAX = 0.750
BASIS = 0.050
PARTITIONS = 4
ERROR = 'l1'

# All 5 features
FEATURE_NAMES = ['envelope', 'onsets', 'pitch', 'centroid', 'mfcc3']

# All bands
BANDS = {
    'delta':    (1, 4),
    'theta':    (4, 8),
    'alpha':    (8, 12),
    'beta':     (12, 30),
    'wideband': (1, 30),
}



# Load EEG data from one subject
def load_subject_raw_eeg(filepath, subject):

    # Extract subject index from string (e.g., 'Sub18' -> 18)
    subject_idx = int(subject[3:])

    # Load the .mat file
    mat_data = loadmat(filepath, struct_as_record=False, squeeze_me=True)
    eeg = mat_data["eeg"]
    target_fs = 500  # sampling frequency
    orig_fs = int(eeg.fs)
    resample_needed = orig_fs != target_fs
    for i in range(len(eeg.data)):

        # Scale data 
        trial_data = 100 * eeg.data[i].astype(np.float32) / np.iinfo(np.int32).max

        # Resample to 500Hz, to be consistent with the "TRF for Alice EEG Dataset" pipeline
        if resample_needed:
            n_samples = int(trial_data.shape[0] * target_fs / orig_fs)
            trial_data = resample(trial_data, n_samples, axis=0)

        eeg.data[i] = trial_data

    # Extract key information into a dictionary
    raw_data = {
        'trials': eeg.data,
        'fs': target_fs,
        'chanlocs': eeg.chanlocs,
        'pad_start': int(eeg.paddingStartSample * target_fs / orig_fs) if resample_needed else int(eeg.paddingStartSample),
        'subject_type': 'Musician' if subject_idx >= 11 else 'Non-musician'
    }

    print(f"Loaded {raw_data['subject_type']} (Subject {subject})")
    print(f"  - {len(raw_data['trials'])} trials, {raw_data['trials'][0].shape[1]} channels")

    return raw_data

#Convert already-loaded Bach data to MNE Raw object with channel positions
def create_mne_raw_from_loaded(subject_data):

    trials = subject_data['trials']
    sfreq = subject_data['fs']
    pad_start = subject_data['pad_start']    
    chanlocs = subject_data['chanlocs']

    # Get channel names and positions
    ch_names = []
    positions = []

    for ch in chanlocs:

        # Get channel label        
        ch_names.append(ch.labels)

        # Get channel positions if available
        if hasattr(ch, 'X') and hasattr(ch, 'Y') and hasattr(ch, 'Z'):
            positions.append([ch.Y, ch.X, ch.Z])

    # Concatenate all trials
    all_trials = []
    trial_lengths = []

    for trial in trials:
        # Remove padding and transpose to channels x time
        trial_clean = trial[pad_start:, :].T
        all_trials.append(trial_clean)
        trial_lengths.append(trial_clean.shape[1])

    # Concatenate
    eeg_continuous = np.hstack(all_trials)
    n_channels, n_samples = eeg_continuous.shape

    # Create stimulus channel with trial markers
    stim_data = np.zeros((1, n_samples))

    # Mark all 30 trial onsets
    current_sample = 0
    marker_positions = []
    for i in range(30):
        # Place marker at current position (offset by 1 if at sample 0)
        marker_sample = 1 if current_sample == 0 else current_sample
        stim_data[0, marker_sample] = i + 1 # Use 1-30 as event IDs
        marker_positions.append((i+1, marker_sample))
        current_sample += trial_lengths[i] # Move to next trial start

    # Combine EEG and stim
    data_with_stim = np.vstack([eeg_continuous, stim_data])

    # Channel setup
    ch_names = ch_names + ['STI']
    ch_types = ['eeg'] * n_channels + ['stim']

    # Create Raw
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data_with_stim, info)

    montage = make_dig_montage(
        ch_pos=dict(zip(ch_names[:n_channels], positions)),
        coord_frame='head'
    )
    raw.set_montage(montage)
    return raw

# Create eelbrain events with correct column structure
def create_eelbrain_events(raw):

    # Find events in the MNE raw object
    mne_events = mne.find_events(raw, stim_channel='STI', verbose=False)

    # Create eelbrain Dataset with the required columns
    events_data = {
        'i_start': mne_events[:, 0], # Sample indices
        'trigger': mne_events[:, 2], # Event IDs (1-30)
        'event': mne_events[:, 2] # Same as trigger (1-30)
    }

    events = eelbrain.Dataset(events_data)

    # Link raw data to events for use in variable_length_epochs
    events.info['raw'] = raw
    return events



# Load pre-extracted acoustic features

with open(DATA_ROOT / 'improved_song_features.pkl', 'rb') as f:
    song_features = pickle.load(f)



# Generate all 31 non-empty subsets of the 5 features for encoding

feature_combos = []
for r in range(1, len(FEATURE_NAMES) + 1):
    for combo in combinations(FEATURE_NAMES, r):
        feature_combos.append(combo)



# Load EEG once for this subject
eeg_data = load_subject_raw_eeg(EEG_DIR / f'data{SUBJECT}.mat', SUBJECT)

# Run all 5 bands for this subject
for BAND_NAME, (LOW_FREQUENCY, HIGH_FREQUENCY) in BANDS.items():

    out_path = OUTPUT_DIR / f'{SUBJECT}_{BAND_NAME}.pkl'
    print(f'{SUBJECT} — {BAND_NAME} ({LOW_FREQUENCY}-{HIGH_FREQUENCY} Hz)')
    run_start = time.time()

    # Filter EEG and create events
    raw = create_mne_raw_from_loaded(eeg_data)
    raw.filter(LOW_FREQUENCY, HIGH_FREQUENCY, n_jobs=1, verbose=False) # Follow "TRF for Alice EEG Dataset" pipeline
    events = create_eelbrain_events(raw)

    # Add pre-extracted features to the events table (use the similar iteration pattern as "TRF for Alice EEG Dataset")
    for feat_name in FEATURE_NAMES:
        feat_list = []
        for stimulus_id in events['event']:
            song_id = stimulus_id % 10
            song_id = song_id if song_id != 0 else 10
            feat_list.append(song_features[song_id][feat_name].copy())
        events[feat_name] = feat_list

    # Extract the stimulus duration (in seconds) from the envelopes (follow "TRF for Alice EEG Dataset")
    events['duration'] = eelbrain.Var([env.time.tstop for env in events['envelope']])

    # Extract EEG data corresponding exactly to the timing of the envelopes (follow "TRF for Alice EEG Dataset")
    events['eeg'] = eelbrain.load.mne.variable_length_epochs(events, 0, tstop='duration', decim=5, connectivity='auto')


    # Run encoding TRFs for all 31 feature combinations
    encode_results = {}

    for combo_idx, combo in enumerate(feature_combos):
        combo_name = '+'.join(combo)
        feature_keys = list(combo) if len(combo) > 1 else combo[0]

        print(f'  [{combo_idx+1:2d}/{len(feature_combos)}] Encoding: {combo_name} to eeg')

        try:

            # Train encoder
            model = eelbrain.boosting(
                'eeg', feature_keys, ENCODE_TMIN, ENCODE_TMAX,
                data=events, basis=BASIS, partitions=PARTITIONS,
                test=True, error=ERROR
            )

            # Per-trial predictions (same normalisation as decoding)
            trials = {}
            trial_rs = []

            for trial_num in range(events.n_cases):

                # Per-trial encoding normalisation used "TRF for Alice EEG Dataset" decoding procedure applied to encoding
                # Normalize the features      
                if isinstance(feature_keys, list):
                    predictors = [events[trial_num, fk] / model.x_scale[i] for i, fk in enumerate(feature_keys)]
                else:
                    predictors = events[trial_num, feature_keys] / model.x_scale

                # Predict EEG by convolving encoder with normalised features
                eeg_pred = eelbrain.convolve(model.h, predictors)

                # Normalise true EEG and compute correlation with prediction
                eeg_true = events[trial_num, 'eeg']
                eeg_true = eeg_true - model.y_mean
                eeg_true /= model.y_scale / eeg_pred.std()

                # Compute correlation between predicted and true EEG
                r_trial = eelbrain.correlation_coefficient(eeg_true, eeg_pred)
                trial_rs.append(float(np.mean(r_trial)))

                # Compute proportion explained (was not used in the thesis, but was left for future research)
                ss_total = eeg_true.abs().sum('time')
                ss_residual = (eeg_true - eeg_pred).abs().sum('time')
                prop_explained_trial = 1 - (ss_residual / ss_total)

                # Store trial data
                trials[f'trial{trial_num}'] = {
                    'r': r_trial,
                    'proportion_explained': prop_explained_trial,
                }

            # Store encoding results
            encode_results[combo_name] = {
                'model': model,
                'trial_rs': trial_rs,
                'trials': trials,
            }

        except Exception as e:
            print(f'FAILED: {e}')
            encode_results[combo_name] = {'error': str(e)}


    # Run decoding TRFs
    decode_results = {}

    for feat_idx, feat in enumerate(FEATURE_NAMES):
        print(f'  [{feat_idx+1}/{len(FEATURE_NAMES)}] Decoding: eeg to {feat}')

        try:

            # Train decoder
            model = eelbrain.boosting(
                feat, 'eeg', DECODE_TMIN, DECODE_TMAX,
                data=events, basis=BASIS, partitions=PARTITIONS,
                test=True, error=ERROR
            )

            # Per-trial predictions (same normalisation as encoding)
            trials = {}
            trial_rs = []

            for trial_num in range(events.n_cases):

                # Per-trial decoding normalisation followed "TRF for Alice EEG Dataset"
                # Normalize the EEG
                eeg_one_event = events[trial_num, 'eeg'] / model.x_scale

                # Predict the feature by convolving the decoder with the EEG
                y_pred = eelbrain.convolve(model.h, eeg_one_event, name=f'predicted {feat}')

                # Normalize trial feature and compute correlation with prediction
                y_true = events[trial_num, feat]
                y_true = y_true - model.y_mean
                y_true /= model.y_scale / y_pred.std()

                # Compute correlation between predicted and true feature
                r = eelbrain.correlation_coefficient(y_true, y_pred)
                trial_rs.append(float(r))  

                # Compute proportion explained (was not used in the thesis, but was left for future research)
                ss_total = y_true.abs().sum('time')
                ss_residual = (y_true - y_pred).abs().sum('time')
                prop_explained_trial = 1 - (ss_residual / ss_total)

                # Store trial data
                trials[f'trial{trial_num}'] = {
                    'y_pred': y_pred,
                    'y': y_true,
                    'r': r,
                    'proportion_explained': prop_explained_trial,
                }

            # Store decoding results
            decode_results[feat] = {
                'model': model,
                'trial_rs': trial_rs,
                'trials': trials,
            }

        except Exception as e:
            print(f'FAILED: {e}')
            decode_results[feat] = {'error': str(e)}

    # Save all results for this band
    all_results = {
        'encode_results': encode_results,
        'decode_results': decode_results,
    }

    with open(out_path, 'wb') as f:
        pickle.dump(all_results, f)

    # Print summary
    n_enc = len([v for v in encode_results.values() if 'error' not in v])
    n_dec = len([v for v in decode_results.values() if 'error' not in v])
    elapsed = time.time() - run_start
    file_size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f'  Saved {out_path.name} ({file_size_mb:.1f} MB) — {n_enc} enc + {n_dec} dec — {elapsed/60:.1f} min')