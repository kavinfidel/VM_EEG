
import os

import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
from scipy.signal import welch
#import regex as reg
from sklearn.pipeline import Pipeline
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import mne
import numpy as np
from scipy.stats import iqr
import warnings
import torch
from torch_geometric.data import Data
from utils.train_loop import train_riemannian_gnn
from models.RiemannianGAT import RiemannianGAT
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

mne.set_log_level('ERROR')   # silence MNE

warnings.filterwarnings("ignore")  # silence warnings

time_window = 0.5
fs = 500

ica = True



latest_channel_list = [
    # Left sensorimotor area channels
    'E29', 'E30', 'E35', 'E36', 'E41', 'E42',
    # Right sensorimotor area channels
    'E103', 'E104', 'E109', 'E110', 'E115', 'E116',
    # Mid-parietal & bilateral parietal
    'E62', 'E67', 'E72', 'E77'
 ]

new_latest = ['E24', 'E124', 'E36', 'E104', 'E47','E52', 'E60', 'E67', 'E72', 'E77', 'E85', 'E92', 'E98', 'E62','E70', 'E75', 'E83','E58','E96','E90','E65','E69','E74','E82','E89'
              ,'E1', 'E32','E14', 'E21']

bad_channels = ['E17', 'E38', 'E94', 'E113', 'E119', 'E121', 'E125', 'E128', 'E73', 'E81', 'E88', 'E43', 'E44', 'E120', 'E114','E127', 'E126',
                 'E68', 'E23', 'E3','E49','E48', "E8", "E25",
     "E56", "E63", "E99", "E107"]

bad_channels = ['E17', 'E38', 'E94', 'E113', 'E119', 'E121', 'E125', 'E128', 'E73', 'E81', 'E88', 'E43', 'E44', 'E120', 'E114','E127', 'E126',
                 'E68', 'E23', 'E3','E49','E48', "E8", "E25",
     "E56", "E63", "E99", "E107",]
                 
#bad_channels = ['E48', 'E119', 'E49', 'E113', 'E94', 'E68', 'E23', 'E3', 'E126', 'E127']



#label_dict = {'OBBA': 0, 'OBBY': 1, 'OBDO': 2, 'OBMO': 3, 'OBSI':4}
label_dict = {'IMMO': 0,'IMBY': 1,'IMSI':2 } # banana, baby, sitar
directions = ['IMMO', 'IMBY', 'IMSI']

#directions = ['OBBA', 'OBBY', 'OBDO', 'OBDO','OBSI']  # Left, Right, Up, Down

channel_tuple = (new_latest, bad_channels)



class preprocessing_pipeline:
    def __init__(self, filename, *channel_tuple, 
                 l_freq=10.0, h_freq=80.0, notch_freq=50.0, fs=500.0, time_window=time_window,
                 apply_ica=ica, remove_muscle=False,
                 eog_vertical_chs=('E14', 'E21'), eog_horizontal_chs=('E1', 'E32')):
        
        self.filename = filename
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.notch_freq = notch_freq
        self.time_window = time_window
        self.fs = fs
        self.active_channels = channel_tuple[0]
        self.bad_channels = channel_tuple[1]
        self.apply_ica = apply_ica
        self.remove_muscle = remove_muscle
        self.eog_vertical_chs = list(eog_vertical_chs)
        self.eog_horizontal_chs = list(eog_horizontal_chs)
        self.ica = None  # Store for inspection later

        self.raw = self.file_process()
        self.annotations = self.raw.annotations

    def file_process(self):
        raw = mne.io.read_raw_egi(self.filename, preload=True)
        
        if 'VREF' in raw.ch_names:
            raw.drop_channels(['VREF'])


        
        raw.pick('eeg')
        raw.pick_channels(self.active_channels)

        
        # if self.bad_channels:
        #     raw.drop_channels([ch for ch in self.bad_channels if ch in raw.ch_names])

  

        # Filter BEFORE ICA (ICA needs broadband signal to detect artifacts)
        # Use 1Hz high-pass for ICA fitting even if analysis band is higher
        raw.notch_filter(freqs=self.notch_freq, picks='eeg', verbose=False, pad='edge')
        raw.filter(l_freq=1.0, h_freq=self.h_freq, picks='eeg', verbose=False, pad='edge')

        if self.apply_ica:
            raw = self._run_ica(raw)

        # Apply analysis band-pass AFTER ICA (if l_freq > 1.0)
        if self.l_freq > 1.0:
            raw.filter(l_freq=self.l_freq, h_freq=None, picks='eeg', verbose=False, pad='edge')

        # Average reference AFTER ICA
        raw.set_eeg_reference('average', projection=False)

        return raw

    def _run_ica(self, raw):
        """
        Adds EOG proxies, fits ICA, removes artifact components, 
        then strips proxy channels. Returns cleaned raw (EEG only).
        """
        # --- 1. Add EOG proxy channels temporarily ---
        eog_proxies_added = []

        vert_chs = [ch for ch in self.eog_vertical_chs if ch in raw.ch_names]
        if vert_chs:
            proxy = raw.copy().pick_channels(vert_chs).get_data().mean(axis=0)
            info = mne.create_info(['EOG_vertical'], raw.info['sfreq'], ch_types=['eog'])
            raw.add_channels([mne.io.RawArray(proxy[np.newaxis, :], info)], force_update_info=True)
            eog_proxies_added.append('EOG_vertical')

        horiz_chs = [ch for ch in self.eog_horizontal_chs if ch in raw.ch_names]
        if horiz_chs:
            proxy = raw.copy().pick_channels(horiz_chs).get_data().mean(axis=0)
            info = mne.create_info(['EOG_horizontal'], raw.info['sfreq'], ch_types=['eog'])
            raw.add_channels([mne.io.RawArray(proxy[np.newaxis, :], info)], force_update_info=True)
            eog_proxies_added.append('EOG_horizontal')

        # --- 2. Fit ICA on EEG channels only (not proxies) ---
        eeg_only = raw.copy().pick_types(eeg=True)
        rank = mne.compute_rank(eeg_only, tol=1e-6, tol_kind='relative')
        n_components = min(25, rank['eeg'])

        print(f"\n🔧 Fitting ICA with {n_components} components on {len(eeg_only.ch_names)} EEG channels...")
        ica = mne.preprocessing.ICA(
            n_components=n_components, 
            random_state=42,
            method='fastica', 
            max_iter=500
        )
        ica.fit(eeg_only)

        # --- 3. Detect bad components ---
        bad_components = []

        if 'EOG_vertical' in raw.ch_names:
            idx, _ = ica.find_bads_eog(raw, ch_name='EOG_vertical', threshold=2.5)
            print(f"  Vertical EOG (blinks): {idx}")
            bad_components.extend(idx)

        if 'EOG_horizontal' in raw.ch_names:
            idx, _ = ica.find_bads_eog(raw, ch_name='EOG_horizontal', threshold=2.5)
            print(f"  Horizontal EOG (saccades): {idx}")
            bad_components.extend(idx)

        if self.remove_muscle:
            try:
                idx, _ = ica.find_bads_muscle(raw, threshold=0.2)
                print(f"  Muscle artifacts: {idx}")
                bad_components.extend(idx)
            except Exception as e:
                print(f"  Muscle detection skipped: {e}")

        ica.exclude = sorted(set(bad_components))
        print(f"\n  Excluding {len(ica.exclude)}/{n_components} components: {ica.exclude}")
        self.ica = ica  # Save for later inspection

        # --- 4. Apply ICA to EEG-only copy, then re-attach annotations ---
        # Apply only to EEG channels (proxy channels are NOT passed to apply)
        raw_eeg_clean = ica.apply(eeg_only)  # operates on the eeg-only copy
        
        # Restore annotations (crop/copy loses them)
        raw_eeg_clean.set_annotations(raw.annotations)

        print(f"  ✅ ICA done. Final channel count: {len(raw_eeg_clean.ch_names)}")
        return raw_eeg_clean  # Pure EEG, proxies never re-added

    # def baseline_stats(self):
    #     """Extract baseline statistics."""

    #     try:
    #         tmin = [ann['onset'] for ann in self.annotations if ann['description'] == 'BLOS'][0]
    #         tmax = [ann['onset'] for ann in self.annotations if ann['description'] == 'BLOE'][0]

    #         baseline_raw = self.raw.copy().crop(tmin = tmin, tmax = tmax)
    #         baseline_data = baseline_raw.get_data(picks = 'eeg')

    #         # median = np.median(baseline_data, axis=1, keepdims=True)
    #         # scale = iqr(baseline_data, axis=1, keepdims=True)

    #         mean = np.mean(baseline_data, axis = 1 , keepdims = True)
    #         std  = np.std(baseline_data, axis = 1, keepdims = True)

    
    #     except Exception as e:
    #         tmin = [ann['onset'] for ann in self.annotations if ann['description'] == 'BSST'][0]
    #         tmax = [ann['onset'] for ann in self.annotations if ann['description'] == 'BSEN'][0]
    #         baseline_raw = self.raw.copy().crop(tmin = tmin, tmax = tmax)
    #         baseline_data = baseline_raw.get_data(picks = 'eeg')

    #         # median = np.median(baseline_data, axis=1, keepdims=True)
    #         # scale = iqr(baseline_data, axis=1, keepdims=True)

    #         mean = np.mean(baseline_data, axis = 1 , keepdims = True)
    #         std  = np.std(baseline_data, axis = 1, keepdims = True)

    #     return mean, std
    def baseline_stats(self):
        """Extract baseline statistics with multiple marker fallbacks."""
        baseline_data = None
        
        # Define pairs of (start, end) markers to look for
        marker_pairs = [('BLOS', 'BLOE'), ('BSST', 'BSEN')]
        
        for start_mark, end_mark in marker_pairs:
            try:
                tmin = [ann['onset'] for ann in self.annotations if ann['description'] == start_mark]
                tmax = [ann['onset'] for ann in self.annotations if ann['description'] == end_mark]
                baseline_raw = self.raw.copy().crop(tmin=tmin, tmax=tmax)
                baseline_data = baseline_raw.get_data(picks='eeg')
                print(f"✅ Using {start_mark}/{end_mark} for normalization.")
                break # Exit loop if found
            except (IndexError, Exception):
                continue

        # Final Fallback: If no markers found, use the whole recording
        if baseline_data is None:
            print("⚠️ No baseline markers found. Using full recording for normalization.")
            baseline_data = self.raw.get_data(picks='eeg')

        # Calculate stats channel-wise
        mean = np.mean(baseline_data, axis=1, keepdims=True)
        std  = np.std(baseline_data, axis=1, keepdims=True)

        # CRITICAL: Prevent division by zero for flat/dead channels
        std[std == 0] = 1.0 

        return mean, std
        

    def extracting_data(self, start_offset=0.0, end_offset=0.0, overlap_factor=0.75, normalize = True):
        base_mean, base_std = self.baseline_stats()
        #base_mean, base_std = 0, 1
        #classes = ['BA', 'BY', 'DO', 'MO', 'SI']
        classes = ['MO', 'BY', 'SI']
        # Changed from flat list to a dictionary grouped by class
        trial_groups = {cls: [] for cls in classes} 

        for cls in classes:
            starts = [ann['onset'] for ann in self.annotations if ann['description'] == f'IS{cls}']
            ends   = [ann['onset'] for ann in self.annotations if ann['description'] == f'IE{cls}']

            for start, end in zip(starts, ends):
                segment = self.raw.copy().crop(tmin=start+start_offset, tmax=end+end_offset)
                data = segment.get_data(picks='eeg').astype(np.float32)

                if normalize is not None:
                    data = (data - base_mean)/base_std

                window_samples = int(self.time_window * self.fs)
                step_samples = int(window_samples * (1-overlap_factor))
                
                total_samples = data.shape[1]
                this_trial_windows = []
                
                for start_pt in range(0, total_samples - window_samples + 1, step_samples):
                    chunk = data[:, start_pt:start_pt + window_samples]
                    this_trial_windows.append(chunk)

                if this_trial_windows:
                    # Store as a tuple: (Array of Windows, Label)
                    X_windows = np.stack(this_trial_windows, axis=0)
                    y_windows = np.full(X_windows.shape[0], label_dict[f'IM{cls}'])
                    trial_groups[cls].append((X_windows, y_windows))

        return trial_groups




# Point this to the parent "Data" directory
base_dir = "/Users/kavinfidel/Desktop/GNN+CNS+Hopf/CNS_Lab/VM_EEG/Data"
#base_dir = "/home/kavinfidel/projects/VM_EEG/Data"
#base_dir = "/Users/kavinfidel/Desktop/GNN+CNS+Hopf/CNS_Lab/VM_EEG/data_2"
subject_dirs = {}

# 1. Get all items in the Data folder
# 2. Filter for directories that start with 'S'
sub_folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f)) and f.startswith('S')]

for folder in sub_folders:
    folder_path = os.path.join(base_dir, folder)
    files = []
    
    # List all .mff files within each subject's folder
    for file_name in os.listdir(folder_path):
        if not file_name.startswith('.') and file_name.endswith('.mff'):
            files.append(file_name)
    
    # Using the actual folder name (e.g., 'S1', 'S113') as the key
    subject_dirs[folder] = files

# Verification
print(f"Found {len(subject_dirs)} subjects.")
print("Subjects identified:", list(subject_dirs.keys()))

total_data = {}
test_data = {}

#base_dir = r'D:\0001_AK\KRISHNA\001_EEG_work_recent\extracted_files'


for subject, files in subject_dirs.items(): # subject is id, files are the all the files associated with a subject
    print(f"Processing {subject}")
    
    total_data[f"{subject}"] = {} #?
    test_data[f"{subject}"] = {}

    signals = [] #?
    labels = []#?
    signals_test = []
    labels_test = []

    k = 0
    for file_name in files:
        k +=1
        file_path = os.path.join(base_dir,subject, file_name) # grabbing file path, the mff file?
        
        if not file_name.endswith('.mff'):
            print(f"Skipping non-raw file: {file_name}")
            continue
        
        required_parts = ["signal1.bin", "info1.xml"]
        missing_parts = [p for p in required_parts if not os.path.exists(os.path.join(file_path, p))] # wha tis happenign here?
        if missing_parts:
            print(f"Skipping {file_name} due to parts being missing")
            continue
        
        print(f"File is intact: {file_name}\n Beginning extraction...")
        

        try:
            processor = preprocessing_pipeline(file_path, *channel_tuple)
            # trial_data is now a dict: {'BA': [(win, lab), (win, lab), (win, lab)], ...}
            trial_data = processor.extracting_data()

            if k == 2:
                print(f"Splitting Block {k} into Training and Test...")
                for cls, trials in trial_data.items():
                    # 1. Take the LAST trial (image event) for Testing
                    test_trial_x, test_trial_y = trials.pop() 
                    signals_test.append(test_trial_x)
                    labels_test.append(test_trial_y)

                    # 2. Put the REMAINING trials from this block into Training
                    for x, y in trials:
                        signals.append(x)
                        labels.append(y)
            else:
                # For Block 1 or 3, just put everything into Training
                for cls, trials in trial_data.items():
                    for x, y in trials:
                        signals.append(x)
                        labels.append(y)

        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            continue

    
    total_data[f"{subject}"]['data'] = np.concatenate(signals, axis=0)
    total_data[f"{subject}"]['labels'] = np.concatenate(labels, axis=0)   
    
    test_data[f"{subject}"]['data'] = np.concatenate(signals_test, axis=0)
    test_data[f"{subject}"]['labels'] = np.concatenate(labels_test, axis=0)  



for subject in total_data.keys():
    data = total_data[subject]['data']
    labels = total_data[subject]['labels']

    print(f"--- Verification for {subject} ---")
    print(f"Data Shape: {data.shape}") 
    # Expected: (Total_Windows, Channels, Samples_per_Window)
    # Example: (1500, 128, 50) 

    print(f"Labels Shape: {labels.shape}")
    print(f"Unique Labels: {np.unique(labels)}")

    # Check if classes are balanced (should be roughly equal)
    unique, counts = np.unique(labels, return_counts=True)
    print("Samples per class:", dict(zip(unique, counts)))
    print("-" * 30)
    

def riemannian_model_build(X_train,y_train,X_test, clf_type):
    cov_est = Covariances(estimator='oas')
    ts = TangentSpace(metric='riemann')
    #scaler = StandardScaler()
    
    # step 01
    cov_train = cov_est.fit_transform(X_train)
    cov_test = cov_est.transform(X_test)
    
    # Step 02
    X_train_ts = ts.fit_transform(cov_train)
    X_test_ts = ts.transform(cov_test)
    #scaler = StandardScaler()
    # St    ep 03
    #X_train_scaled = scaler.fit_transform(X_train_ts)
    #X_test_scaled = scaler.transform(X_test_ts)
    X_train_scaled = X_train_ts
    X_test_scaled = X_test_ts
    
    if clf_type == 'logreg':
        clf = LogisticRegression(
            penalty='l2',
            solver='lbfgs',
            class_weight='balanced',
            max_iter=1000
        )
    
    elif clf_type == 'svm':
        clf =   SVC(
            kernel='linear',
            class_weight='balanced',
            C=1.0
        )
    elif clf_type == 'xgboost':
        # Ensure y_train is 0-indexed for XGBoost
        clf = XGBClassifier(n_estimators=100, learning_rate=0.01, max_depth=10)
        
    elif clf_type == 'mlp':
        clf = MLPClassifier(
            hidden_layer_sizes=(16,),  # Small, single layer
            activation='relu',         # Standard, but 'tanh' can be smoother for small data
            solver='adam',             # Good default, but try 'lbfgs' for very small N
            alpha=0.01,                # L2 penalty (increase this if still overfitting)
            learning_rate_init=0.001,
            max_iter=1000,
            early_stopping=True,       # Prevents training too long
            validation_fraction=0.1,    # Percentage of data to use for early stopping
            random_state=42            # Ensure reproducibility
        )
    else:
        raise ValueError('No classifier model supported')
    
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    
    return y_pred        

def model_performance(train_dict, test_dict, clf_type, class_names=['MO', 'BY', 'SI']):
    total_acc_results = {}
    
    # We iterate over subjects in the training dictionary
    for subject in train_dict.keys():
        # Safety check: make sure this subject actually has a test block
        if subject not in test_dict or 'data' not in test_dict[subject]:
            print(f"Skipping {subject}: No test data found.")
            continue

        print(f"--- Evaluating {subject} ---")
        
        # 1. Prepare Training Data (all blocks except the 3rd)
        X_train = train_dict[subject]['data']
        y_train = train_dict[subject]['labels']
        
        # 2. Prepare Test Data (the 3rd block we kept separate)
        X_test = test_dict[subject]['data']
        y_test = test_dict[subject]['labels']
        
        print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")

        # 3. Predict 
        # We don't need the loop for K-Fold anymore because we have a fixed test set
        y_pred = riemannian_model_build(X_train, y_train, X_test, clf_type)
        
        # 4. Calculate Accuracy
        acc = accuracy_score(y_test, y_pred)
        total_acc_results[subject] = acc
        print(f"Final Test Accuracy: {acc:.4f}")

        # 5. Plot Confusion Matrix for this specific test block
        cm = confusion_matrix(y_test, y_pred, normalize='true')
        
        fig, ax = plt.subplots(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(cmap='Blues', ax=ax, values_format='.2g')
        plt.title(f"Hold-out Test: {subject} ({clf_type})")
        
        # Save and Show
        #save_path = f"/Users/kavinfidel/Desktop/GNN+CNS+Hopf/CNS_Lab/VM_EEG/confusion_matrices/holdout_{clf_type}_{subject}.png"
        #plt.savefig(save_path)
        plt.show()
    
    return total_acc_results

logreg_acc_dict = model_performance(total_data,test_data, clf_type='logreg')

svm_acc_dict = model_performance(total_data, test_data,clf_type='svm')