
import os
import mne
import numpy as np
from scipy.stats import iqr

class preprocessing_pipeline:
    def __init__(self, filename, *channel_tuple, 
                 l_freq=8.0, h_freq=30.0, notch_freq=50.0, fs=500.0, time_window=1.0,
                 apply_ica=True, remove_muscle=True,
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

        if self.bad_channels:
            raw.drop_channels([ch for ch in self.bad_channels if ch in raw.ch_names])


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
            idx, _ = ica.find_bads_eog(raw, ch_name='EOG_vertical', threshold=1.5)
            print(f"  Vertical EOG (blinks): {idx}")
            bad_components.extend(idx)

        if 'EOG_horizontal' in raw.ch_names:
            idx, _ = ica.find_bads_eog(raw, ch_name='EOG_horizontal', threshold=1.5)
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
    
    def baseline_stats(self):
        """Extract baseline statistics."""

        try:
            tmin = [ann['onset'] for ann in self.annotations if ann['description'] == 'BLCS'][0]
            tmax = [ann['onset'] for ann in self.annotations if ann['description'] == 'BLCE'][0]

            baseline_raw = self.raw.copy().crop(tmin = tmin, tmax = tmax)
            baseline_data = baseline_raw.get_data(picks = 'eeg')

            median = np.median(baseline_data, axis=1, keepdims=True)
            scale = iqr(baseline_data, axis=1, keepdims=True)

            return median, scale

        except Exception as e:
            print(f"No baseline beta:{e}")
            return 0, 1
        
    def extracting_data(self, start_offset=0.0, end_offset=0.0, overlap_factor=0.7, normalize = True):
        base_mean, base_std = self.baseline_stats()
        
        #classes = ['BA', 'BY', 'DO', 'MO', 'SI']
        classes = ['BA', 'DO', 'SI']
        # Changed from flat list to a dictionary grouped by class
        trial_groups = {cls: [] for cls in classes} 

        for cls in classes:
            starts = [ann['onset'] for ann in self.annotations if ann['description'] == f'OS{cls}']
            ends   = [ann['onset'] for ann in self.annotations if ann['description'] == f'OE{cls}']

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
                    y_windows = np.full(X_windows.shape[0], label_dict[f'OB{cls}'])
                    trial_groups[cls].append((X_windows, y_windows))

        return trial_groups