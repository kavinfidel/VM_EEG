"""
EEG Imagery Classification Pipeline
=====================================
All experiment parameters are controlled from the ExperimentConfig dataclass below.
Change values there and re-run 
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import os
import warnings

import numpy as np
import matplotlib.pyplot as plt
import mne
import torch
from scipy.stats import iqr
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

mne.set_log_level('ERROR')
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT CONFIGURATION — tweak everything here
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExperimentConfig:
    # ── Data paths ────────────────────────────────────────────────────────────
    base_dir: str = "/Users/kavinfidel/Desktop/GNN+CNS+Hopf/CNS_Lab/VM_EEG/Data"

    # ── Epoch / windowing ─────────────────────────────────────────────────────
    time_window: float = 0.5        # seconds per window
    overlap_factor: float = 0.75    # fraction of window to overlap (0 = no overlap)
    start_offset: float = 0.5       # seconds to trim from trial start
    end_offset: float = 0.5         # seconds to trim from trial end

    # ── Signal processing ─────────────────────────────────────────────────────
    fs: float = 500.0               # sampling frequency (Hz)
    l_freq: float = 5.0            # high-pass cutoff for analysis band
    h_freq: float = 30.0            # low-pass cutoff
    notch_freq: float = 50.0        # notch filter (powerline)

    # ── ICA ───────────────────────────────────────────────────────────────────
    apply_ica: bool = True
    remove_muscle: bool = False
    ica_n_components: int = 25      # max ICA components (capped by rank)
    ica_eog_threshold: float = 3.5  # z-score threshold for EOG rejection
    ica_max_iter: int = 500

    # ── EOG proxy channels ────────────────────────────────────────────────────
    eog_vertical_chs: Tuple[str, ...] = ('E14', 'E21')
    eog_horizontal_chs: Tuple[str, ...] = ('E1', 'E32')

    # ── Channel selection ─────────────────────────────────────────────────────
    active_channels: List[str] = field(default_factory=lambda: [
        'E24', 'E124', 'E36', 'E104', 'E47', 'E52', 'E60', 'E67', 'E72', 'E77',
        'E85', 'E92', 'E98', 'E62', 'E70', 'E75', 'E83', 'E58', 'E96', 'E90',
        'E65', 'E69', 'E74', 'E82', 'E89', 'E1', 'E32', 'E14', 'E21'
    ])
    bad_channels: List[str] = field(default_factory=lambda: [
        'E17', 'E38', 'E94', 'E113', 'E119', 'E121', 'E125', 'E128',
        'E73', 'E81', 'E88', 'E43', 'E44', 'E120', 'E114', 'E127', 'E126',
        'E68', 'E23', 'E3', 'E49', 'E48', 'E8', 'E25', 'E56', 'E63', 'E99', 'E107',
    ])

    # ── Task / labels ─────────────────────────────────────────────────────────
    classes: List[str] = field(default_factory=lambda: ['BA', 'BY', 'DO', 'MO', 'SI'])
    label_dict: Dict[str, int] = field(default_factory=lambda: {
        'IMBA': 0, 'IMBY': 1, 'IMDO': 2, 'IMMO':3, 'IMSI': 4
    })

    # ── Normalisation ─────────────────────────────────────────────────────────
    normalize: bool = False

    # ── Classifiers to run ───────────────────────────────────────────────────
    # Options: 'logreg', 'svm', 'xgboost', 'mlp'
    classifiers: List[str] = field(default_factory=lambda: ['logreg', 'svm'])

    # ── Classifier hyper-parameters ──────────────────────────────────────────
    logreg_C: float = 1.0
    logreg_max_iter: int = 1000

    svm_C: float = 1.0
    svm_kernel: str = 'linear'

    xgb_n_estimators: int = 100
    xgb_learning_rate: float = 0.05
    xgb_max_depth: int = 3

    mlp_hidden_layers: Tuple[int, ...] = (16,)
    mlp_alpha: float = 0.01
    mlp_lr: float = 0.001
    mlp_max_iter: int = 1000

    # ── Covariance / tangent-space ────────────────────────────────────────────
    cov_estimator: str = 'oas'      # 'oas', 'scm', 'lwf', etc.
    ts_metric: str = 'riemann'      # 'riemann', 'euclid', 'logeuclid'

    # ── Output ────────────────────────────────────────────────────────────────
    save_confusion_matrices: bool = False
    output_dir: str = "/Users/kavinfidel/Desktop/GNN+CNS+Hopf/CNS_Lab/VM_EEG/confusion_matrices"
    show_plots: bool = True


# ─────────────────────────────────────────────────────────────────────────────
# Instantiate the config — only edit this object between runs
# ─────────────────────────────────────────────────────────────────────────────

cfg = ExperimentConfig(
    time_window=0.5,
    overlap_factor=0.75,
    l_freq=5.0,
    h_freq=30.0,
    apply_ica=True,
    classifiers=['logreg', 'svm'],
    cov_estimator='oas',
    ts_metric='riemann',

)
# 

# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing pipeline
# ─────────────────────────────────────────────────────────────────────────────

class PreprocessingPipeline:
    def __init__(self, filename: str, config: ExperimentConfig):
        self.filename = filename
        self.cfg = config
        self.ica_obj = None

        self.raw = self._file_process()
        self.annotations = self.raw.annotations

    def _file_process(self):
        cfg = self.cfg
        raw = mne.io.read_raw_egi(self.filename, preload=True)

        if 'VREF' in raw.ch_names:
            raw.drop_channels(['VREF'])

        raw.pick('eeg')
        # raw.pick_channels(cfg.active_channels)
        if cfg.bad_channels:
            raw.drop_channels([ch for ch in cfg.bad_channels if ch in raw.ch_names])        

        raw.notch_filter(freqs=cfg.notch_freq, picks='eeg', verbose=False, pad='edge')
        raw.filter(l_freq=1.0, h_freq=cfg.h_freq, picks='eeg', verbose=False, pad='edge')

        if cfg.apply_ica:
            raw = self._run_ica(raw)

        if cfg.l_freq > 1.0:
            raw.filter(l_freq=cfg.l_freq, h_freq=None, picks='eeg', verbose=False, pad='edge')

        raw.set_eeg_reference('average', projection=False)
        return raw

    def _run_ica(self, raw):
        cfg = self.cfg
        eog_proxies_added = []

        def _add_proxy(chs, name, kind):
            present = [ch for ch in chs if ch in raw.ch_names]
            if present:
                proxy = raw.copy().pick_channels(present).get_data().mean(axis=0)
                info = mne.create_info([name], raw.info['sfreq'], ch_types=[kind])
                raw.add_channels([mne.io.RawArray(proxy[np.newaxis, :], info)],
                                  force_update_info=True)
                eog_proxies_added.append(name)

        _add_proxy(list(cfg.eog_vertical_chs),   'EOG_vertical',   'eog')
        _add_proxy(list(cfg.eog_horizontal_chs),  'EOG_horizontal', 'eog')

        eeg_only = raw.copy().pick_types(eeg=True)
        rank = mne.compute_rank(eeg_only, tol=1e-6, tol_kind='relative')
        n_components = min(cfg.ica_n_components, rank['eeg'])

        print(f"🔧 ICA: {n_components} components on {len(eeg_only.ch_names)} channels …")
        ica = mne.preprocessing.ICA(
            n_components=n_components,
            random_state=42,
            method='fastica',
            max_iter=cfg.ica_max_iter,
        )
        ica.fit(eeg_only)

        bad = []
        for ch_name in ['EOG_vertical', 'EOG_horizontal']:
            if ch_name in raw.ch_names:
                idx, _ = ica.find_bads_eog(raw, ch_name=ch_name,
                                            threshold=cfg.ica_eog_threshold)
                print(f"  {ch_name}: {idx}")
                bad.extend(idx)

        if cfg.remove_muscle:
            try:
                idx, _ = ica.find_bads_muscle(raw, threshold=0.2)
                print(f"  Muscle: {idx}")
                bad.extend(idx)
            except Exception as e:
                print(f"  Muscle detection skipped: {e}")

        ica.exclude = sorted(set(bad))
        print(f"  Excluding {len(ica.exclude)}/{n_components} components: {ica.exclude}")
        self.ica_obj = ica

        raw_clean = ica.apply(eeg_only)
        raw_clean.set_annotations(raw.annotations)
        print(f"  ✅ ICA done. Channels: {len(raw_clean.ch_names)}")
        return raw_clean

    def baseline_stats(self):
        baseline_data = None
        for start_m, end_m in [('BLOS', 'BLOE'), ('BSST', 'BSEN')]:
            try:
                tmin = next(a['onset'] for a in self.annotations if a['description'] == start_m)
                tmax = next(a['onset'] for a in self.annotations if a['description'] == end_m)
                baseline_data = self.raw.copy().crop(tmin=tmin, tmax=tmax).get_data(picks='eeg')
                print(f"✅ Baseline: {start_m}/{end_m}")
                break
            except StopIteration:
                continue

        if baseline_data is None:
            print("⚠️  No baseline markers — using full recording.")
            baseline_data = self.raw.get_data(picks='eeg')

        mean = np.mean(baseline_data, axis=1, keepdims=True)
        std  = np.std(baseline_data,  axis=1, keepdims=True)
        std[std == 0] = 1.0
        return mean, std

    def extract_data(self):
        cfg = self.cfg

        if cfg.normalize:
            base_mean, base_std = self.baseline_stats()
        #base_mean, base_std = 0, 1
        trial_groups = {cls: [] for cls in cfg.classes}

        window_samples = int(cfg.time_window * cfg.fs)
        step_samples   = max(1, int(window_samples * (1 - cfg.overlap_factor)))

        for cls in cfg.classes:
            starts = [a['onset'] for a in self.annotations if a['description'] == f'IS{cls}']
            ends   = [a['onset'] for a in self.annotations if a['description'] == f'IE{cls}']

            for start, end in zip(starts, ends):
                seg = self.raw.copy().crop(
                    tmin=start + cfg.start_offset,
                    tmax=end   + cfg.end_offset
                )
                data = seg.get_data(picks='eeg').astype(np.float32)

                if cfg.normalize:
                    data = (data - base_mean) / base_std

                total_samples = data.shape[1]
                windows = [
                    data[:, s:s + window_samples]
                    for s in range(0, total_samples - window_samples + 1, step_samples)
                ]

                if windows:
                    X = np.stack(windows, axis=0)
                    y = np.full(X.shape[0], cfg.label_dict[f'IM{cls}'])
                    trial_groups[cls].append((X, y))

        return trial_groups


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_all_subjects(config: ExperimentConfig):
    base_dir = config.base_dir
    sub_folders = [
        f for f in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, f)) and f.startswith('S')
    ]
    print(f"Found {len(sub_folders)} subjects: {sub_folders}")

    total_data, test_data = {}, {}

    for subject in sub_folders:
        print(f"\n{'='*40}\nProcessing {subject}")
        folder_path = os.path.join(base_dir, subject)
        files = [f for f in os.listdir(folder_path)
                 if not f.startswith('.') and f.endswith('.mff')]

        signals, labels = [], []
        signals_test, labels_test = [], []

        for k, file_name in enumerate(files, start=1):
            file_path = os.path.join(folder_path, file_name)
            required = ["signal1.bin", "info1.xml"]
            if any(not os.path.exists(os.path.join(file_path, p)) for p in required):
                print(f"  Skipping {file_name}: missing parts")
                continue

            print(f"  [{k}] {file_name}")
            try:
                proc = PreprocessingPipeline(file_path, config)
                trial_data = proc.extract_data()

                if k == 2:
                    print("  Splitting block 2 → last trial → test set")
                    for cls, trials in trial_data.items():
                        tx, ty = trials.pop()
                        signals_test.append(tx)
                        labels_test.append(ty)
                        for x, y in trials:
                            signals.append(x); labels.append(y)
                else:
                    for cls, trials in trial_data.items():
                        for x, y in trials:
                            signals.append(x); labels.append(y)

            except Exception as e:
                print(f"  ⚠️  Error in {file_name}: {e}")

        if signals:
            total_data[subject] = {
                'data':   np.concatenate(signals,      axis=0),
                'labels': np.concatenate(labels,       axis=0),
            }
        if signals_test:
            test_data[subject] = {
                'data':   np.concatenate(signals_test, axis=0),
                'labels': np.concatenate(labels_test,  axis=0),
            }

    # Quick verification
    for subj in total_data:
        d, l = total_data[subj]['data'], total_data[subj]['labels']
        unique, counts = np.unique(l, return_counts=True)
        print(f"{subj}: train {d.shape} | classes {dict(zip(unique, counts))}")

    return total_data, test_data


# ─────────────────────────────────────────────────────────────────────────────
# Riemannian classifier
# ─────────────────────────────────────────────────────────────────────────────

def build_clf(clf_type: str, config: ExperimentConfig):
    cfg = config
    if clf_type == 'logreg':
        return LogisticRegression(
            penalty='l2', solver='lbfgs',
            C=cfg.logreg_C, max_iter=cfg.logreg_max_iter,
            class_weight='balanced'
        )
    if clf_type == 'svm':
        return SVC(kernel=cfg.svm_kernel, C=cfg.svm_C, class_weight='balanced')
    if clf_type == 'xgboost':
        return XGBClassifier(
            n_estimators=cfg.xgb_n_estimators,
            learning_rate=cfg.xgb_learning_rate,
            max_depth=cfg.xgb_max_depth,
        )
    if clf_type == 'mlp':
        return MLPClassifier(
            hidden_layer_sizes=cfg.mlp_hidden_layers,
            activation='relu', solver='adam',
            alpha=cfg.mlp_alpha, learning_rate_init=cfg.mlp_lr,
            max_iter=cfg.mlp_max_iter,
            early_stopping=True, validation_fraction=0.1,
            random_state=42,
        )
    raise ValueError(f"Unknown classifier: {clf_type!r}")


def riemannian_predict(X_train, y_train, X_test, clf_type: str, config: ExperimentConfig):
    cov = Covariances(estimator=config.cov_estimator)
    ts  = TangentSpace(metric=config.ts_metric)

    X_tr_ts = ts.fit_transform(cov.fit_transform(X_train))
    X_te_ts = ts.transform(cov.transform(X_test))

    clf = build_clf(clf_type, config)
    clf.fit(X_tr_ts, y_train)
    return clf.predict(X_te_ts)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation 
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_all(train_dict, test_dict, config: ExperimentConfig):
    class_names = config.classes
    all_results = {}   # { clf_type: { subject: acc } }

    for clf_type in config.classifiers:
        print(f"\n{'─'*50}\nClassifier: {clf_type.upper()}\n{'─'*50}")
        acc_results = {}

        for subject in train_dict:
            if subject not in test_dict or 'data' not in test_dict[subject]:
                print(f"  {subject}: no test data — skipping")
                continue

            X_train = train_dict[subject]['data']
            y_train = train_dict[subject]['labels']
            X_test  = test_dict[subject]['data']
            y_test  = test_dict[subject]['labels']

            print(f"  {subject} — train {len(X_train)} | test {len(X_test)}")

            y_pred = riemannian_predict(X_train, y_train, X_test, clf_type, config)
            acc = accuracy_score(y_test, y_pred)
            acc_results[subject] = acc
            print(f"  Accuracy: {acc:.4f}")

            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred, normalize='true')
            fig, ax = plt.subplots(figsize=(7, 5))
            ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
                cmap='Blues', ax=ax, values_format='.2g'
            )
            ax.set_title(f"{subject} | {clf_type} | acc={acc:.3f}")
            plt.tight_layout()

            if config.save_confusion_matrices:
                os.makedirs(config.output_dir, exist_ok=True)
                path = os.path.join(config.output_dir, f"cm_{clf_type}_{subject}.png")
                plt.savefig(path, dpi=150)
                print(f"  Saved → {path}")

            if config.show_plots:
                plt.show()
            else:
                plt.close()

        all_results[clf_type] = acc_results

    # Summary table
    print(f"\n{'='*50}\nSUMMARY\n{'='*50}")
    subjects = sorted({s for r in all_results.values() for s in r})
    header = f"{'Subject':<12}" + "".join(f"{c:>10}" for c in config.classifiers)
    print(header)
    print("─" * len(header))
    for subj in subjects:
        row = f"{subj:<12}"
        for clf_type in config.classifiers:
            acc = all_results[clf_type].get(subj, float('nan'))
            row += f"{acc:>10.4f}"
        print(row)

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    total_data, test_data = load_all_subjects(cfg)
    results = evaluate_all(total_data, test_data, cfg)
