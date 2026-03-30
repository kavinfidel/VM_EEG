"""
EEG Data Processing and Dataset Module
This module provides functions for loading EEG data from MAT and TSV files,
applying sliding window segmentation, and creating PyTorch datasets.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
import scipy.io as sio
import numpy as np
from typing import Tuple, Optional, Union, List


def load_tsv_as_string(file_path: str, skip_header: bool = False, dtype: type = str) -> Optional[np.ndarray]:
    """
    Load TSV file and return as string array using NumPy.
    
    Parameters:
        file_path (str): Path to the TSV file
        skip_header (bool): Whether to skip the header row, defaults to False
        dtype (type): Data type, defaults to str

    Returns:
        np.ndarray: Numpy array containing TSV data, or None if error occurs
    """
    try:
        # Load TSV file using numpy.loadtxt, specifying tab as delimiter
        data = np.loadtxt(
            file_path,
            dtype=dtype,
            delimiter="\t",
            skiprows=1 if skip_header else 0,
            comments=None,  # Allow # symbols in data
        )
        print(f"Successfully loaded TSV file: {file_path}, shape: {data.shape}")
        return data
    except FileNotFoundError:
        print(f"Error: File not found '{file_path}'")
        return None
    except ValueError as ve:
        print(f"Error: Data format error - {ve}")
        return None
    except PermissionError:
        print(f"Error: No permission to read file '{file_path}'")
        return None
    except Exception as e:
        print(f"Error: Unexpected error occurred while loading file: {e}")
        return None


def load_mat_data(file_path: str, data_key: str = "EEG") -> Optional[np.ndarray]:
    """
    Load MAT file and extract data.
    
    Parameters:
        file_path (str): Path to the MAT file
        data_key (str): Key for the data in the MAT file, defaults to "EEG"
        
    Returns:
        numpy.ndarray: Loaded data, or None if error occurs
    """
    try:
        mat_data = sio.loadmat(file_path)
        # Extract data using the specified key
        data = np.array(mat_data[data_key])
        print(f"Successfully loaded MAT file: {file_path}, shape: {data.shape}")
        return data
    except KeyError:
        print(f"Error: Key '{data_key}' not found in MAT file {file_path}")
        return None
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None


def read_tsv_to_array(file_path: str, delimiter: str = '\t') -> Optional[np.ndarray]:
    """
    Read TSV file and convert to NumPy array.
    
    Parameters:
        file_path (str): Path to the TSV file
        delimiter (str): Delimiter, defaults to tab '\t'
        
    Returns:
        numpy.ndarray: Converted array, or None if error occurs
    """
    try:
        # Read file in string format
        labels = np.loadtxt(file_path, delimiter=delimiter, dtype=str)
        print(f"Successfully read label file: {file_path}, shape: {labels.shape}")
        return labels
    except Exception as e:
        print(f"Error reading file: {e}")
        return None


def sliding_window(X: np.ndarray, y: np.ndarray, sampling_rate: float, 
                  window_length_sec: float, step_sec: float, 
                  drop_last: bool = True) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Segment 3D EEG data into sliding window samples (time unit: seconds).
    
    Parameters:
        X : numpy.ndarray, shape (n_trials, n_channels, n_times)
            Input EEG data
        y : numpy.ndarray, shape (n_trials,)
            Labels corresponding to each trial
        sampling_rate : float
            Sampling rate (Hz), e.g., 200.0 means 200 data points per second
        window_length_sec : float
            Window length in seconds, e.g., 0.5 means 500ms
        step_sec : float
            Sliding step in seconds, e.g., 0.25 means 250ms
        drop_last : bool, default True
            Whether to discard the last incomplete window
            
    Returns:
        X_windowed : numpy.ndarray, shape (n_windows, n_channels, window_length_points)
            Windowed EEG data, or (False, False) if label count doesn't match trials
        y_windowed : numpy.ndarray, shape (n_windows,)
            Expanded labels, or (False, False) if label count doesn't match trials
    """
    # Convert seconds to time points
    window_length_points = int(round(window_length_sec * sampling_rate))
    step_points = int(round(step_sec * sampling_rate))

    # Parameter validation
    if X.ndim != 3:
        raise ValueError("Input data must be a 3D array (trials, channels, time)")
    
    n_trials, n_channels, n_times = X.shape
    
    # Check if label count matches trial count
    if y.shape[0] != n_trials:
        print(f"Warning: Label count ({y.shape[0]}) doesn't match trial count ({n_trials})")
        return False, False
    
    if window_length_points > n_times:
        raise ValueError(
            f"Window length ({window_length_sec}s = {window_length_points} points) "
            f"exceeds total time points ({n_times})"
        )
    
    if step_points < 1:
        raise ValueError("Step size must be ≥1 time point")

    # Calculate window start points (in points)
    starts = []
    current = 0
    while current + window_length_points <= n_times:
        starts.append(current)
        current += step_points

    # Handle last window
    if not drop_last:
        last_start = n_times - window_length_points
        if last_start not in starts and last_start >= 0:
            starts.append(last_start)

    # Remove duplicates and filter invalid start points
    starts = sorted(list(set(starts)))
    starts = [s for s in starts if s + window_length_points <= n_times]

    # Generate windowed data
    X_windowed = []
    y_windowed = []
    
    for trial_idx in range(n_trials):
        trial_data = X[trial_idx]  # (n_channels, n_times)
        label = y[trial_idx]

        for start in starts:
            end = start + window_length_points
            window = trial_data[:, start:end]  # (n_channels, window_length_points)
            X_windowed.append(window)
            y_windowed.append(label)

    return np.array(X_windowed), np.array(y_windowed)


class EEGDataset(Dataset):
    """
    EEG Dataset class for PyTorch.
    
    This class handles loading and preprocessing of EEG data for deep learning models.
    """
    
    def __init__(self, data: np.ndarray, labels: np.ndarray, augment: bool = False):
        """
        Initialize EEG Dataset.
        
        Parameters:
            data (np.ndarray): EEG data array
            labels (np.ndarray): Corresponding labels
            augment (bool): Whether to apply data augmentation, defaults to False
        """
        self.data = data
        self.labels = labels
        self.augment = augment  # Control whether to apply augmentation

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Parameters:
            idx (int): Index of the sample to retrieve
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (EEG data tensor, label tensor)
        """
        # Convert numpy array to torch tensor
        data_tensor = torch.from_numpy(self.data[idx]).float()
        
        # Convert label to torch tensor
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)

        # Apply data augmentation if enabled
        if self.augment:
            data_tensor = self._apply_augmentation(data_tensor)

        return data_tensor, label_tensor

    def _apply_augmentation(self, data_tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply data augmentation to EEG signal.
        
        Parameters:
            data_tensor (torch.Tensor): Input EEG data tensor
            
        Returns:
            torch.Tensor: Augmented EEG data tensor
        """
        # Example augmentation techniques for EEG data
        augmented_data = data_tensor.clone()
        
        # Add random Gaussian noise
        if random.random() < 0.5:
            noise = torch.randn_like(data_tensor) * 0.01
            augmented_data += noise
        
        # Random scaling
        if random.random() < 0.3:
            scale_factor = random.uniform(0.9, 1.1)
            augmented_data *= scale_factor
        
        # Random time shift (for temporal data)
        if data_tensor.dim() > 1 and random.random() < 0.4:
            shift = random.randint(-2, 2)
            augmented_data = torch.roll(augmented_data, shifts=shift, dims=-1)
        
        return augmented_data

    def get_class_distribution(self) -> dict:
        """
        Get the distribution of classes in the dataset.
        
        Returns:
            dict: Dictionary with class counts
        """
        unique, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(unique, counts))

    def get_data_shape(self) -> tuple:
        """
        Get the shape of the EEG data.
        
        Returns:
            tuple: Shape of the data (n_samples, n_channels, n_timesteps)
        """
        return self.data.shape