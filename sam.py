import mne
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
import mne
import numpy as np
from scipy.stats import iqr
import warnings
import torch
from torch_geometric.data import Data
from utils.processing_pipeline import preprocessing_pipeline
from utils.data import convert_to_graph_list
from utils.train_loop import train_riemannian_gnn
from utils.test_loop import evaluate_riemannian_gnn
mne.set_log_level('ERROR')   # silence MNE

warnings.filterwarnings("ignore")  # silence warnings



latest_channel_list = [
    # Left sensorimotor area channels
    'E29', 'E30', 'E35', 'E36', 'E41', 'E42',
    # Right sensorimotor area channels
    'E103', 'E104', 'E109', 'E110', 'E115', 'E116',
    # Mid-parietal & bilateral parietal
    'E62', 'E67', 'E72', 'E77'
 ]

new_latest = ['E24', 'E124', 'E36', 'E104', 'E47','E52', ' E60', 'E67', 'E67', 'E72', 'E77', 'E85', 'E92', 'E98', 'E62','E70', 'E75', 'E83','E58','E96','E90','E65','E69','E74','E82','E89'
              ]

bad_channels = ['E17', 'E38', 'E94', 'E113', 'E119', 'E121', 'E125', 'E128', 'E73', 'E81', 'E88', 'E43', 'E44', 'E120', 'E114','E127', 'E126',
                 'E68', 'E23', 'E3','E49','E48', "E8", "E25",
     "E56", "E63", "E99", "E107"]


                 
#bad_channels = ['E48', 'E119', 'E49', 'E113', 'E94', 'E68', 'E23', 'E3', 'E126', 'E127']



#label_dict = {'OBBA': 0, 'OBBY': 1, 'OBDO': 2, 'OBMO': 3, 'OBSI':4}
label_dict = {'IMBA': 0,'IMDO': 1,'IMSI':2 } # banana, dog, sitar
directions = ['IMBA', 'IMDO', 'IMSI']
channel_tuple = (new_latest, bad_channels)


import os

# Point this to the parent "Data" directory
base_dir = "/home/kavinfidel/projects/VM_EEG/Data 11-15-19-255"

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
    break


device = torch.device('cuda')


for subject in total_data.keys():
    model, test_loader = train_riemannian_gnn(total_data, test_data, subject)

    y_true, y_pred = evaluate_riemannian_gnn(model, test_loader, device)

