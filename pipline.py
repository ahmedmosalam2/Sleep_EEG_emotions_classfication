"""
Project: Predicting Emotions Using Brain Waves (EEG)
Target: Classify Neutral vs Emotional states using Theta Band Power
"""

import numpy as np
import h5py
import os
import pandas as pd
import glob
from scipy.signal import hilbert, butter, filtfilt
from scipy.stats import zscore
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import matplotlib

# إعداد محرك الرسم ليناسب العمل محلياً (Local GUI)
try:
    matplotlib.use('TkAgg')
except:
    pass

print("[1/6] All packages imported successfully")

# ---------------------------------------------------------
# 2. Configuration
# ---------------------------------------------------------
if os.path.exists('/kaggle/input'):
    TRAIN_PATH = '/kaggle/input/predicting-emotions-using-brain-waves/training'
    TEST_PATH = '/kaggle/input/predicting-emotions-using-brain-waves/testing'
else:
    # عدل هذه المسارات لتناسب جهازك
    TRAIN_PATH = './training'
    TEST_PATH = './testing'

# ---------------------------------------------------------
# 3. Data Loading and Feature Extraction Functions
# ---------------------------------------------------------
def load_hdf5_data(filepath):
    """تحميل ملفات الماتلاب والتعامل مع مراجع البيانات"""
    def load_field(f, data_ref, field_name):
        field = data_ref[field_name]
        if isinstance(field, h5py.Dataset):
            ref_value = field[()]
            if isinstance(ref_value, h5py.Reference):
                return f[ref_value]
            elif hasattr(ref_value, 'shape') and ref_value.shape == (1, 1):
                ref = ref_value.item()
                if isinstance(ref, h5py.Reference): return f[ref]
                else:
                    if isinstance(ref, bytes): ref = ref.decode('utf-8')
                    return f[ref]
        return field

    with h5py.File(filepath, 'r') as f:
        data_ref = f['data']
        trial_data = np.array(load_field(f, data_ref, 'trial')).T # (Trials, Channels, Time)
        time_data = np.array(load_field(f, data_ref, 'time')).flatten()
        
        try:
            trialinfo_data = load_field(f, data_ref, 'trialinfo')
            trialinfo = np.array(trialinfo_data).T
        except:
            trialinfo = None

        # تنظيف الوقت (البدء من ثانية 0)
        mask = time_data >= 0
        return {
            'trial': trial_data[:, :, mask],
            'trialinfo': trialinfo,
            'time': time_data[mask]
        }

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """تصميم وتطبيق فلتر بترورث"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=-1)

def extract_hilbert_power(data, freq_band=(4, 8), fs=200):
    """استخراج طاقة الإشارة باستخدام تحويل هيلبرت"""
    n_trials, n_channels, n_timepoints = data.shape
    data_filtered = np.zeros_like(data)
    
    # 1. Filtering
    for trial in range(n_trials):
        for ch in range(n_channels):
            data_filtered[trial, ch, :] = butter_bandpass_filter(
                data[trial, ch, :], freq_band[0], freq_band[1], fs
            )
            
    # 2. Hilbert Transform & Power
    analytic_signal = hilbert(data_filtered, axis=-1)
    power = np.abs(analytic_signal) ** 2
    return power

print("[2/6] Helper functions defined")

# ---------------------------------------------------------
# 4. Load & Process Training Data
# ---------------------------------------------------------
print("\n" + "="*50)
print("LOADING & PROCESSING DATA")
print("="*50)

neu_path = os.path.join(TRAIN_PATH, 'sleep_neu')
if not os.path.exists(neu_path):
    print(f"Error: Path {neu_path} not found!")
    exit()

train_files = sorted([f for f in os.listdir(neu_path) if f.endswith('.mat')])

train_data_list = []
train_labels_list = []
train_counts = []
time_vector = None

for subj_file in train_files:
    subj_id = subj_file.split('_')[1]
    print(f"Processing Participant {subj_id}...")

    # تحميل البيانات المحايدة والعاطفية
    neu = load_hdf5_data(os.path.join(TRAIN_PATH, 'sleep_neu', subj_file))
    emo = load_hdf5_data(os.path.join(TRAIN_PATH, 'sleep_emo', subj_file))

    # دمج البيانات وتحديد التصنيفات (1=Neutral, 2=Emotional)
    combined_trials = np.concatenate([neu['trial'], emo['trial']], axis=0)
    combined_labels = np.concatenate([np.ones(len(neu['trial'])), np.ones(len(emo['trial']))*2])

    # استخراج الميزات (Theta Power) وعمل Normalization
    power_feat = extract_hilbert_power(combined_trials)
    power_z = zscore(power_feat, axis=0)

    train_data_list.append(power_z)
    train_labels_list.append(combined_labels)
    train_counts.append(len(combined_labels))
    if time_vector is None: time_vector = neu['time']

train_data_all = np.concatenate(train_data_list, axis=0)
train_labels_all = np.concatenate(train_labels_list, axis=0)

print(f"\nTotal Trials: {train_data_all.shape[0]} | Shape: {train_data_all.shape}")

# ---------------------------------------------------------
# 5. LOOCV & Classification
# ---------------------------------------------------------
def classify_timepoint(train_X, train_y, test_X, test_y, t_idx):
    X_tr = np.nan_to_num(train_X[:, :, t_idx], 0)
    X_ts = np.nan_to_num(test_X[:, :, t_idx], 0)
    
    clf = LinearDiscriminantAnalysis()
    clf.fit(X_tr, train_y)
    
    try:
        probs = clf.predict_proba(X_ts)
        return roc_auc_score(test_y, probs[:, 1])
    except:
        return 0.5

print("\nRunning Leave-One-Participant-Out Cross-Validation...")
ranges = np.concatenate([[0], np.cumsum(train_counts)])
n_subj = len(train_counts)
n_times = train_data_all.shape[2]
loocv_results = np.zeros((n_subj, n_times))

for i in range(n_subj):
    print(f"Testing Subject {i+1}/{n_subj}...", end='\r')
    test_idx = np.arange(ranges[i], ranges[i+1])
    train_idx = np.delete(np.arange(len(train_labels_all)), test_idx)

    for t in range(n_times):
        loocv_results[i, t] = classify_timepoint(
            train_data_all[train_idx], train_labels_all[train_idx],
            train_data_all[test_idx], train_labels_all[test_idx], t
        )

# ---------------------------------------------------------
# 6. Final Visualization
# ---------------------------------------------------------
mean_auc = loocv_results.mean(axis=0)
sem_auc = loocv_results.std(axis=0) / np.sqrt(n_subj)

plt.figure(figsize=(10, 6))
plt.plot(time_vector, mean_auc, 'b-', label='Mean AUC (LOOCV)')
plt.fill_between(time_vector, mean_auc - sem_auc, mean_auc + sem_auc, alpha=0.2, color='blue')
plt.axhline(0.5, color='r', linestyle='--', label='Chance Level')
plt.xlabel('Time (s)')
plt.ylabel('AUC Score')
plt.title('Emotion Classification Performance Over Time')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("\nProcess Completed Successfully!")