# import numpy as np
# import pickle
# import copy
# import matplotlib.pyplot as plt
# import os
# from sklearn.preprocessing import MinMaxScaler

# # --- CONFIGURATION ---
# INPUT_PATH = 'data/final_models_input/train_dataset.pkl'
# # We will save the anomalies file here
# OUTPUT_PATH = 'data/final_models_input/train_dataset_with_anomalies.pkl'
# # We will save the Scaler here (Crucial!)
# SCALER_PATH = 'data/final_models_input/anomalies_scaler.pkl'

# # Feature Indices
# IDX_POWER = 0
# IDX_GV = 1
# IDX_PRESS_DOWN = 2 
# IDX_PRESS_UP = 3

# # --- ANOMALY FUNCTIONS ---

# def anomaly_seal_leakage(signal, severity=15.0):
#     T = len(signal)
#     start_idx = int(T * 0.6)
#     ramp = np.linspace(0, severity, T - start_idx)
#     anom_signal = signal.copy()
#     anom_signal[start_idx:] += ramp
#     return anom_signal

# def anomaly_sensor_offset(signal, magnitude=-10.0):
#     return signal + magnitude

# def anomaly_sensor_drift(signal, magnitude=5.0):
#     T = len(signal)
#     drift = np.linspace(0, magnitude, T)
#     return signal + drift

# # --- MAIN GENERATION SCRIPT ---

# def generate_synthetic_test_set():
#     # 1. Load Data
#     print(f"Loading raw data from {INPUT_PATH}...")
#     with open(INPUT_PATH, 'rb') as f:
#         data = pickle.load(f)
    
#     # Extract Raw Data (N, 1500, 4)
#     # We assume this is raw (physical units) because we haven't scaled it yet
#     X_raw = data['X'] 
    
#     # === NEW: FIT AND SAVE SCALER ===
#     print("Fitting MinMaxScaler on clean training data...")
#     scaler = MinMaxScaler()
    
#     # We must flatten the data to 2D to fit the scaler: (N_samples * Time_steps, N_features)
#     N, T, F = X_raw.shape
#     X_flat = X_raw.reshape(-1, F)
    
#     scaler.fit(X_flat)
    
#     # Save the scaler immediately
#     with open(SCALER_PATH, 'wb') as f:
#         pickle.dump(scaler, f)
#     print(f"Scaler saved to {SCALER_PATH}")
#     # ================================
    
#     # 2. Generate Anomalies
#     X_anom = []
#     y_labels = [] # 0=Normal, 1=Leak, 2=Offset, 3=Drift
#     descriptions = []
    
#     print(f"Generating anomalies on {len(X_raw)} base samples...")
    
#     for i in range(len(X_raw)):
#         sample = X_raw[i]
        
#         # --- Case 0: Normal (Keep original) ---
#         X_anom.append(sample)
#         y_labels.append(0)
#         descriptions.append("Normal")
        
#         # --- Case 1: Seal Leakage (Pressure Down) ---
#         sample_leak = sample.copy()
#         # Adding +15 Bar ramp
#         sample_leak[:, IDX_PRESS_DOWN] = anomaly_seal_leakage(sample_leak[:, IDX_PRESS_DOWN], severity=15.0) 
#         X_anom.append(sample_leak)
#         y_labels.append(1)
#         descriptions.append("Seal Leakage")
        
#         # --- Case 2: Sensor Offset (Pressure Up) ---
#         sample_offset = sample.copy()
#         # Subtracting 10 Bar
#         sample_offset[:, IDX_PRESS_UP] = anomaly_sensor_offset(sample_offset[:, IDX_PRESS_UP], magnitude=-10.0) 
#         X_anom.append(sample_offset)
#         y_labels.append(2)
#         descriptions.append("Sensor Offset")
        
#         # --- Case 3: Drift (Active Power) ---
#         sample_drift = sample.copy()
#         # Adding +5 MW drift
#         sample_drift[:, IDX_POWER] = anomaly_sensor_drift(sample_drift[:, IDX_POWER], magnitude=5.0) 
#         X_anom.append(sample_drift)
#         y_labels.append(3)
#         descriptions.append("Sensor Drift")

#     # 3. Convert to Numpy
#     X_anom = np.array(X_anom)
#     y_labels = np.array(y_labels)
    
#     print(f"Generation Complete. Final Shape: {X_anom.shape}")
    
#     # 4. Save the Anomalous Dataset (Still RAW values)
#     save_dict = {
#         "X": X_anom,
#         "y": y_labels,
#         "descriptions": descriptions
#     }
    
#     with open(OUTPUT_PATH, 'wb') as f:
#         pickle.dump(save_dict, f)
    
#     print(f"Saved anomalous dataset to {OUTPUT_PATH}")
#     return X_anom, y_labels

# # --- VISUALIZATION CHECK ---
# def visualize_anomalies(X, y):
#     classes = np.unique(y)
#     plt.figure(figsize=(15, 10))
    
#     for i, cls in enumerate(classes):
#         if cls == 0: continue 
        
#         idx = np.where(y == cls)[0][0]
#         original_idx = idx - cls 
        
#         plt.subplot(2, 2, i)
        
#         if cls == 1: 
#             plt.plot(X[original_idx, :, IDX_PRESS_DOWN], 'b', label='Normal')
#             plt.plot(X[idx, :, IDX_PRESS_DOWN], 'r--', label='Anomaly')
#             plt.title("Seal Leakage (Pressure Down)")
#         elif cls == 2: 
#             plt.plot(X[original_idx, :, IDX_PRESS_UP], 'b', label='Normal')
#             plt.plot(X[idx, :, IDX_PRESS_UP], 'r--', label='Anomaly')
#             plt.title("Offset (Pressure Up)")
#         elif cls == 3: 
#             plt.plot(X[original_idx, :, IDX_POWER], 'b', label='Normal')
#             plt.plot(X[idx, :, IDX_POWER], 'r--', label='Anomaly')
#             plt.title("Drift (Active Power)")
            
#         plt.legend()
#         plt.grid(True, alpha=0.3)
    
#     plt.tight_layout()
#     plt.show()

# if __name__ == "__main__":
#     X_gen, y_gen = generate_synthetic_test_set()
#     visualize_anomalies(X_gen, y_gen)

import numpy as np
import pickle
import copy
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
import random

# --- CONFIG ---
INPUT_PATH = 'data/final_models_input/train_dataset.pkl'
OUTPUT_PATH = 'data/final_models_input/train_dataset_with_ALL_anomalies.pkl'

# Indices
IDX_POWER = 0
IDX_GV = 1
IDX_P_DOWN = 2
IDX_P_UP = 3

# --- PREVIOUS ANOMALIES (Keep these) ---
def anomaly_seal_leakage(signal, severity=15.0):
    T = len(signal)
    start_idx = int(T * 0.6) # Tail
    ramp = np.linspace(0, severity, T - start_idx)
    noise = np.random.normal(0, 0.5, size=len(ramp))
    anom = signal.copy()
    anom[start_idx:] += (ramp + noise)
    return anom

def anomaly_delayed_closure(signal, delay_steps=30):
    anom = np.roll(signal, delay_steps)
    anom[:delay_steps] = signal[0]
    return anom

def anomaly_sensor_offset(signal, magnitude=-15.0):
    return signal + magnitude

def anomaly_sensor_drift(signal, magnitude=30.0):
    drift = np.linspace(0, magnitude, len(signal))
    return signal + drift

# --- NEW ANOMALIES (Closing Transient Specific) ---

def anomaly_water_hammer_spike(signal, intensity=1.5):
    """
    Simulates a dangerous Water Hammer (Pressure Spike).
    Logic: Finds the max peak of pressure and amplifies it.
    """
    anom = signal.copy()
    
    # Find the region where pressure is highest (The spike)
    # usually in the first 20-40% of the window for upstream, 
    # or right at the closure moment for downstream.
    
    # Let's target the peak
    peak_idx = np.argmax(anom)
    
    # Define a window around the peak to amplify
    window = 20 # +/- 20 steps
    start = max(0, peak_idx - window)
    end = min(len(anom), peak_idx + window)
    
    # Amplify the spike (e.g., multiply by 1.5)
    # We add an offset proportional to the value to preserve shape
    spike_shape = anom[start:end]
    amplification = spike_shape * (intensity - 1.0) 
    
    anom[start:end] += amplification
    return anom

def anomaly_jerky_movement(signal, freq=0.5, amp=0.05):
    """
    Simulates Stick-Slip or Hydraulic instability in Guide Vanes.
    Adds a sinusoidal wobble to the smooth closing curve.
    """
    T = len(signal)
    t = np.arange(T)
    
    # Create wobble (Sine wave)
    # freq controls how fast it wobbles
    # amp controls how strong the wobble is (relative to signal magnitude)
    wobble = np.sin(2 * np.pi * freq * t) * amp
    
    # Add wobble only during the transition (where signal is changing)
    # We estimate transition roughly by checking where derivative is non-zero
    # Or just apply to the whole curve for simplicity
    
    return signal + wobble

def anomaly_signal_dropout(signal, num_drops=3):
    """
    Simulates loose wiring / sensor failure.
    Randomly drops values to 0 for 1-2 time steps.
    """
    anom = signal.copy()
    T = len(signal)
    
    for _ in range(num_drops):
        idx = np.random.randint(5, T-5)
        # Drop to 0 (or a very low value relative to range)
        anom[idx] = 0.0
        anom[idx+1] = 0.0 # 2-step dropout
        
    return anom

# --- GENERATION LOOP ---

def generate_comprehensive_anomalies():
    print(f"Loading Raw Train Data from {INPUT_PATH}...")
    with open(INPUT_PATH, 'rb') as f:
        data = pickle.load(f)
    X_raw = data['X'][-500:] 
    
    X_anom = []
    y_labels = [] 
    descriptions = []
    
    print("Generating comprehensive anomaly set...")
    
    for sample in X_raw:
        # 0. Normal
        X_anom.append(sample)
        y_labels.append(0)
        descriptions.append("Normal")
        
        # 1. Seal Leakage (Pressure Down)
        s1 = sample.copy()
        s1[:, IDX_P_DOWN] = anomaly_seal_leakage(s1[:, IDX_P_DOWN])
        X_anom.append(s1)
        y_labels.append(1)
        descriptions.append("Seal Leakage")
        
        # 2. Delayed Closure (All Signals, primarily P_Down)
        s2 = sample.copy()
        s2[:, IDX_P_DOWN] = anomaly_delayed_closure(s2[:, IDX_P_DOWN])
        X_anom.append(s2)
        y_labels.append(2)
        descriptions.append("Delayed Closure")
        
        # 3. Sensor Offset (Pressure Up)
        s3 = sample.copy()
        s3[:, IDX_P_UP] = anomaly_sensor_offset(s3[:, IDX_P_UP])
        X_anom.append(s3)
        y_labels.append(3)
        descriptions.append("Sensor Offset")
        
        # 4. Sensor Drift (Power)
        s4 = sample.copy()
        s4[:, IDX_POWER] = anomaly_sensor_drift(s4[:, IDX_POWER])
        X_anom.append(s4)
        y_labels.append(4)
        descriptions.append("Sensor Drift")
        
        # --- NEW ANOMALIES ---
        
        # 5. Water Hammer Spike (Pressure Up) - Critical Safety
        s5 = sample.copy()
        # Increase spike by 30% (dangerous pressure)
        s5[:, IDX_P_UP] = anomaly_water_hammer_spike(s5[:, IDX_P_UP], intensity=1.3) 
        X_anom.append(s5)
        y_labels.append(5)
        descriptions.append("Water Hammer Spike")
        
        # 6. Jerky Movement (Guide Vane) - Mechanical/Hydraulic
        s6 = sample.copy()
        # Add wobble (amplitude 0.05 is significant for GV which is 0-1)
        s6[:, IDX_GV] = anomaly_jerky_movement(s6[:, IDX_GV], freq=0.1, amp=0.05)
        X_anom.append(s6)
        y_labels.append(6)
        descriptions.append("Jerky GV Movement")
        
        # 7. Signal Dropout (Pressure Down) - Sensor Fault
        s7 = sample.copy()
        s7[:, IDX_P_DOWN] = anomaly_signal_dropout(s7[:, IDX_P_DOWN])
        X_anom.append(s7)
        y_labels.append(7)
        descriptions.append("Signal Dropout")

    # Save
    save_dict = {"X": np.array(X_anom), "y": np.array(y_labels), "descriptions": descriptions}
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(save_dict, f)
        
    print(f"Saved {len(X_anom)} samples to {OUTPUT_PATH}")
# --- VISUALIZATION CHECK ---
def visualize_anomalies(X, y):
    classes = np.unique(y)
    plt.figure(figsize=(15, 10))
    
    for i, cls in enumerate(classes):
        if cls == 0: continue 
        
        idx = np.where(y == cls)[0][0]
        original_idx = idx - cls 
        
        plt.subplot(2, 2, i)
        
        if cls == 1: 
            plt.plot(X[original_idx, :, IDX_P_DOWN], 'b', label='Normal')
            plt.plot(X[idx, :, IDX_P_DOWN], 'r--', label='Anomaly')
            plt.title("Seal Leakage (Pressure Down)")
        elif cls == 2: 
            plt.plot(X[original_idx, :, IDX_P_UP], 'b', label='Normal')
            plt.plot(X[idx, :, IDX_P_UP], 'r--', label='Anomaly')
            plt.title("Offset (Pressure Up)")
        elif cls == 3: 
            plt.plot(X[original_idx, :, IDX_POWER], 'b', label='Normal')
            plt.plot(X[idx, :, IDX_POWER], 'r--', label='Anomaly')
            plt.title("Drift (Active Power)")
            
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    generate_comprehensive_anomalies()