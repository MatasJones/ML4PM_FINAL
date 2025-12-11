import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
import datetime

def ramp_anomaly(T = 100, sigma = 0.5, max_y=5.0):
    dx = max_y/(T/2)
    delta = np.arange(0,T).astype(float)
    f = lambda x: np.piecewise(x,
            [x < T//2, x >= T//2],
            [lambda x: dx*x,
            lambda x: -dx*(x-T)
        ])
    return f(delta) + np.random.normal(0, sigma, T)

def constant_anomaly(T = 100, sigma = 0.5, max_y=5.0):
    return np.random.normal(0, sigma, T) + max_y


# ==============================================================================
# WINDOW-BASED ANOMALY INJECTION FUNCTIONS FOR AUTOENCODER TESTING
# ==============================================================================
# These functions inject anomalies into 360-point standardized windows.
# Anomalies are restricted to the CLOSING SEQUENCE: indices [180, 360).
# Compatible with:
#   - Per-signal windows: np.ndarray shape (360,)
#   - Slices from combined windows: extracted 360-point signal slices
# ==============================================================================

def _sample_transition_centered_start(rng, closing_start, closing_end, segment_length, transition_center, spread):
    """
    Sample a start index biased towards the transition center using a truncated normal.
    
    Args:
        rng: numpy random generator
        closing_start: Start of closing sequence (180)
        closing_end: End of closing sequence (360)
        segment_length: Length of segment to place
        transition_center: Target center position for the anomaly
        spread: Standard deviation for the normal distribution (controls how spread out)
    
    Returns:
        start: Start index for the segment
    """
    max_start = closing_end - segment_length
    min_start = closing_start
    
    # Sample from truncated normal centered on transition_center
    # Adjust center to be the segment start (so segment center is at transition_center)
    target_start = transition_center - segment_length // 2
    
    # Sample with truncated normal
    for _ in range(100):  # Try up to 100 times
        sampled = int(rng.normal(target_start, spread))
        if min_start <= sampled <= max_start:
            return sampled
    
    # Fallback: clamp to valid range
    return max(min_start, min(max_start, target_start))


def inject_closing_spikes(
    window: np.ndarray,
    n_spikes: int,
    magnitude_range: tuple[float, float],
    random_state: int | None = None,
    transition_center: int = 200,
    spread: float = 20.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Inject point spike anomalies only in the closing sequence of a windowed time series.
    
    Anomalies are restricted to the closing sequence: indices [180, 360).
    Spike positions are biased towards the transition center (around index 180-220).
    Each spike replaces a single point with: local_mean + sign * factor * local_std
    
    Args:
        window: 1D array of shape (360,), assumed standardized (mean~0, std~1).
        n_spikes: Number of spike points to inject.
        magnitude_range: (min_factor, max_factor) - spike size in local std units.
        random_state: Random seed for reproducibility.
        transition_center: Target center for spike placement (default 200, near transition).
        spread: Standard deviation for position sampling (default 20).
    
    Returns:
        window_perturbed: Copy of window with spikes injected.
        spike_indices: 1D array of indices where spikes were injected.
    """
    rng = np.random.default_rng(random_state)
    window_perturbed = window.copy()
    
    # Closing sequence range
    closing_start = 180
    closing_end = 360
    
    # Sample spike indices biased towards transition center
    spike_indices = []
    for _ in range(min(n_spikes, closing_end - closing_start)):
        for _ in range(100):  # Try to find valid position
            idx = int(rng.normal(transition_center, spread))
            if closing_start <= idx < closing_end and idx not in spike_indices:
                spike_indices.append(idx)
                break
    spike_indices = np.array(spike_indices)
    
    local_half_window = 10  # ±10 points for local stats
    
    for idx in spike_indices:
        # Build local window, clipped to valid range
        local_start = max(0, idx - local_half_window)
        local_end = min(360, idx + local_half_window + 1)
        local_segment = window[local_start:local_end]
        
        local_mean = local_segment.mean()
        local_std = local_segment.std()
        if local_std == 0:
            local_std = 1.0  # Fallback for constant segments
        
        # Sample spike magnitude
        factor = rng.uniform(magnitude_range[0], magnitude_range[1])
        sign = rng.choice([-1, 1])
        
        # Inject spike
        window_perturbed[idx] = local_mean + sign * factor * local_std
    
    return window_perturbed, spike_indices


def inject_closing_level_shift(
    window: np.ndarray,
    segment_length: int,
    shift_factor: float,
    random_state: int | None = None,
    transition_center: int = 200,
    spread: float = 20.0
) -> tuple[np.ndarray, tuple[int, int], float]:
    """
    Inject a step change (level shift) in mean only within the closing sequence.
    
    Anomalies are restricted to the closing sequence: indices [180, 360).
    Segment placement is biased towards the transition center.
    A constant offset is added to a contiguous segment.
    
    Args:
        window: 1D array of shape (360,), assumed standardized.
        segment_length: Length of shifted segment, must be <= 180.
        shift_factor: Magnitude of shift in global std units.
        random_state: Random seed for reproducibility.
        transition_center: Target center for segment placement (default 200).
        spread: Standard deviation for position sampling (default 20).
    
    Returns:
        window_shifted: Modified copy with level shift.
        (start, end): Indices of the shifted segment.
        shift: The actual numeric value added.
    """
    rng = np.random.default_rng(random_state)
    window_shifted = window.copy()
    
    # Ensure segment fits in closing range [180, 360)
    segment_length = min(segment_length, 180)
    closing_start = 180
    closing_end = 360
    
    # Sample start index biased towards transition center
    start = _sample_transition_centered_start(rng, closing_start, closing_end, 
                                               segment_length, transition_center, spread)
    end = start + segment_length
    
    # Compute shift
    global_std = window.std()
    if global_std == 0:
        global_std = 1.0
    
    direction = rng.choice([-1, 1])
    shift = direction * shift_factor * global_std
    
    # Apply shift
    window_shifted[start:end] += shift
    
    return window_shifted, (start, end), shift


def inject_closing_linear_drift(
    window: np.ndarray,
    segment_length: int,
    drift_std_factors: tuple[float, float],
    random_state: int | None = None,
    transition_center: int = 200,
    spread: float = 20.0
) -> tuple[np.ndarray, tuple[int, int], float]:
    """
    Inject a gradual linear drift anomaly inside the closing sequence.
    
    Anomalies are restricted to the closing sequence: indices [180, 360).
    Segment placement is biased towards the transition center.
    A linearly increasing/decreasing offset is added to a contiguous segment.
    
    Args:
        window: 1D array of shape (360,), assumed standardized.
        segment_length: Length of drift segment, must be <= 180.
        drift_std_factors: (min_factor, max_factor) - total drift amplitude at 
                          segment end, in global std units.
        random_state: Random seed for reproducibility.
        transition_center: Target center for segment placement (default 200).
        spread: Standard deviation for position sampling (default 20).
    
    Returns:
        window_drifted: Modified copy with linear drift.
        (start, end): Indices of the drift segment.
        A: Final drift amplitude (value at end of segment).
    """
    rng = np.random.default_rng(random_state)
    window_drifted = window.copy()
    
    # Ensure segment fits in closing range [180, 360)
    segment_length = min(segment_length, 180)
    closing_start = 180
    closing_end = 360
    
    # Sample start index biased towards transition center
    start = _sample_transition_centered_start(rng, closing_start, closing_end, 
                                               segment_length, transition_center, spread)
    end = start + segment_length
    
    # Compute drift amplitude
    global_std = window.std()
    if global_std == 0:
        global_std = 1.0
    
    factor = rng.uniform(drift_std_factors[0], drift_std_factors[1])
    sign = rng.choice([-1, 1])
    A = sign * factor * global_std  # Final amplitude
    
    # Create linear drift: starts at 0, ends at A
    drift = np.linspace(0.0, A, segment_length, dtype=window.dtype)
    
    # Apply drift
    window_drifted[start:end] += drift
    
    return window_drifted, (start, end), A


def inject_closing_variance_change(
    window: np.ndarray,
    segment_length: int,
    variance_factor_range: tuple[float, float],
    random_state: int | None = None,
    transition_center: int = 200,
    spread: float = 20.0
) -> tuple[np.ndarray, tuple[int, int], float]:
    """
    Inject a variance-change anomaly (noise burst or damping) in the closing sequence.
    
    Anomalies are restricted to the closing sequence: indices [180, 360).
    Segment placement is biased towards the transition center.
    The segment's deviations from its mean are scaled by a factor.
    
    Args:
        window: 1D array of shape (360,), assumed standardized.
        segment_length: Length of variance-changed segment, must be <= 180.
        variance_factor_range: (min_factor, max_factor)
            - factor > 1: increased volatility (noise burst)
            - factor < 1: decreased volatility (damped/smooth)
        random_state: Random seed for reproducibility.
        transition_center: Target center for segment placement (default 200).
        spread: Standard deviation for position sampling (default 20).
    
    Returns:
        window_var_changed: Modified copy with variance change.
        (start, end): Indices of the modified segment.
        variance_factor: The factor used to scale variance.
    """
    rng = np.random.default_rng(random_state)
    window_var_changed = window.copy()
    
    # Ensure segment fits in closing range [180, 360)
    segment_length = min(segment_length, 180)
    closing_start = 180
    closing_end = 360
    
    # Sample start index biased towards transition center
    start = _sample_transition_centered_start(rng, closing_start, closing_end, 
                                               segment_length, transition_center, spread)
    end = start + segment_length
    
    # Extract segment
    segment = window[start:end]
    segment_mean = segment.mean()
    segment_std = segment.std()
    
    # Handle constant segment
    if segment_std == 0:
        return window_var_changed, (start, end), 1.0
    
    # Sample variance factor
    variance_factor = rng.uniform(variance_factor_range[0], variance_factor_range[1])
    
    # Transform: scale deviations from mean
    new_segment = segment_mean + variance_factor * (segment - segment_mean)
    
    # Apply
    window_var_changed[start:end] = new_segment
    
    return window_var_changed, (start, end), variance_factor


def inject_closing_time_warp(
    window: np.ndarray,
    segment_length: int,
    warp_factor_range: tuple[float, float],
    random_state: int | None = None,
    transition_center: int = 200,
    spread: float = 20.0
) -> tuple[np.ndarray, tuple[int, int], float]:
    """
    Inject a time-warped anomaly (speed-up or slow-down) in the closing sequence.
    
    Anomalies are restricted to the closing sequence: indices [180, 360).
    Segment placement is biased towards the transition center.
    The segment is stretched or compressed in time, then resampled back to 
    original length. Mimics "closing happens too fast/slow".
    
    Args:
        window: 1D array of shape (360,), assumed standardized.
        segment_length: Length of segment to warp, must be <= 180.
        warp_factor_range: (min_factor, max_factor)
            - warp_factor < 1: pattern compressed (faster closing)
            - warp_factor > 1: pattern stretched (slower closing)
        random_state: Random seed for reproducibility.
        transition_center: Target center for segment placement (default 200).
        spread: Standard deviation for position sampling (default 20).
    
    Returns:
        window_warped: Modified copy with time-warped segment.
        (start, end): Indices of the warped segment.
        warp_factor: The factor used for time warping.
    """
    rng = np.random.default_rng(random_state)
    window_warped = window.copy()
    
    # Ensure segment fits in closing range [180, 360)
    segment_length = min(segment_length, 180)
    closing_start = 180
    closing_end = 360
    
    # Sample start index biased towards transition center
    start = _sample_transition_centered_start(rng, closing_start, closing_end, 
                                               segment_length, transition_center, spread)
    end = start + segment_length
    
    # Extract segment
    segment = window[start:end]
    
    # Sample warp factor, ensuring it's not too close to 1.0
    min_warp, max_warp = warp_factor_range
    warp_factor = rng.uniform(min_warp, max_warp)
    
    # Re-sample if too close to 1.0 (try up to 10 times)
    for _ in range(10):
        if abs(warp_factor - 1.0) >= 0.05:
            break
        warp_factor = rng.uniform(min_warp, max_warp)
    
    # If still too close, force it away from 1.0
    if abs(warp_factor - 1.0) < 0.05:
        if warp_factor >= 1.0:
            warp_factor = 1.05
        else:
            warp_factor = 0.95
    
    # Time warp: create warped version
    original_idx = np.arange(segment_length)
    warped_len = max(2, int(round(segment_length * warp_factor)))
    warped_idx = np.linspace(0, segment_length - 1, num=warped_len)
    warped_segment = np.interp(warped_idx, original_idx, segment)
    
    # Resample back to original segment length
    resampled_idx = np.linspace(0, warped_len - 1, num=segment_length)
    warped_resampled = np.interp(resampled_idx, np.arange(warped_len), warped_segment)
    
    # Apply
    window_warped[start:end] = warped_resampled
    
    return window_warped, (start, end), warp_factor


@dataclass
class AnomalyDef():
    columns: list[str]
    f_args: dict
    random_seed: int
    title: str
    unit: str
    anomaly_f: callable
    N_anomalies: int = 3
    anomaly_duration_days: int = 10
    resolution_in_seconds: int = 2
    only_for_valve_open_state: bool = True

def add_anomalies(df, anomaly_f, anomaly_columns: list[bool], N_anomalies=3, anomaly_f_args={}, anomaly_duration_days=10, random_seed=3, resolution_in_seconds=2, only_for_valve_open_state=True):
    df = df.copy()
    N = df.shape[0]
    print(N)

    n_days_index = lambda d, n=10: pd.date_range(d, end = d+datetime.timedelta(days=n), freq=datetime.timedelta(seconds=resolution_in_seconds))

    # sample anomaly start times and find the machine on date closest to it
    np.random.seed(random_seed); anomaly_info = pd.DataFrame(np.random.exponential(scale=N/8, size=N_anomalies).astype(int), columns=["delta"])


    def shift_anomaly_date(date, df):
        """Shift the anomaly date to the closest machine_on_date"""
        df = df.loc[date:]
        return (df[~df["ball_valve_open"].isna()].iloc[0,:].name)

    for id_c, row in anomaly_info.iterrows():
        start_index = int(row.delta + (0 if id_c == 0 else anomaly_info.loc[id_c -1, "end_index"]))

        anomaly_info.loc[id_c, "sample_time"] = df.index[start_index]
        anomaly_info.loc[id_c, "start_index"] = start_index
        anomaly_info.loc[id_c, "start_time"] = shift_anomaly_date(anomaly_info.loc[id_c, "sample_time"], df)
        anomaly_duration = n_days_index(anomaly_info.loc[id_c, "start_time"] , anomaly_duration_days)
        anomaly_info.loc[id_c, "start_time"] = anomaly_duration[0]
        anomaly_info.loc[id_c, "end_time"] = anomaly_duration[-1]
        anomaly_info.loc[id_c, "end_index"] = start_index + len(anomaly_duration)

    # some analysis code
    # anomaly_info['start_index'].plot.hist(bins=100)
    # df.loc[n_days_index(anomaly_info.start_time[0] , anomaly_duration_days)][["machine_on"]].astype(int).plot()

    df["ground_truth"] = np.zeros(N)
    for id_d, d in enumerate(anomaly_info.start_time):
        idx = n_days_index(d, anomaly_duration_days)
        # df index may not be continuous
        idx = idx[idx.isin(df.index)]
        df_nd = df.loc[idx, anomaly_columns]

        if only_for_valve_open_state:
            machine_on = df.loc[idx, "ball_valve_open"]
            df_nd_on = df_nd.iloc[machine_on.values]
        else:
            df_nd_on = df_nd

        anomaly_info.loc[id_d, "data_start"] = str(df_nd_on.index[0])
        anomaly_info.loc[id_d, "data_end"] = str(df_nd_on.index[-1])
        anomaly_info.loc[id_d, "anomaly_length"] = str(df_nd_on.shape[0])

        anomalies = np.stack([anomaly_f(T=df_nd_on.shape[0], **anomaly_f_args) for _ in range(df_nd.shape[1])]).T
        df.loc[df_nd_on.index, anomaly_columns] += anomalies
        df.loc[df_nd_on.index, "ground_truth"] = np.ones(df_nd_on.shape[0])

    return anomaly_info, df

def create_anomaly(df, adef: AnomalyDef, dataset_root: str):
    info, df_an = add_anomalies(
        df,
        adef.anomaly_f,
        anomaly_f_args=adef.f_args,
        anomaly_columns=adef.columns,
        random_seed=adef.random_seed,
        resolution_in_seconds=adef.resolution_in_seconds,
        N_anomalies=adef.N_anomalies,
        anomaly_duration_days=adef.anomaly_duration_days,
        only_for_valve_open_state=adef.only_for_valve_open_state
    )

    fig, ax = plt.subplots(3,1,figsize=(20,10))
    df_an[adef.columns][::100].plot(ax=ax[0], title="anomaly")
    df_an["ground_truth"][::100].plot(ax=ax[1], title="ground truth")
    df[adef.columns][::100].plot(ax=ax[2], title="original")
    fig.suptitle(adef.title)
    df_an.to_parquet(dataset_root / "synthetic_anomalies" / f"Anomaly_{adef.title}_{adef.unit}.parquet")
    fig.tight_layout()
    fig.savefig(dataset_root / "synthetic_anomalies" / f"Anomaly_{adef.title}_{adef.unit}.png")
    return info


def main():
    # Define script parameters
    resampling_resolution_in_seconds = ...
    columns_to_apply_anomaly = ...

    # Load and homogenize data
    data = ...

    # Create anomaly
    create_anomaly(data,
                   AnomalyDef(anomaly_f=constant_anomaly,
                              columns=columns_to_apply_anomaly,
                              f_args={"max_y": 10.0, "sigma": 1},
                              random_seed=12,
                              resolution_in_seconds=resampling_resolution_in_seconds,
                              title=site_name,
                              unit=machine_name,
                              only_for_valve_open_state=False),
                   base_path)


if __name__ == '__main__':
    main()

