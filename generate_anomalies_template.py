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

