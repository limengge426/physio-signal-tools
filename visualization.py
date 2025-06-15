import os
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
from scipy.signal import butter, filtfilt


sensor_mapping = {
    'sensor1': 'wrist',
    'sensor2': 'nose',
    'sensor3': 'finger',
    'sensor4': 'forearm',
    'sensor5': 'ear'
}

# —— 0.5-3Hz 带通滤波 —— #
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='band')

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    if nyq <= lowcut:
        return data
    high = min(highcut, 0.9 * nyq)

    if high <= lowcut:
        return data

    b, a = butter(order, [lowcut/nyq, high/nyq], btype='band')
    padlen = 3 * (max(len(a), len(b)) - 1)
    if len(data) <= padlen:
        return data

    return filtfilt(b, a, data)

# —— 读取csv文件 —— #
def load_signals(input_dir):
    files = sorted(glob.glob(os.path.join(input_dir, '*.csv')))
    print(f"Processing '{input_dir}', found {len(files)} CSV files.")
    dfs = {}
    for fp in files:
        name = os.path.splitext(os.path.basename(fp))[0]
        dfs[name] = (
            pd.read_csv(fp)
              .dropna()
              .reset_index(drop=True)
        )
    return dfs


# —— 按时间窗口筛选 —— #
def filter_window(dfs, start_ts, end_ts):
    print(f"Filtering to window: {start_ts:.3f} - {end_ts:.3f}")
    dfs_filt = {}
    for name, df in dfs.items():
        df_w = df[(df['timestamp'] >= start_ts) & (df['timestamp'] <= end_ts)]
        dfs_filt[name] = df_w.reset_index(drop=True)
    return dfs_filt


# —— 真值 + 血氧总图 —— #
def plot_combined(dfs_filt, save=True):
    plt.figure(figsize=(14, 8))

    # 全局时间起点
    all_ts = np.concatenate([
        df['timestamp'].values
        for name, df in dfs_filt.items()
        if not df.empty and name.split('-')[0] not in sensor_mapping
    ])
    t0 = all_ts.min()

    for name, df in dfs_filt.items():
        prefix = name.split('-')[0]
        if prefix in sensor_mapping:
            continue
        if df.empty:
            continue

        x = df['timestamp'].values - t0
        y = df.iloc[:, 1].values

        plt.plot(x, y,
                 linewidth=1.5,
                 alpha=0.8,
                 label=prefix)

        # 打印采样率
        if len(x) > 1:
            avg_dt = np.mean(np.diff(x))
            print(f"  {name}: ~{1/avg_dt:.1f} Hz")

    plt.xlabel("Time since window start (s)")
    plt.ylabel("Signal value")
    plt.title("Aligned Signals", fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save:
        plt.savefig('combined_excluding_sensors.png', dpi=300, bbox_inches='tight')
        print("Saved: combined_excluding_sensors.png")
    plt.show()


# —— 真值+血氧拼接子图 —— #
def plot_subplots(dfs_filt, save=True):
    items = [
        (name, df)
        for name, df in dfs_filt.items()
        if name.split('-')[0] not in sensor_mapping and not df.empty
    ]
    n = len(items)
    if n == 0:
        print("No non-sensor1-5 data to plot.")
        return

    # 创建子图
    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n), sharex=True)
    axes = np.atleast_1d(axes)

    # 计算全局起始时间
    all_ts = np.concatenate([df['timestamp'].values for _, df in items])
    t0 = all_ts.min()

    for ax, (name, df) in zip(axes, items):
        x = df['timestamp'].values - t0
        y = df.iloc[:, 1].values
        prefix = name.split('-')[0]

        ax.plot(x, y, linewidth=1.5, alpha=0.8)
        ax.set_ylabel(prefix, fontsize=11)

        # 打印采样率
        if len(x) > 1:
            avg_dt = np.mean(np.diff(x))
            print(f"  {name}: ~{1/avg_dt:.1f} Hz")

    axes[-1].set_xlabel("Time since window start (s)")
    plt.suptitle("Aligned Signals (Subplots)", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save:
        plt.savefig('subplots_excluding_sensors.png', dpi=300, bbox_inches='tight')
        print("Saved: subplots_excluding_sensors.png")
    plt.show()


# —— 可穿戴总图 —— #
def plot_channels_grid(dfs_filt, save=True):
    sensor_mapping = {
        'sensor1': 'wrist',
        'sensor2': 'nose',
        'sensor3': 'finger',
        'sensor4': 'forearm',
        'sensor5': 'ear'
    }
    channels = ['red', 'ir', 'green']

    # 提取并按 sensor1–5 顺序组织数据
    sensor_dfs = OrderedDict()
    for name, df in dfs_filt.items():
        prefix = name.split('-')[0]
        if prefix in sensor_mapping:
            sensor_dfs[prefix] = df.sort_values('timestamp')

    if not sensor_dfs:
        print("No sensor1-5 data found.")
        return

    n_rows = len(sensor_dfs)
    n_cols = len(channels)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols*4, n_rows*3),
                             sharex=True)

    # 计算全局起始时间
    all_ts = np.concatenate([df['timestamp'].values for df in sensor_dfs.values()])
    t0 = all_ts.min()

    for i, (prefix, df) in enumerate(sensor_dfs.items()):
        x = df['timestamp'].values - t0
        part = sensor_mapping[prefix]

        for j, ch in enumerate(channels):
            ax = axes[i, j] if n_rows > 1 else axes[j]
            col_idx = j + 1
            if df.shape[1] <= col_idx:
                ax.text(0.5, 0.5, 'No data',
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"{part}-{ch}")
                continue

            y = df.iloc[:, col_idx].values
            ax.plot(x, y, linewidth=1.2)
            ax.set_title(f"{part}-{ch}")
            if i == n_rows - 1:
                ax.set_xlabel("Time since start (s)")
            if j == 0:
                ax.set_ylabel(part, rotation=0, labelpad=30)

    plt.suptitle("Sensor Channels Grid", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save:
        plt.savefig("channels_grid.png", dpi=300, bbox_inches="tight")
        print("Saved: channels_grid.png")

    plt.show()


# —— 可穿戴子图 —— #
def plot_channels_separately(dfs_filt, save=True):
    sensor_mapping = {
        'sensor1': 'wrist',
        'sensor2': 'nose',
        'sensor3': 'finger',
        'sensor4': 'forearm',
        'sensor5': 'ear'
    }

    for name, df in dfs_filt.items():
        prefix = name.split('-')[0]
        if prefix not in sensor_mapping:
            continue

        # 按时间排序并计算相对时间
        df_sorted = df.sort_values('timestamp')
        x = df_sorted['timestamp'].values - df_sorted['timestamp'].values.min()

        part = sensor_mapping[prefix]
        channels = ['red', 'ir', 'green']

        for i, ch in enumerate(channels):
            col_idx = i + 1
            if df_sorted.shape[1] <= col_idx:
                break

            y = df_sorted.iloc[:, col_idx].values

            plt.figure(figsize=(10, 5))
            plt.plot(x, y, linewidth=1.5)
            plt.title(f"{part} — {ch}", fontsize=14)
            plt.xlabel("Time since start (s)")
            plt.ylabel("Signal value")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            if save:
                fn = f"{name}_{ch}.png"
                plt.savefig(fn, dpi=300, bbox_inches='tight')
                print(f"Saved: {fn}")

            plt.show()
            plt.close()


# —— 主流程 —— #
def main():
    parser = argparse.ArgumentParser(
        description='Batch visualization of aligned multi-channel physiological signals')
    parser.add_argument(
        '-i', '--input_dir',
        default='output',
        help='Directory containing CSV files (default: output)')
    parser.add_argument(
        '-m', '--mode',
        choices=['combined', 'subplots', 'both'],
        default='both',
        help='Plot mode (default: both)')
    args = parser.parse_args()

    # 输入起止时间
    start_ts = float(input("Enter start timestamp (Unix seconds): "))
    end_ts   = float(input("Enter end timestamp (Unix seconds): "))

    dfs = load_signals(args.input_dir)
    dfs_filt = filter_window(dfs, start_ts, end_ts)

    for name, df in dfs_filt.items():
        if df.empty:
            continue

        ts = df['timestamp'].values
        if len(ts) < 2:
            print(f"{name}: 点不足，跳过滤波")
            continue

        ts = df['timestamp'].values
        # 只保留唯一的时间戳，去掉重复值，否则 diff 全为 0
        tsu = np.unique(ts)
        if len(tsu) < 2:
            print(f"{name}: 没有足够不同时间戳，跳过滤波")
            continue
        # 计算相邻不同时间戳的中位数
        dt = np.median(np.diff(tsu))
        if dt <= 0:
            print(f"{name}: 时间戳异常 (dt={dt:.6f}), 跳过滤波")
            continue

        fs = 1.0 / dt
        print(f"{name}: 单路采样率 ≈ {fs:.1f} Hz, Nyquist={fs/2:.1f} Hz")

        fs = 1.0 / dt
        print(f"{name}: 单路采样率 ≈ {fs:.1f} Hz, Nyquist={fs/2:.1f} Hz")

        for col in df.columns:
            if col == 'timestamp':
                continue
            df[col] = bandpass_filter(
                df[col].values,
                lowcut=0.5,
                highcut=3.0,
                fs=fs)
    

    # 可穿戴总图
    print("\n-- Grid Channel View --")
    plot_channels_grid(dfs_filt)

    # 可穿戴子图
    #print("\n-- Individual Channel Plots --")
    #plot_channels_separately(dfs_filt)

    # 真值+血氧总图
    if args.mode in ['combined', 'both']:
        print("\n-- Combined View --")
        plot_combined(dfs_filt)

    # 真值+血氧拼接子图
    if args.mode in ['subplots', 'both']:
        print("\n-- Subplots View --")
        plot_subplots(dfs_filt)

    print("\nDone!")

if __name__ == '__main__':
    main()