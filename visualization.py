import os
import glob
import math
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
from scipy.signal import welch, butter, filtfilt


sensor_mapping = {
    'sensor1': 'wrist',
    'sensor2': 'nose',
    'sensor3': 'finger',
    'sensor4': 'forearm',
    'sensor5': 'ear'
}

# —— 0.5-3Hz 带通滤波 —— #
def bandpass_filter(data, lowcut=0.5, highcut=3, fs=30, order=3):
    b, a = butter(order, [lowcut, highcut], fs=fs, btype='band')  # Using Butterworth filter to filt wave frequency between 0.5, 3 Hz (30 ~ 180 BPM).
    return filtfilt(b, a, data)

def get_hr(y, sr=30, min=30, max=180):
    p, q = welch(y, sr, nfft=1e5/sr, nperseg=np.min((len(y)-1, 256)))
    return p[(p>min/60)&(p<max/60)][np.argmax(q[(p>min/60)&(p<max/60)])]*60 # Using welch method to caculate PSD and find the peak of it.

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
            fs = 1/avg_dt
            print(f"  {name}: ~{fs:.1f} Hz")

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

    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n), sharex=True)
    axes = np.atleast_1d(axes)

    all_ts = np.concatenate([df['timestamp'].values for _, df in items])
    t0 = all_ts.min()

    for ax, (name, df) in zip(axes, items):
        x = df['timestamp'].values - t0
        y = df.iloc[:, 1].values
        prefix = name.split('-')[0]

        ax.plot(x, y, linewidth=1.5, alpha=0.8)
        ax.set_ylabel(prefix, fontsize=11)

        if len(x) > 1:
            avg_dt = np.mean(np.diff(x))
            fs = 1/avg_dt
            print(f"  {name}: ~{fs:.1f} Hz")

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
    


# —— 带滤波的五个传感器每个通道叠加总图 —— #
def plot_all_channels_overlay_filtered(dfs_filt, save=True):
    """
    绘制五个传感器的所有通道叠加图（带滤波），每个通道(red/ir/green)在一个子图中显示所有传感器的信号
    """
    sensor_mapping = {
        'sensor1': 'wrist',
        'sensor2': 'nose', 
        'sensor3': 'finger',
        'sensor4': 'forearm',
        'sensor5': 'ear'
    }
    
    # 获取传感器数据并进行深拷贝以避免修改原数据
    import copy
    sensor_dfs = OrderedDict()
    for name, df in dfs_filt.items():
        prefix = name.split('-')[0]
        if prefix in sensor_mapping and not df.empty:
            sensor_dfs[prefix] = copy.deepcopy(df.sort_values('timestamp'))
    
    if not sensor_dfs:
        print("No sensor data found for filtered overlay plot.")
        return
    
    # 对每个传感器数据进行滤波处理
    print("Applying bandpass filter to sensor data...")
    for prefix, df in sensor_dfs.items():
        if df.empty:
            continue
            
        ts = df['timestamp'].values
        # 只保留唯一的时间戳，去掉重复值
        tsu = np.unique(ts)
        if len(tsu) < 2:
            print(f"{prefix}: 没有足够不同时间戳，跳过滤波")
            continue
            
        # 计算相邻不同时间戳的中位数
        dt = np.median(np.diff(tsu))
        if dt <= 0:
            print(f"{prefix}: 时间戳异常 (dt={dt:.6f}), 跳过滤波")
            continue

        fs = 1.0 / dt
        print(f"{prefix}: 采样率 ≈ {fs:.1f} Hz, Nyquist={fs/2:.1f} Hz")

        # 检查采样率是否足够高以支持滤波
        nyquist = fs / 2
        if nyquist <= 1.5:  # Need nyquist > 1.5 Hz to safely filter at 0.5-3 Hz
            print(f"{prefix}: 采样率太低 (Nyquist={nyquist:.1f} Hz <= 1.5 Hz)，跳过滤波")
            continue
        
        # 对于低采样率信号，调整滤波参数
        if fs < 10:  # 如果采样率低于 10 Hz
            # 确保高频截止不超过 Nyquist 频率的 0.9 倍
            highcut = min(3.0, 0.9 * nyquist)
            print(f"{prefix}: 调整滤波范围为 0.5-{highcut:.1f} Hz")
        else:
            highcut = 3.0

        # 对各个通道进行滤波
        for col in df.columns:
            if col == 'timestamp':
                continue
            try:
                sensor_dfs[prefix][col] = bandpass_filter(
                    df[col].values,
                    lowcut=0.5,
                    highcut=highcut,
                    fs=fs)
                print(f"  {prefix} {col}: 滤波成功")
            except Exception as e:
                print(f"  {prefix} {col}: 滤波失败 - {str(e)}")
    
    # 计算全局时间起点
    all_ts = np.concatenate([df['timestamp'].values for df in sensor_dfs.values()])
    t0 = all_ts.min()
    
    channels = ['red', 'ir', 'green']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']  # 为5个传感器定义不同颜色
    
    # 创建3x1的子图布局
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    
    for ch_idx, channel in enumerate(channels):
        ax = axes[ch_idx]
        
        # 为每个通道绘制所有传感器的数据
        for sensor_idx, (prefix, df) in enumerate(sensor_dfs.items()):
            part_name = sensor_mapping[prefix]
            
            # 检查是否有该通道的数据
            col_idx = ch_idx + 1  # red=1, ir=2, green=3
            if df.shape[1] <= col_idx:
                continue
                
            # 获取时间和信号数据
            x = df['timestamp'].values - t0
            y = df.iloc[:, col_idx].values
            
            # 为了更好的可视化效果，可以对信号进行归一化
            if len(y) > 0:
                y_normalized = (y - np.min(y)) / (np.max(y) - np.min(y)) if np.max(y) != np.min(y) else y
                
                # 绘制信号线
                ax.plot(x, y_normalized, 
                       color=colors[sensor_idx % len(colors)], 
                       linewidth=1.5, 
                       alpha=0.8, 
                       label=f'{part_name}')
        
        # 设置子图属性
        ax.set_title(f'{channel.upper()} Channel - All Sensors Overlay (Filtered 0.5-3Hz)', 
                    fontsize=14, fontweight='bold')
        ax.set_ylabel('Normalized Signal', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        
        # 设置y轴范围
        ax.set_ylim(-0.1, 1.1)
    
    # 设置x轴标签
    axes[-1].set_xlabel('Time since window start (s)', fontsize=12)
    
    # 设置总标题
    plt.suptitle('Multi-Sensor Channel Overlay Visualization (Filtered)', 
                fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.85, 0.96])
    
    if save:
        plt.savefig('all_channels_overlay_filtered.png', dpi=300, bbox_inches='tight')
        print("Saved: all_channels_overlay_filtered.png")
    
    plt.show()


# —— PSD 频谱图 —— #
def plot_psd_analysis(dfs_filt, save=True):
    # —— Sensor 单通道 PSD —— #
    sensor_items = [
        (name, df)
        for name, df in dfs_filt.items()
        if name.split('-')[0] in sensor_mapping and not df.empty
    ]
    if sensor_items:
        sensor_dfs = OrderedDict()
        for name, df in sensor_items:
            prefix = name.split('-')[0]
            sensor_dfs[prefix] = df.sort_values('timestamp')

        n_sensors = len(sensor_dfs)
        channels = ['red', 'ir', 'green']
        fig, axes = plt.subplots(n_sensors, 3, figsize=(15, 4 * n_sensors))
        if n_sensors == 1:
            axes = axes.reshape(1, -1)

        for i, (prefix, df) in enumerate(sensor_dfs.items()):
            part = sensor_mapping[prefix]
            ts = df['timestamp'].values
            tsu = np.unique(ts)
            if len(tsu) < 2:
                for j in range(3):
                    axes[i, j].text(0.5, 0.5, '时间戳不足',
                                    ha='center', va='center')
                    axes[i, j].set_title(f"{part}-{channels[j]}")
                continue

            dt = np.median(np.diff(tsu))
            fs = 1.0 / dt

            for j, ch in enumerate(channels):
                ax = axes[i, j]
                col_idx = j + 1
                if df.shape[1] <= col_idx:
                    ax.text(0.5, 0.5, 'No data',
                            ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f"{part}-{ch}")
                    continue

                y = df.iloc[:, col_idx].values
                try:
                    p, q = welch(y, fs, nfft=int(1e5/fs), nperseg=min(len(y)-1, 256))
                    bpm = p * 60
                    mask = (bpm >= 30) & (bpm <= 180)

                    ax.plot(bpm[mask], q[mask], linewidth=1.5, color='C0')
                    ax.set_title(f"{part}-{ch}")
                    ax.grid(True, alpha=0.3)

                    if np.any(mask) and len(q[mask]) > 0:
                        peak_idx = np.argmax(q[mask])
                        peak_bpm = bpm[mask][peak_idx]
                        ax.axvline(peak_bpm, color='red', linestyle='--', alpha=0.5)
                        ax.text(0.98, 0.95, f'{peak_bpm:.1f} BPM',
                                transform=ax.transAxes,
                                ha='right', va='top',
                                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
                except Exception as e:
                    ax.text(0.5, 0.5, f"PSD 失败\n{str(e)[:30]}",
                            ha='center', va='center', transform=ax.transAxes)

                if i == n_sensors - 1:
                    ax.set_xlabel("Frequency (BPM)")
                if j == 0:
                    ax.set_ylabel(f"{part}\nPSD", rotation=0, labelpad=30)

        plt.suptitle("Power Spectral Density Analysis (Sensor signals)", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        if save:
            plt.savefig('psd_sensors.png', dpi=300, bbox_inches='tight')
            print("Saved: psd_sensors.png")
        plt.show()

    # —— Sensor 聚合网格 PSD —— #
    if sensor_items:
        prefixes = list(sensor_dfs.keys())
        n = len(prefixes)
        cols = 3
        rows = math.ceil(n / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
        axes = axes.flatten()

        for ax, prefix in zip(axes, prefixes):
            df = sensor_dfs[prefix]
            part = sensor_mapping[prefix]

            ts = df['timestamp'].values
            dt = np.median(np.diff(np.unique(ts)))
            fs = 1.0 / dt

            for ch in ['red', 'ir', 'green']:
                if ch in df.columns:
                    y = df[ch].values
                    f, Pxx = welch(y, fs, nperseg=min(len(y)-1, 256))
                    ax.plot(f, Pxx, label=ch)

            ax.set_title(part, fontsize=12)
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("PSD")
            ax.set_xlim(0, 5)
            ax.grid(alpha=0.3)
            ax.legend()

        # 隐藏多余
        for ax in axes[n:]:
            ax.axis('off')

        plt.suptitle("Aggregated Sensor PSD Grid", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        if save:
            plt.savefig('psd_sensor_grid.png', dpi=300, bbox_inches='tight')
            print("Saved: psd_sensor_grid.png")
        plt.show()


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

    start_ts = float(input("Enter start timestamp (Unix seconds): "))
    end_ts   = float(input("Enter end timestamp (Unix seconds): "))

    dfs = load_signals(args.input_dir)
    dfs_filt = filter_window(dfs, start_ts, end_ts)

    # 保存原始数据的深拷贝（滤波前）
    import copy
    dfs_original = copy.deepcopy(dfs_filt)

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

        # 检查采样率是否足够高以支持滤波
        nyquist = fs / 2
        if nyquist <= 1.5:  # Need nyquist > 1.5 Hz to safely filter at 0.5-3 Hz
            print(f"{name}: 采样率太低 (Nyquist={nyquist:.1f} Hz <= 1.5 Hz)，跳过滤波")
            continue
        
        # 对于低采样率信号，调整滤波参数
        if fs < 10:  # 如果采样率低于 10 Hz
            # 确保高频截止不超过 Nyquist 频率的 0.9 倍
            highcut = min(3.0, 0.9 * nyquist)
            print(f"{name}: 调整滤波范围为 0.5-{highcut:.1f} Hz")
        else:
            highcut = 3.0

        for col in df.columns:
            if col == 'timestamp':
                continue
            try:
                df[col] = bandpass_filter(
                    df[col].values,
                    lowcut=0.5,
                    highcut=highcut,
                    fs=fs)
            except Exception as e:
                print(f"{name} {col}: 滤波失败 - {str(e)}")
    

    # 先绘制无滤波的图
    print("\n====== WITHOUT FILTERING ======")
    
    # 无滤波的可穿戴总图
    print("\n-- Grid Channel View (No Filter) --")
    plot_channels_grid(dfs_original, save=True)
    # 重命名保存的文件
    if os.path.exists("channels_grid.png"):
        os.rename("channels_grid.png", "channels_grid_no_filter.png")
        print("Renamed to: channels_grid_no_filter.png")
    
    # 无滤波的真值+血氧总图
    '''
    if args.mode in ['combined', 'both']:
        print("\n-- Combined View (No Filter) --")
        plot_combined(dfs_original, save=True)
        if os.path.exists("combined_excluding_sensors.png"):
            os.rename("combined_excluding_sensors.png", "combined_excluding_sensors_no_filter.png")
            print("Renamed to: combined_excluding_sensors_no_filter.png")
    '''
    # 无滤波的真值+血氧拼接子图
    if args.mode in ['subplots', 'both']:
        print("\n-- Subplots View (No Filter) --")
        plot_subplots(dfs_original, save=True)
        if os.path.exists("subplots_excluding_sensors.png"):
            os.rename("subplots_excluding_sensors.png", "subplots_excluding_sensors_no_filter.png")
            print("Renamed to: subplots_excluding_sensors_no_filter.png")
    
    # 再绘制滤波后的图
    #print("\n\n====== WITH FILTERING ======")
    
    # 可穿戴总图
    #print("\n-- Grid Channel View (Filtered) --")
    #plot_channels_grid(dfs_filt)

    # 可穿戴子图 ###
    #print("\n-- Individual Channel Plots --")
    #plot_channels_separately(dfs_filt)

    # 真值+血氧总图
    #if args.mode in ['combined', 'both']:
    #    print("\n-- Combined View (Filtered) --")
    #    plot_combined(dfs_filt)

    # 真值+血氧拼接子图
    #if args.mode in ['subplots', 'both']:
    #    print("\n-- Subplots View (Filtered) --")
    #    plot_subplots(dfs_filt)
    
    print("\n-- All Channels Overlay View --")
    plot_all_channels_overlay_filtered(dfs_original, save=True)
    if os.path.exists("all_channels_overlay.png"):
        os.rename("all_channels_overlay.png", "all_channels_overlay_no_filter.png")
        print("Renamed to: all_channels_overlay_no_filter.png")

    # PSD 频谱分析
    #print("\n-- PSD Analysis --")
    #plot_psd_analysis(dfs_filt)

    print("\nDone!")

if __name__ == '__main__':
    main()
