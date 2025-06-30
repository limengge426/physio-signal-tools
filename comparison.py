import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
from collections import Counter
import warnings

warnings.filterwarnings('ignore')

# --- 全局配置 ---
sensor_mapping = {
    'sensor1': 'forearm',
    'sensor2': 'nose',
    'sensor3': 'finger',
    'sensor4': 'wrist',
    'sensor5': 'ear'
}
WINDOW_SIZE_SECONDS = 10
REQUIRED_VALID_WINDOWS = 100
OUTPUT_DIR_NAME = 'automated_analysis_ir_only'
OUTPUT_FILENAME = 'comparison_ir_scatter_15_fit.png' # 更新输出文件名
TARGET_CHANNEL = 'ir'

# --- 核心工具函数 (无修改) ---

def bandpass_filter(data, lowcut=0.5, highcut=3, fs=30, order=3):
    """0.5-3Hz 带通滤波"""
    nyquist = 0.5 * fs
    highcut = min(highcut, nyquist * 0.99)
    if lowcut >= highcut:
        return data
    b, a = butter(order, [lowcut, highcut], fs=fs, btype='band')
    return filtfilt(b, a, data)

def load_data_for_segment(segment_num, base_dir):
    """加载指定段的所有HUB和Biopac数据"""
    data = {'hub': {}, 'biopac': {}}
    hub_dir = os.path.join(base_dir, str(segment_num), 'HUB')
    if os.path.exists(hub_dir):
        for file in os.listdir(hub_dir):
            if file.startswith('sensor') and file.endswith('.csv'):
                name = file[:-4]
                try:
                    df = pd.read_csv(os.path.join(hub_dir, file))
                    data['hub'][name] = df.dropna().reset_index(drop=True)
                except Exception as e:
                    print(f"读取HUB文件失败 {file}: {e}")
    biopac_dir = os.path.join(base_dir, str(segment_num), 'Biopac')
    if os.path.exists(biopac_dir):
        for file in os.listdir(biopac_dir):
            if file.endswith('.csv'):
                name = file[:-4]
                try:
                    df = pd.read_csv(os.path.join(biopac_dir, file))
                    data['biopac'][name] = df.dropna().reset_index(drop=True)
                except Exception as e:
                    print(f"读取Biopac文件失败 {file}: {e}")
    return data

def get_peak_stats(data_series, fs):
    """对单个通道数据进行滤波、找波峰，并返回平均间隔和波峰数"""
    if data_series.empty or len(data_series) < fs * 2:
        return None, 0
    y_filtered = bandpass_filter(data_series.values, lowcut=0.5, highcut=3.0, fs=fs)
    min_distance = int(0.4 * fs)
    peaks, _ = find_peaks(y_filtered, height=np.percentile(y_filtered, 50), distance=min_distance)
    if len(peaks) < 2:
        return None, len(peaks)
    avg_interval_samples = np.mean(np.diff(peaks))
    avg_interval_time = avg_interval_samples / fs
    return avg_interval_time, len(peaks)

# --- 核心分析函数 (无修改) ---

def find_data_for_segment(segment_num, base_dir, target_channel):
    """
    自动化核心：为指定的目标通道切分窗口，验证多设备通道一致性并收集样本。
    """
    print(f"\n--- 正在处理 Segment {segment_num} ---")
    print(f"目标通道: {target_channel.upper()}")

    all_data = load_data_for_segment(segment_num, base_dir)
    hub_dfs = {name: df for name, df in all_data['hub'].items() if name.startswith('sensor')}
    biopac_df = all_data['biopac'].get('mean_bp', pd.DataFrame())

    if not hub_dfs:
        print(f"错误: 在 Segment {segment_num} 中未找到任何 sensor*.csv 文件。")
        return []

    collected_samples = []
    
    any_sensor_df = next(iter(hub_dfs.values())).sort_values('timestamp')
    min_ts, max_ts = any_sensor_df['timestamp'].min(), any_sensor_df['timestamp'].max()
    dt = np.median(np.diff(any_sensor_df['timestamp']))
    fs = 1.0 / dt
    print(f"检测到采样率 (fs) ≈ {fs:.2f} Hz。时间范围 [{min_ts:.2f}s, {max_ts:.2f}s]")
    
    current_start_time = min_ts

    while current_start_time < max_ts and len(collected_samples) < REQUIRED_VALID_WINDOWS:
        current_end_time = current_start_time + WINDOW_SIZE_SECONDS
        if current_end_time > max_ts:
            break
        
        col_idx = ['red', 'ir', 'green'].index(target_channel) + 1
        intervals_per_device = {}
        counts_per_device = {}

        for device_name, device_df in hub_dfs.items():
            window_df = device_df[(device_df['timestamp'] >= current_start_time) & (device_df['timestamp'] < current_end_time)]
            if window_df.shape[1] > col_idx:
                interval, count = get_peak_stats(window_df.iloc[:, col_idx], fs)
                if interval is not None:
                    intervals_per_device[device_name] = interval
                    counts_per_device[device_name] = count
        
        if counts_per_device:
            count_freq = Counter(counts_per_device.values())
            for peak_count, num_devices in count_freq.most_common():
                if peak_count > 1 and num_devices >= 2:
                    consistent_devices = [d for d, c in counts_per_device.items() if c == peak_count]
                    consistent_intervals = [intervals_per_device[d] for d in consistent_devices]
                    avg_interval = np.mean(consistent_intervals)
                    freq = 1.0 / avg_interval
                    
                    window_biopac_df = biopac_df[(biopac_df['timestamp'] >= current_start_time) & (biopac_df['timestamp'] < current_end_time)]
                    avg_bp = np.nan
                    if not window_biopac_df.empty and window_biopac_df.shape[1] > 1:
                        avg_bp = window_biopac_df.iloc[:, 1].mean()

                    if not np.isnan(avg_bp):
                        collected_samples.append({'frequency': freq, 'mean_bp': avg_bp})
                        print(f"  > 找到样本 {len(collected_samples)}/{REQUIRED_VALID_WINDOWS} (t=[{current_start_time:.1f}s]), Freq={freq:.2f} Hz, MeanBP={avg_bp:.2f} mmHg")
                    break
        
        current_start_time += WINDOW_SIZE_SECONDS
    
    if len(collected_samples) < REQUIRED_VALID_WINDOWS:
        print(f"警告: 数据已耗尽，仅为通道 {target_channel.upper()} 找到 {len(collected_samples)} 个样本。")
        
    return collected_samples


def plot_comparison(seg1_results, seg7_results, channel_name, save_dir):
    """在一张图中绘制两组数据点，并加入一条总的拟合线。"""
    if not seg1_results or not seg7_results:
        print("错误: 段1或段7缺少分析数据，无法生成图表。")
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 8))

    # 提取并绘制散点
    x1 = [res['frequency'] for res in seg1_results]
    y1 = [res['mean_bp'] for res in seg1_results]
    plt.plot(x1, y1, 'o', color='royalblue', label=f'Segment 1 ({len(x1)} windows)', markersize=8)

    x2 = [res['frequency'] for res in seg7_results]
    y2 = [res['mean_bp'] for res in seg7_results]
    plt.plot(x2, y2, 's', color='crimson', label=f'Segment 7 ({len(x2)} windows)', markersize=8)

    # --- 新增：计算并绘制拟合直线 ---
    # 1. 合并所有数据点
    all_x = np.array(x1 + x2)
    all_y = np.array(y1 + y2)

    # 2. 只有当点数超过1个时才进行拟合
    if len(all_x) > 1:
        # 3. 计算拟合线的斜率(m)和截距(b)
        m, b = np.polyfit(all_x, all_y, 1)

        # 4. 创建拟合线上的点用于绘图
        x_fit = np.array([min(all_x), max(all_x)])
        y_fit = m * x_fit + b

        # 5. 绘制拟合线
        plt.plot(x_fit, y_fit, color='green', linestyle=':', linewidth=2, 
                 label=f'Trendline (y={m:.2f}x + {b:.2f})')
    # --- 拟合线代码结束 ---

    plt.title(f'Frequency vs. Mean BP ({channel_name.upper()} Channel)', fontsize=16, fontweight='bold')
    plt.xlabel('Frequency (1/t) [Hz]', fontsize=12)
    plt.ylabel('Mean Blood Pressure (mmHg)', fontsize=12)
    plt.legend(fontsize=10)
    plt.tight_layout()

    save_path = os.path.join(save_dir, OUTPUT_FILENAME)
    plt.savefig(save_path, dpi=300)
    print(f"\n比较图已成功保存至: {save_path}")
    plt.show()

def run_analysis():
    """自动化分析流程的总控制器"""
    base_dir = '.'
    output_dir = os.path.join(base_dir, OUTPUT_DIR_NAME)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    seg1_plot_data = find_data_for_segment(1, base_dir, target_channel=TARGET_CHANNEL)
    
    if not seg1_plot_data:
        print("\n分析中止：未能从Segment 1收集到任何数据。")
        return

    seg7_plot_data = find_data_for_segment(7, base_dir, target_channel=TARGET_CHANNEL)

    plot_comparison(seg1_plot_data, seg7_plot_data, TARGET_CHANNEL, output_dir)
    
    print("\n--- 分析完成 ---")


if __name__ == '__main__':
    run_analysis()
