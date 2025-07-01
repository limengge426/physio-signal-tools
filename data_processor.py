import os
import re
import csv
import glob
import math
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timezone, timedelta
from collections import OrderedDict, Counter
from scipy.signal import welch, butter, filtfilt, find_peaks
import copy
import warnings

warnings.filterwarnings('ignore')


# ==================== 全局配置 ====================
sensor_mapping = {
    'sensor1': 'forearm',
    'sensor2': 'nose',
    'sensor3': 'finger',
    'sensor4': 'wrist',
    'sensor5': 'ear'
}


# ==================== BIOPAC同步模块 ====================
def parse_biopac_header(lines, utc_offset=8):
    """解析Biopac文件头信息"""
    header_info = {}
    column_names = []
    data_start_row = 0

    # Record the starting time
    for i, line in enumerate(lines):
        if '记录在:' in line or 'Recorded at:' in line:
            match = re.search(r'(?:记录在:|Recorded at:)\s*(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)', line)
            if match:
                time_str = match.group(1)
                local_dt = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S.%f')
                utc_dt = local_dt - timedelta(hours=utc_offset)
                utc_dt = utc_dt.replace(tzinfo=timezone.utc)
                header_info['record_timestamp'] = utc_dt.timestamp()
                print(f"Recording start time (local): {time_str}")
                print(f"Recording start time (UTC): {utc_dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
                print(f"UTC timestamp: {header_info['record_timestamp']}")

        if 'ms/采样点' in line or 'ms/sample' in line:
            match = re.search(r'([\d.]+)\s*(?:ms/采样点|ms/sample)', line)
            if match:
                header_info['sample_interval_ms'] = float(match.group(1))
                print(f"Sample interval: {header_info['sample_interval_ms']} ms")

        if re.search(r'sec.*CH\d+', line):
            data_start_row = i + 2
            cols = [c.strip() for c in line.strip().split(',')]
            header_info['column_indices'] = [j for j,c in enumerate(cols) if c.startswith('CH')]
            break

    # collect the channel name and unit
    channel_info = []
    skip_keywords = ['曲线图创建于', 'Chart created at', '记录在:', 'Recorded at:', 'ms/采样点', 'ms/sample', '通道', 'channels', 'sec,CH', 'samples']
    for i in range(data_start_row):
        line = lines[i].strip()
        if not line or any(k in line for k in skip_keywords):
            continue
        parts = [p.strip() for p in line.split(',') if p.strip()]
        for part in parts:
            if not part.replace('.', '').replace('-', '').isdigit():
                unit = None
                if i+1 < len(lines):
                    next_parts = [p.strip() for p in lines[i+1].split(',')]
                    for npart in next_parts:
                        if npart in ['mmHg','BPM','V','l/min','L/min','dyn*s/cm','L/min/m^2','%','bpm'] or '/' in npart:
                            unit = npart
                            break
                if unit:
                    clean_name = re.sub(r'\s*-\s*\w+', '', part)
                    channel_info.append({
                        'name': clean_name,
                        'unit': unit,
                        'safe_name': create_safe_filename(clean_name)
                    })
                    break

    if not channel_info:
        print("No channels found using standard parsing, trying alternative...")
        for i in range(data_start_row):
            line = lines[i].strip()
            if not line or any(k in line for k in skip_keywords):
                continue
            if not line.replace('.', '').replace('-', '').replace(',', '').isdigit() and len(line)<50:
                name = line.split()[0]
                unit = 'value'
                for j in range(i+1, min(i+3, len(lines))):
                    if lines[j].strip() in ['mmHg','BPM','V','l/min','L/min','dyn*s/cm','L/min/m^2','%','bpm']:
                        unit = lines[j].strip()
                        break
                channel_info.append({
                    'name': name,
                    'unit': unit,
                    'safe_name': create_safe_filename(name)
                })
    header_info['channels'] = channel_info
    header_info['data_start_row'] = data_start_row
    print(f"Found {len(channel_info)} channels:")
    for ch in channel_info:
        print(f"  - {ch['name']} ({ch['unit']}) -> {ch['safe_name']}.csv")
    return header_info


def create_safe_filename(name):
    """创建安全的文件名"""
    safe = re.sub(r'[^\w\s-]','', name)
    safe = re.sub(r'[-\s]+','_', safe)
    abbrev = {
        'RSP':'rsp','血压':'bp','心输出量':'cardiac_output','全身血管阻力':'systemic_vascular_resistance','心脏指数':'cardiac_index',
        'Systolic_BP':'systolic_bp','Diastolic_BP':'diastolic_bp','Mean_BP':'mean_bp','Heart_Rate':'hr'
    }
    for k,v in abbrev.items():
        if k in safe:
            return v
    return safe.lower()


def get_segment_timestamps(input_dir):
    """获取所有数字命名文件夹中的时间段信息"""
    segments = []
    
    # 查找所有数字命名的文件夹
    for folder in os.listdir(input_dir):
        folder_path = os.path.join(input_dir, folder)
        if os.path.isdir(folder_path) and folder.isdigit():
            # 检查Camera1/timestamps.csv是否存在
            timestamps_file = os.path.join(folder_path, 'Camera1', 'timestamps.csv')
            if os.path.exists(timestamps_file):
                try:
                    with open(timestamps_file, 'r', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        timestamps = []
                        for row in reader:
                            if 'timestamp' in row:
                                try:
                                    timestamps.append(float(row['timestamp']))
                                except (ValueError, KeyError):
                                    pass
                    
                    if len(timestamps) >= 2:
                        start_ts = timestamps[0]
                        end_ts = timestamps[-1]
                        segments.append((int(folder), start_ts, end_ts))
                        print(f"Segment {folder}: {start_ts:.3f} - {end_ts:.3f} ({end_ts-start_ts:.1f}s)")
                except Exception as e:
                    print(f"Error reading timestamps from {timestamps_file}: {e}")
    
    segments.sort(key=lambda x: x[0])
    return segments


def segment_csv_files(output_dir, input_dir, segments):
    """根据时间段切分已生成的标准CSV文件，并将其放到对应的段文件夹中"""
    if not segments:
        print("No segments found, skipping segmentation.")
        return
    
    print(f"\nSegmenting files based on {len(segments)} time periods...")
    
    csv_files = [f for f in os.listdir(output_dir) if f.endswith('.csv') and not '-' in f]
    
    for csv_file in csv_files:
        base_name = csv_file[:-4]
        csv_path = os.path.join(output_dir, csv_file)
        
        print(f"\nProcessing {csv_file}...")
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
        if not rows:
            print(f"  No data in {csv_file}, skipping.")
            continue
        
        fieldnames = reader.fieldnames
        
        for seg_num, start_ts, end_ts in segments:
            segment_rows = []
            for row in rows:
                try:
                    ts = float(row['timestamp'])
                    if start_ts <= ts <= end_ts:
                        segment_rows.append(row)
                except (ValueError, KeyError):
                    pass
            
            if segment_rows:
                segment_folder = os.path.join(input_dir, str(seg_num))
                
                if base_name.startswith('sensor'):
                    target_folder = os.path.join(segment_folder, 'HUB')
                    if not os.path.exists(target_folder):
                        os.makedirs(target_folder)
                    segment_path = os.path.join(target_folder, csv_file)
                elif base_name in ['bvp', 'spo2']:
                    target_folder = os.path.join(segment_folder, 'Oximeter')
                    if not os.path.exists(target_folder):
                        os.makedirs(target_folder)
                    segment_path = os.path.join(target_folder, csv_file)
                else:
                    target_folder = output_dir
                    segment_filename = f"{base_name}-{seg_num}.csv"
                    segment_path = os.path.join(target_folder, segment_filename)
                    
                    with open(segment_path, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(segment_rows)
                    
                    print(f"  Segment {seg_num}: {len(segment_rows)} rows -> {segment_filename}")
                    continue
                
                with open(segment_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(segment_rows)
                
                print(f"  Segment {seg_num}: {len(segment_rows)} rows -> {target_folder}/{csv_file}")
            else:
                print(f"  Segment {seg_num}: No data in time range")


def move_biopac_files_to_segments(output_dir, parent_dir):
    """将output文件夹中的分段Biopac文件移动到对应段的Biopac文件夹"""
    print("\n" + "="*50)
    print("移动Biopac分段文件到对应文件夹...")
    print("="*50)
    
    biopac_segment_files = []
    for file in os.listdir(output_dir):
        if file.endswith('.csv') and '-' in file:
            parts = file.split('-')
            if len(parts) == 2:
                name_part = parts[0]
                number_part = parts[1].replace('.csv', '')
                if number_part.isdigit():
                    segment_num = int(number_part)
                    if 1 <= segment_num <= 11:
                        if not name_part.startswith('sensor') and name_part not in ['bvp', 'spo2']:
                            biopac_segment_files.append((file, segment_num, name_part))
    
    if not biopac_segment_files:
        print("没有找到需要移动的Biopac分段文件")
        return
    
    print(f"找到 {len(biopac_segment_files)} 个Biopac分段文件需要移动")
    
    moved_count = 0
    for file, segment_num, name_part in biopac_segment_files:
        source_path = os.path.join(output_dir, file)
        segment_dir = os.path.join(parent_dir, str(segment_num))
        
        if not os.path.exists(segment_dir):
            os.makedirs(segment_dir)
            print(f"  创建段文件夹: {segment_dir}")
        
        biopac_dir = os.path.join(segment_dir, 'Biopac')
        if not os.path.exists(biopac_dir):
            os.makedirs(biopac_dir)
            print(f"  创建Biopac文件夹: {biopac_dir}")
        
        target_filename = f"{name_part}.csv"
        target_path = os.path.join(biopac_dir, target_filename)
        
        try:
            os.rename(source_path, target_path)
            print(f"  移动: {file} -> {segment_num}/Biopac/{target_filename}")
            moved_count += 1
        except Exception as e:
            print(f"  移动失败 {file}: {str(e)}")
    
    print(f"\n成功移动了 {moved_count} 个文件")


def convert_biopac_to_csv(input_file, output_dir='output', utc_offset=8, auto_segment=True):
    """转换Biopac数据到CSV格式"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    input_dir = os.path.dirname(input_file)
    if not input_dir:
        input_dir = '.'
    
    with open(input_file,'r',encoding='utf-8-sig') as f:
        lines = f.readlines()
    header = parse_biopac_header(lines, utc_offset)
    base_ts = header.get('record_timestamp', datetime.now().timestamp())
    indices = header['column_indices']
    start = header['data_start_row']
    files, writers = {}, {}
    for idx, ch in enumerate(header['channels']):
        if idx < len(indices):
            fn = os.path.join(output_dir, f"{ch['safe_name']}.csv")
            files[idx] = open(fn,'w',newline='',encoding='utf-8')
            w = csv.writer(files[idx]); w.writerow(['timestamp', ch['safe_name']])
            writers[idx] = w
    interval = header.get('sample_interval_ms',0.5)/1000.0
    count=0
    for line in lines[start:]:
        vals = line.strip().split(',')
        if len(vals)<2: continue
        try: rt=float(vals[0])
        except: continue
        ts=base_ts+rt
        for i,ch in enumerate(header['channels']):
            if i<len(indices):
                ci=indices[i]
                if ci<len(vals):
                    try: dv=float(vals[ci]); writers[i].writerow([f"{ts:.7f}",dv])
                    except: pass
        count+=1
    for f in files.values(): f.close()
    print(f"Conversion complete! Processed {count} rows")
    print(f"Output files saved in: {output_dir}/")
    
    if auto_segment:
        segments = get_segment_timestamps(input_dir)
        if segments:
            segment_csv_files(output_dir, input_dir, segments)
            move_biopac_files_to_segments(output_dir, input_dir)
        else:
            print("\nNo numbered folders with timestamps found for segmentation.")


# ==================== 对比分析模块 (Fusion from comparison.py) ====================
# --- 对比分析模块配置 ---
COMP_WINDOW_SIZE_SECONDS = 10
COMP_REQUIRED_VALID_WINDOWS = 100
COMP_TARGET_CHANNEL = 'ir'
COMP_OUTPUT_DIR_NAME = 'automated_analysis_ir_only'
COMP_OUTPUT_FILENAME = 'comparison.jpg'

# --- 对比分析模块工具函数 ---
def comp_bandpass_filter(data, lowcut=0.5, highcut=3, fs=30, order=3):
    """0.5-3Hz 带通滤波 (对比分析专用)"""
    nyquist = 0.5 * fs
    highcut = min(highcut, nyquist * 0.99)
    if lowcut >= highcut:
        return data
    b, a = butter(order, [lowcut, highcut], fs=fs, btype='band')
    return filtfilt(b, a, data)

def load_comparison_data(segment_num, base_dir):
    """加载对比分析所需的HUB和Biopac数据"""
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
            # 仅加载mean_bp
            if file == 'mean_bp.csv':
                name = file[:-4]
                try:
                    df = pd.read_csv(os.path.join(biopac_dir, file))
                    data['biopac'][name] = df.dropna().reset_index(drop=True)
                except Exception as e:
                    print(f"读取Biopac文件失败 {file}: {e}")
    return data

def get_comp_peak_stats(data_series, fs):
    """对单个通道数据进行滤波、找波峰，并返回平均间隔和波峰数 (对比分析专用)"""
    if data_series.empty or len(data_series) < fs * 2:
        return None, 0
    y_filtered = comp_bandpass_filter(data_series.values, lowcut=0.5, highcut=3.0, fs=fs)
    min_distance = int(0.4 * fs)
    peaks, _ = find_peaks(y_filtered, height=np.percentile(y_filtered, 50), distance=min_distance)
    if len(peaks) < 2:
        return None, len(peaks)
    avg_interval_samples = np.mean(np.diff(peaks))
    avg_interval_time = avg_interval_samples / fs
    return avg_interval_time, len(peaks)

def find_data_for_comparison(segment_num, base_dir, target_channel):
    """为对比分析寻找有效数据窗口"""
    print(f"\n--- 正在为对比分析处理 Segment {segment_num} ---")
    print(f"目标通道: {target_channel.upper()}")

    all_data = load_comparison_data(segment_num, base_dir)
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

    while current_start_time < max_ts and len(collected_samples) < COMP_REQUIRED_VALID_WINDOWS:
        current_end_time = current_start_time + COMP_WINDOW_SIZE_SECONDS
        if current_end_time > max_ts:
            break
        
        col_idx = ['red', 'ir', 'green'].index(target_channel) + 1
        intervals_per_device = {}
        counts_per_device = {}

        for device_name, device_df in hub_dfs.items():
            window_df = device_df[(device_df['timestamp'] >= current_start_time) & (device_df['timestamp'] < current_end_time)]
            if window_df.shape[1] > col_idx:
                interval, count = get_comp_peak_stats(window_df.iloc[:, col_idx], fs)
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
                        print(f"  > 找到对比样本 {len(collected_samples)}/{COMP_REQUIRED_VALID_WINDOWS} (t=[{current_start_time:.1f}s]), Freq={freq:.2f} Hz, MeanBP={avg_bp:.2f} mmHg")
                    break
        
        current_start_time += COMP_WINDOW_SIZE_SECONDS
    
    if len(collected_samples) < COMP_REQUIRED_VALID_WINDOWS:
        print(f"警告: 数据已耗尽，仅为通道 {target_channel.upper()} 找到 {len(collected_samples)} 个对比样本。")
        
    return collected_samples

def plot_comparison(seg1_results, seg7_results, channel_name, save_dir):
    """绘制对比分析图"""
    if not seg1_results or not seg7_results:
        print("错误: 对比分析缺少段1或段7的数据，无法生成图表。")
        return

    fig = plt.figure(figsize=(12, 8))
    plt.style.use('seaborn-v0_8-whitegrid')

    x1 = [res['frequency'] for res in seg1_results]
    y1 = [res['mean_bp'] for res in seg1_results]
    plt.plot(x1, y1, 'o', color='royalblue', label=f'Segment 1 ({len(x1)} windows)', markersize=8)

    x2 = [res['frequency'] for res in seg7_results]
    y2 = [res['mean_bp'] for res in seg7_results]
    plt.plot(x2, y2, 's', color='crimson', label=f'Segment 7 ({len(x2)} windows)', markersize=8)

    all_x = np.array(x1 + x2)
    all_y = np.array(y1 + y2)

    if len(all_x) > 1:
        m, b = np.polyfit(all_x, all_y, 1)
        x_fit = np.array([min(all_x), max(all_x)])
        y_fit = m * x_fit + b
        plt.plot(x_fit, y_fit, color='green', linestyle=':', linewidth=2, 
                 label=f'Trendline (y={m:.2f}x + {b:.2f})')

    plt.title(f'Frequency vs. Mean BP ({channel_name.upper()} Channel)', fontsize=16, fontweight='bold')
    plt.xlabel('Frequency (1/t) [Hz]', fontsize=12)
    plt.ylabel('Mean Blood Pressure (mmHg)', fontsize=12)
    plt.legend(fontsize=10)
    plt.tight_layout()

    save_path = os.path.join(save_dir, COMP_OUTPUT_FILENAME)
    plt.savefig(save_path, dpi=300)
    print(f"\n对比分析图已成功保存至: {save_path}")
    plt.close(fig) # 自动关闭

def run_comparison_analysis(base_dir, save_dir):
    """运行对比分析的总控制器"""
    print("\n" + "="*50)
    print("步骤 4: 运行对比分析 (Seg1 vs Seg7)")
    print("="*50)
    
    seg1_plot_data = find_data_for_comparison(1, base_dir, target_channel=COMP_TARGET_CHANNEL)
    
    if not seg1_plot_data:
        print("\n对比分析中止：未能从Segment 1收集到任何数据。")
        return

    seg7_plot_data = find_data_for_comparison(7, base_dir, target_channel=COMP_TARGET_CHANNEL)

    plot_comparison(seg1_plot_data, seg7_plot_data, COMP_TARGET_CHANNEL, save_dir)
    print("\n--- 对比分析完成 ---")


# ==================== 可视化模块 ====================
def bandpass_filter(data, lowcut=0.5, highcut=3, fs=30, order=3):
    """0.5-3Hz 带通滤波 (主可视化模块使用)"""
    b, a = butter(order, [lowcut, highcut], fs=fs, btype='band')
    return filtfilt(b, a, data)


def get_hr(y, sr=30, min=30, max=180):
    """计算心率"""
    p, q = welch(y, sr, nfft=1e5/sr, nperseg=np.min((len(y)-1, 256)))
    return p[(p>min/60)&(p<max/60)][np.argmax(q[(p>min/60)&(p<max/60)])]*60


def load_signals(input_dir):
    """读取csv文件"""
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


def filter_window(dfs, start_ts, end_ts):
    """按时间窗口筛选"""
    print(f"Filtering to window: {start_ts:.3f} - {end_ts:.3f}")
    dfs_filt = {}
    for name, df in dfs.items():
        df_w = df[(df['timestamp'] >= start_ts) & (df['timestamp'] <= end_ts)]
        dfs_filt[name] = df_w.reset_index(drop=True)
    return dfs_filt


def plot_biopac_distribution_analysis(dfs_filt, save=True):
    """分析真值数据分布的拼接柱状图 - 按指定顺序排列"""
    biopac_data = {
        name: df for name, df in dfs_filt.items()
        if name.split('-')[0] not in sensor_mapping and not df.empty
    }
    
    if not biopac_data:
        print("No Biopac data found for distribution analysis.")
        return
    
    display_order = [
        'bp', 'systolic_bp', 'diastolic_bp',
        'mean_bp', 'hr', 'rsp',
        'cardiac_index', 'cardiac_output', 'systemic_vascular_resistance'
    ]
    
    ordered_data = []
    for key in display_order:
        if key in biopac_data:
            ordered_data.append((key, biopac_data[key]))
    
    for key, df in biopac_data.items():
        if key not in display_order:
            ordered_data.append((key, df))
    
    n_datasets = len(ordered_data)
    if n_datasets == 0:
        return
    
    cols = 3
    rows = math.ceil(n_datasets / cols)
    figsize = (18, 5 * rows)
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    if rows == 1:
        axes = axes.reshape(1, -1) if n_datasets > 1 else [axes]
    axes = axes.flatten()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#17becf', '#bcbd22']
    
    for idx, (name, df) in enumerate(ordered_data):
        ax = axes[idx]
        
        if df.shape[1] > 1:
            data_column = df.iloc[:, 1].values
            
            Q1 = np.percentile(data_column, 25)
            Q3 = np.percentile(data_column, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            filtered_data = data_column[(data_column >= lower_bound) & (data_column <= upper_bound)]
            
            n_bins = min(30, max(10, int(np.sqrt(len(filtered_data)))))
            counts, bins, patches = ax.hist(filtered_data, bins=n_bins, 
                                          color=colors[idx % len(colors)], 
                                          alpha=0.7, edgecolor='black', linewidth=0.5)
            
            clean_name = name.replace('_', ' ').title()
            ax.set_title(f'Distribution of {clean_name}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Value', fontsize=10)
            ax.set_ylabel('Number', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            mean_val = np.mean(filtered_data)
            std_val = np.std(filtered_data)
            median_val = np.median(filtered_data)
            
            ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8, linewidth=2, label=f'Mean: {mean_val:.2f}')
            ax.axvline(median_val, color='green', linestyle='--', alpha=0.8, linewidth=2, label=f'Median: {median_val:.2f}')
            
            stats_text = f'N: {len(filtered_data)}\nMean: {mean_val:.2f}\nStd: {std_val:.2f}\nMedian: {median_val:.2f}'
            ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, 
                   fontsize=9, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.legend(loc='upper left', fontsize=8)
            
            print(f"  {name}: {len(filtered_data)} data points, Mean={mean_val:.2f}, Std={std_val:.2f}")
    
    for idx in range(n_datasets, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Biopac Data Distribution Analysis', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save:
        plt.savefig('biopac_distribution_analysis.png', dpi=300, bbox_inches='tight')
        print("Saved: biopac_distribution_analysis.png")
    
    plt.close(fig)


def plot_oximeter_distribution_analysis(dfs_filt, save=True):
    """分析Oximeter数据分布的拼接柱状图"""
    oximeter_data = {
        name: df for name, df in dfs_filt.items()
        if name.split('-')[0] not in sensor_mapping and not df.empty
    }
    
    if not oximeter_data:
        print("No Oximeter data found for distribution analysis.")
        return
    
    n_datasets = len(oximeter_data)
    if n_datasets == 0:
        return
    
    if n_datasets <= 3:
        rows, cols = 1, n_datasets
        figsize = (6*cols, 5)
    elif n_datasets <= 6:
        rows, cols = 2, 3
        figsize = (18, 10)
    else:
        rows, cols = 3, 3
        figsize = (18, 15)
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    if n_datasets == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57', '#ff9ff3', '#54a0ff']
    
    for idx, (name, df) in enumerate(oximeter_data.items()):
        ax = axes[idx]
        
        if df.shape[1] > 1:
            data_column = df.iloc[:, 1].values
            
            Q1 = np.percentile(data_column, 25)
            Q3 = np.percentile(data_column, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            filtered_data = data_column[(data_column >= lower_bound) & (data_column <= upper_bound)]
            
            n_bins = min(30, max(10, int(np.sqrt(len(filtered_data)))))
            counts, bins, patches = ax.hist(filtered_data, bins=n_bins, 
                                          color=colors[idx % len(colors)], 
                                          alpha=0.7, edgecolor='black', linewidth=0.5)
            
            clean_name = name.replace('_', ' ').title()
            ax.set_title(f'Distribution of {clean_name}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Value', fontsize=10)
            ax.set_ylabel('Number', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            mean_val = np.mean(filtered_data)
            std_val = np.std(filtered_data)
            median_val = np.median(filtered_data)
            
            ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8, linewidth=2, label=f'Mean: {mean_val:.2f}')
            ax.axvline(median_val, color='green', linestyle='--', alpha=0.8, linewidth=2, label=f'Median: {median_val:.2f}')
            
            stats_text = f'N: {len(filtered_data)}\nMean: {mean_val:.2f}\nStd: {std_val:.2f}\nMedian: {median_val:.2f}'
            ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, 
                   fontsize=9, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.legend(loc='upper left', fontsize=8)
            
            print(f"  {name}: {len(filtered_data)} data points, Mean={mean_val:.2f}, Std={std_val:.2f}")
    
    for idx in range(n_datasets, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Oximeter Data Distribution Analysis', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save:
        plt.savefig('oximeter_distribution_analysis.png', dpi=300, bbox_inches='tight')
        print("Saved: oximeter_distribution_analysis.png")
    
    plt.close(fig)


def plot_combined(dfs_filt, save=True):
    """真值 + 血氧总图"""
    fig = plt.figure(figsize=(14, 8))

    all_ts = np.concatenate([
        df['timestamp'].values
        for name, df in dfs_filt.items()
        if not df.empty and name.split('-')[0] not in sensor_mapping
    ])
    
    if len(all_ts) == 0:
        print("No non-sensor data found for combined plot.")
        plt.close(fig)
        return
        
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
    
    plt.close(fig)


def plot_subplots(dfs_filt, save=True):
    """真值+血氧拼接子图"""
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
    
    plt.close(fig)


def plot_channels_grid(dfs_filt, save=True):
    """可穿戴总图"""
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

    if n_rows == 1:
        axes = axes.reshape(1, -1)

    all_ts = np.concatenate([df['timestamp'].values for df in sensor_dfs.values()])
    t0 = all_ts.min()

    for i, (prefix, df) in enumerate(sensor_dfs.items()):
        x = df['timestamp'].values - t0
        part = sensor_mapping[prefix]

        for j, ch in enumerate(channels):
            ax = axes[i, j]
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

    plt.close(fig)


def plot_channels_separately(dfs_filt, save=True):
    """可穿戴子图"""
    for name, df in dfs_filt.items():
        prefix = name.split('-')[0]
        if prefix not in sensor_mapping:
            continue

        df_sorted = df.sort_values('timestamp')
        x = df_sorted['timestamp'].values - df_sorted['timestamp'].values.min()

        part = sensor_mapping[prefix]
        channels = ['red', 'ir', 'green']

        for i, ch in enumerate(channels):
            col_idx = i + 1
            if df_sorted.shape[1] <= col_idx:
                break

            y = df_sorted.iloc[:, col_idx].values

            fig = plt.figure(figsize=(10, 5))
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

            plt.close(fig)


def plot_all_channels_overlay_filtered(dfs_filt, save=True):
    """带滤波的五个传感器每个通道叠加总图"""
    sensor_dfs = OrderedDict()
    for name, df in dfs_filt.items():
        prefix = name.split('-')[0]
        if prefix in sensor_mapping and not df.empty:
            sensor_dfs[prefix] = copy.deepcopy(df.sort_values('timestamp'))
    
    if not sensor_dfs:
        print("No sensor data found for filtered overlay plot.")
        return
    
    print("Applying bandpass filter to sensor data...")
    for prefix, df in sensor_dfs.items():
        if df.empty:
            continue
            
        ts = df['timestamp'].values
        tsu = np.unique(ts)
        if len(tsu) < 2:
            print(f"{prefix}: 没有足够不同时间戳，跳过滤波")
            continue
            
        dt = np.median(np.diff(tsu))
        if dt <= 0:
            print(f"{prefix}: 时间戳异常 (dt={dt:.6f}), 跳过滤波")
            continue

        fs = 1.0 / dt
        print(f"{prefix}: 采样率 ≈ {fs:.1f} Hz, Nyquist={fs/2:.1f} Hz")

        nyquist = fs / 2
        if nyquist <= 1.5:
            print(f"{prefix}: 采样率太低 (Nyquist={nyquist:.1f} Hz <= 1.5 Hz)，跳过滤波")
            continue
        
        if fs < 10:
            highcut = min(3.0, 0.9 * nyquist)
            print(f"{prefix}: 调整滤波范围为 0.5-{highcut:.1f} Hz")
        else:
            highcut = 3.0

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
    
    all_ts = np.concatenate([df['timestamp'].values for df in sensor_dfs.values()])
    t0 = all_ts.min()
    
    channels = ['red', 'ir', 'green']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FECA57']
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    
    for ch_idx, channel in enumerate(channels):
        ax = axes[ch_idx]
        
        for sensor_idx, (prefix, df) in enumerate(sensor_dfs.items()):
            part_name = sensor_mapping[prefix]
            
            col_idx = ch_idx + 1
            if df.shape[1] <= col_idx:
                continue
                
            x = df['timestamp'].values - t0
            y = df.iloc[:, col_idx].values
            
            if len(y) > 0:
                y_normalized = (y - np.min(y)) / (np.max(y) - np.min(y)) if np.max(y) != np.min(y) else y
                
                ax.plot(x, y_normalized, 
                       color=colors[sensor_idx % len(colors)], 
                       linewidth=1.5, 
                       alpha=0.8, 
                       label=f'{part_name}')
        
        ax.set_title(f'{channel.upper()} Channel - All Sensors Overlay (Filtered 0.5-3Hz)', 
                    fontsize=14, fontweight='bold')
        ax.set_ylabel('Normalized Signal', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        
        ax.set_ylim(-0.1, 1.1)
    
    axes[-1].set_xlabel('Time since window start (s)', fontsize=12)
    
    plt.suptitle('Multi-Sensor Channel Overlay Visualization (Filtered)', 
                fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.85, 0.96])
    
    if save:
        plt.savefig('all_channels_overlay_filtered.png', dpi=300, bbox_inches='tight')
        print("Saved: all_channels_overlay_filtered.png")
    
    plt.close(fig)


def plot_psd_analysis(dfs_filt, save=True):
    """PSD 频谱图"""
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
        plt.close(fig)

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

        for ax in axes[n:]:
            ax.axis('off')

        plt.suptitle("Aggregated Sensor PSD Grid", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        if save:
            plt.savefig('psd_sensor_grid.png', dpi=300, bbox_inches='tight')
            print("Saved: psd_sensor_grid.png")
        plt.close(fig)


def run_visualization(base_dir='output'):
    """运行可视化流程 - 全自动修改版本"""
    print("\n" + "="*50)
    print("开始可视化处理...")
    print("="*50)
    
    parent_dir = os.path.dirname(os.path.abspath(base_dir))
    
    available_segments = []
    for segment in range(1, 12):
        segment_dir = os.path.join(parent_dir, str(segment))
        if os.path.exists(segment_dir):
            hub_path = os.path.join(segment_dir, 'HUB')
            oximeter_path = os.path.join(segment_dir, 'Oximeter')
            if os.path.exists(hub_path) or os.path.exists(oximeter_path):
                available_segments.append(segment)
    
    if not available_segments:
        print("没有找到任何包含HUB或Oximeter数据的段文件夹")
        return
        
    print(f"找到 {len(available_segments)} 个数据段: {available_segments}")
    
    vis_dir = os.path.join(parent_dir, 'visualization')
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
        print(f"Created visualization directory: {vis_dir}")
    
    original_cwd = os.getcwd()
    
    def get_biopac_files_for_segment(segment_dir):
        """获取指定段的Biopac文件"""
        biopac_path = os.path.join(segment_dir, 'Biopac')
        if os.path.exists(biopac_path):
            return [f for f in os.listdir(biopac_path) if f.endswith('.csv')]
        return []
    
    segments_processed = 0
    for segment in available_segments:
        segment_dir = os.path.join(parent_dir, str(segment))
        
        print(f"\n" + "="*50)
        print(f"处理段 {segment} 数据...")
        print("="*50)
        
        # --- 自动化时间戳获取 ---
        start_ts, end_ts = None, None
        sensor2_path = os.path.join(segment_dir, 'HUB', 'sensor2.csv')
        if os.path.exists(sensor2_path):
            try:
                df_sensor2 = pd.read_csv(sensor2_path)
                if not df_sensor2.empty and 'timestamp' in df_sensor2.columns:
                    first_ts = df_sensor2['timestamp'].iloc[0]
                    start_ts = first_ts + 60
                    end_ts = start_ts + 10
                    print(f"段 {segment} 自动获取时间窗口: {start_ts:.3f}s - {end_ts:.3f}s (基于sensor2.csv)")
                else:
                    print(f"警告: {sensor2_path} 为空或无时间戳列。")
            except Exception as e:
                print(f"读取 {sensor2_path} 失败: {e}")
        else:
            print(f"警告: 未找到 {sensor2_path}，无法自动获取段 {segment} 的时间戳。")

        if start_ts is None:
            print(f"跳过段 {segment} 的可视化，因为无法确定时间窗口。")
            continue
        # --- 时间戳自动化结束 ---

        segment_vis_dir = os.path.join(vis_dir, str(segment))
        if not os.path.exists(segment_vis_dir):
            os.makedirs(segment_vis_dir)
        
        # 处理Biopac数据
        biopac_files = get_biopac_files_for_segment(segment_dir)
        if biopac_files:
            print(f"\n处理段 {segment} - Biopac数据...")
            biopac_dfs = {}
            biopac_source_path = os.path.join(segment_dir, 'Biopac')
            for file in biopac_files:
                name = file[:-4]
                try:
                    df = pd.read_csv(os.path.join(biopac_source_path, file))
                    biopac_dfs[name] = df.dropna().reset_index(drop=True)
                except Exception as e:
                    print(f"Error reading {file}: {e}")
            
            if biopac_dfs:
                biopac_filt = filter_window(biopac_dfs, start_ts, end_ts)
                biopac_original = copy.deepcopy(biopac_filt)
                has_data = any(not df.empty for df in biopac_filt.values())
                if has_data:
                    apply_filtering(biopac_filt)
                    biopac_dir = os.path.join(segment_vis_dir, 'Biopac')
                    if not os.path.exists(biopac_dir): os.makedirs(biopac_dir)
                    os.chdir(biopac_dir)
                    print(f"  生成段 {segment} Biopac图表...")
                    plot_biopac_distribution_analysis(biopac_original, save=True)
                    if os.path.exists("biopac_distribution_analysis.png"): os.rename("biopac_distribution_analysis.png", f"biopac_distribution_analysis_seg{segment}.png")
                    plot_subplots(biopac_original, save=True)
                    if os.path.exists("subplots_excluding_sensors.png"): os.rename("subplots_excluding_sensors.png", f"biopac_subplots_no_filter_seg{segment}.png")
                    plot_combined(biopac_filt, save=True)
                    if os.path.exists("combined_excluding_sensors.png"): os.rename("combined_excluding_sensors.png", f"biopac_combined_filtered_seg{segment}.png")
                    plot_subplots(biopac_filt, save=True)
                    if os.path.exists("subplots_excluding_sensors.png"): os.rename("subplots_excluding_sensors.png", f"biopac_subplots_filtered_seg{segment}.png")
                    os.chdir(original_cwd)
                else: print(f"  段 {segment} Biopac数据在指定时间窗口内无数据")
        
        # 处理Oximeter数据
        oximeter_source_path = os.path.join(segment_dir, 'Oximeter')
        if os.path.exists(oximeter_source_path):
            print(f"\n处理段 {segment} - Oximeter数据...")
            oximeter_files = [f for f in os.listdir(oximeter_source_path) if f.endswith('.csv')]
            if oximeter_files:
                oximeter_dfs = {}
                for file in oximeter_files:
                    name = file[:-4]
                    try:
                        df = pd.read_csv(os.path.join(oximeter_source_path, file))
                        oximeter_dfs[name] = df.dropna().reset_index(drop=True)
                    except Exception as e: print(f"  Error reading {file}: {e}")
                
                if oximeter_dfs:
                    oximeter_filt = filter_window(oximeter_dfs, start_ts, end_ts)
                    oximeter_original = copy.deepcopy(oximeter_filt)
                    has_data = any(not df.empty for df in oximeter_filt.values())
                    if has_data:
                        oximeter_vis_dir = os.path.join(segment_vis_dir, 'Oximeter')
                        if not os.path.exists(oximeter_vis_dir): os.makedirs(oximeter_vis_dir)
                        os.chdir(oximeter_vis_dir)
                        print(f"  生成段 {segment} Oximeter图表...")
                        plot_oximeter_distribution_analysis(oximeter_original, save=True)
                        if os.path.exists("oximeter_distribution_analysis.png"): os.rename("oximeter_distribution_analysis.png", f"oximeter_distribution_analysis_seg{segment}.png")
                        plot_combined(oximeter_original, save=True)
                        if os.path.exists("combined_excluding_sensors.png"): os.rename("combined_excluding_sensors.png", f"oximeter_combined_no_filter_seg{segment}.png")
                        plot_subplots(oximeter_original, save=True)
                        if os.path.exists("subplots_excluding_sensors.png"): os.rename("subplots_excluding_sensors.png", f"oximeter_subplots_no_filter_seg{segment}.png")
                        os.chdir(original_cwd)
                    else: print(f"  段 {segment} Oximeter数据在指定时间窗口内无数据")
        
        # 处理HUB数据
        hub_source_path = os.path.join(segment_dir, 'HUB')
        if os.path.exists(hub_source_path):
            print(f"\n处理段 {segment} - HUB数据...")
            hub_files = [f for f in os.listdir(hub_source_path) if f.endswith('.csv')]
            if hub_files:
                hub_dfs = {}
                for file in hub_files:
                    name = file[:-4]
                    try:
                        df = pd.read_csv(os.path.join(hub_source_path, file))
                        hub_dfs[name] = df.dropna().reset_index(drop=True)
                    except Exception as e: print(f"  Error reading {file}: {e}")
                
                if hub_dfs:
                    hub_filt = filter_window(hub_dfs, start_ts, end_ts)
                    hub_original = copy.deepcopy(hub_filt)
                    has_data = any(not df.empty for df in hub_filt.values())
                    if has_data:
                        hub_vis_dir = os.path.join(segment_vis_dir, 'HUB')
                        if not os.path.exists(hub_vis_dir): os.makedirs(hub_vis_dir)
                        os.chdir(hub_vis_dir)
                        print(f"  生成段 {segment} HUB图表...")
                        plot_channels_grid(hub_original, save=True)
                        if os.path.exists("channels_grid.png"): os.rename("channels_grid.png", f"hub_channels_grid_seg{segment}.png")
                        plot_all_channels_overlay_filtered(hub_original, save=True)
                        if os.path.exists("all_channels_overlay_filtered.png"): os.rename("all_channels_overlay_filtered.png", f"hub_channels_overlay_seg{segment}.png")
                        plot_psd_analysis(hub_filt, save=True)
                        if os.path.exists("psd_sensors.png"): os.rename("psd_sensors.png", f"hub_psd_sensors_seg{segment}.png")
                        if os.path.exists("psd_sensor_grid.png"): os.rename("psd_sensor_grid.png", f"hub_psd_grid_seg{segment}.png")
                        os.chdir(original_cwd)
                    else: print(f"  段 {segment} HUB数据在指定时间窗口内无数据")
        
        segments_processed += 1
    
    print(f"\n" + "="*60)
    print(f"可视化完成! 处理了 {segments_processed} 个段")
    print(f"所有图表已保存至: {vis_dir}")


def apply_filtering(dfs_filt):
    """对数据应用滤波"""
    for name, df in dfs_filt.items():
        if df.empty:
            continue

        ts = df['timestamp'].values
        tsu = np.unique(ts)
        if len(tsu) < 2:
            print(f"{name}: 没有足够不同时间戳，跳过滤波")
            continue
            
        dt = np.median(np.diff(tsu))
        if dt <= 0:
            print(f"{name}: 时间戳异常 (dt={dt:.6f}), 跳过滤波")
            continue

        fs = 1.0 / dt
        print(f"{name}: 采样率 ≈ {fs:.1f} Hz, Nyquist={fs/2:.1f} Hz")

        nyquist = fs / 2
        if nyquist <= 1.5:
            print(f"{name}: 采样率太低 (Nyquist={nyquist:.1f} Hz <= 1.5 Hz)，跳过滤波")
            continue
        
        if fs < 10:
            highcut = min(3.0, 0.9 * nyquist)
            print(f"{name}: 调整滤波范围为 0.5-{highcut:.1f} Hz")
        else:
            highcut = 3.0

        for col in df.columns:
            if col == 'timestamp':
                continue
            try:
                # 使用 copy() 避免 SettingWithCopyWarning
                df_copy = dfs_filt[name].copy()
                df_copy[col] = bandpass_filter(
                    df[col].values,
                    lowcut=0.5,
                    highcut=highcut,
                    fs=fs)
                dfs_filt[name] = df_copy
            except Exception as e:
                print(f"{name} {col}: 滤波失败 - {str(e)}")


def calculate_average_hr(base_dir):
    """自动计算所有段的平均心率"""
    print("\n" + "="*50)
    print("心率分析 (自动模式)")
    print("="*50)
    
    hr_files = []
    parent_dir = os.path.dirname(os.path.abspath(base_dir))
    for i in range(1, 12):
        filepath = os.path.join(parent_dir, str(i), "Biopac", "hr.csv")
        if os.path.exists(filepath):
            hr_files.append((i, filepath))

    if not hr_files:
        print("错误:在任何段文件夹的 'Biopac' 子目录中没有找到hr.csv文件")
        return
    
    print(f"\n找到 {len(hr_files)} 个HR文件进行分析...")
    print("-" * 50)
    
    all_results = []
    for num, filepath in hr_files:
        try:
            df = pd.read_csv(filepath)
            if 'hr' in df.columns and not df['hr'].empty:
                average_hr = df['hr'].mean()
                all_results.append({
                    'segment': num,
                    'file': os.path.basename(filepath),
                    'data_points': len(df),
                    'avg_hr': average_hr
                })
                print(f"段 {num}: 平均心率 = {average_hr:.2f} BPM ({len(df)} 数据点)")
            else:
                 print(f"段 {num}: 文件 {filepath} 为空或无 'hr' 列")
        except Exception as e:
            print(f"段 {num}: 读取错误 - {str(e)}")
    
    if all_results:
        print("-" * 50)
        print("\n汇总统计:")
        total_hr = sum(r['avg_hr'] for r in all_results)
        overall_avg = total_hr / len(all_results)
        print(f"所有段的平均心率: {overall_avg:.2f} BPM")
        
        max_hr = max(all_results, key=lambda x: x['avg_hr'])
        min_hr = min(all_results, key=lambda x: x['avg_hr'])
        print(f"最高平均心率: 段 {max_hr['segment']} - {max_hr['avg_hr']:.2f} BPM")
        print(f"最低平均心率: 段 {min_hr['segment']} - {min_hr['avg_hr']:.2f} BPM")


# ==================== 主程序 ====================
def main():
    """主程序入口"""
    parser = argparse.ArgumentParser(
        description='综合生理信号处理脚本 - Biopac同步与自动化分析',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('input_file', 
                        help='Biopac导出的CSV文件路径')
    parser.add_argument('-o', '--output', default='output', 
                        help='输出目录 (默认: output)')
    parser.add_argument('-t', '--timezone', type=int, default=8, 
                        help='UTC时差,单位为小时 (默认: 8)')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("生理信号处理系统 - Biopac同步与自动化分析")
    print("="*60)
    
    if not os.path.exists(args.input_file):
        print(f"错误：输入文件 {args.input_file} 不存在")
        return
    
    try:
        # Step 1: Biopac数据同步与切片
        print("\n" + "="*50)
        print("步骤 1: Biopac数据同步与切片")
        print("="*50)
        convert_biopac_to_csv(args.input_file, args.output, args.timezone)
        
        # Step 2: 自动化数据可视化
        print("\n" + "="*50)
        print("步骤 2: 数据可视化 (自动模式)")
        print("="*50)
        run_visualization(args.output)
        
        # Step 3: 自动化心率分析
        print("\n" + "="*60)
        print("步骤 3: 心率分析 (自动模式)")
        print("="*60)
        calculate_average_hr(args.output)
        
        # Step 4: 运行新增的对比分析
        parent_dir = os.path.dirname(os.path.abspath(args.output))
        vis_dir = os.path.join(parent_dir, 'visualization')
        if not os.path.exists(vis_dir): os.makedirs(vis_dir)
        run_comparison_analysis(parent_dir, vis_dir)

        print("\n" + "="*60)
        print("所有处理步骤已完成！")
        print(f"主要输出保存在: {args.output}/")
        print(f"所有图表保存在: {vis_dir}/")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n\n用户中断处理")
    except Exception as e:
        print(f"\n处理过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
