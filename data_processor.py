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
from collections import OrderedDict
from scipy.signal import welch, butter, filtfilt
import copy


# ==================== 全局配置 ====================
sensor_mapping = {
    'sensor1': 'wrist',
    'sensor2': 'nose',
    'sensor3': 'finger',
    'sensor4': 'forearm',
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
                            if 'timestamp' in row:  # 确保timestamp列存在
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
    
    # 按段号排序
    segments.sort(key=lambda x: x[0])
    return segments


def segment_csv_files(output_dir, input_dir, segments):
    """根据时间段切分已生成的标准CSV文件，并将其放到对应的段文件夹中"""
    if not segments:
        print("No segments found, skipping segmentation.")
        return
    
    print(f"\nSegmenting files based on {len(segments)} time periods...")
    
    # 查找所有生成的CSV文件
    csv_files = [f for f in os.listdir(output_dir) if f.endswith('.csv') and not '-' in f]
    
    for csv_file in csv_files:
        base_name = csv_file[:-4]  # 去除.csv后缀
        csv_path = os.path.join(output_dir, csv_file)
        
        print(f"\nProcessing {csv_file}...")
        
        # 读取整个CSV文件
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
        if not rows:
            print(f"  No data in {csv_file}, skipping.")
            continue
        
        # 获取列名
        fieldnames = reader.fieldnames
        
        # 对每个时间段进行切分
        for seg_num, start_ts, end_ts in segments:
            # 筛选在时间段内的行
            segment_rows = []
            for row in rows:
                try:
                    ts = float(row['timestamp'])
                    if start_ts <= ts <= end_ts:
                        segment_rows.append(row)
                except (ValueError, KeyError):
                    pass
            
            if segment_rows:
                # 根据文件类型确定目标文件夹
                segment_folder = os.path.join(input_dir, str(seg_num))
                
                # 判断文件类型并放到对应子文件夹
                if base_name.startswith('sensor'):
                    # HUB数据
                    target_folder = os.path.join(segment_folder, 'HUB')
                    # 创建目标文件夹（如果不存在）
                    if not os.path.exists(target_folder):
                        os.makedirs(target_folder)
                    # 保存切分后的文件（不带段号后缀）
                    segment_path = os.path.join(target_folder, csv_file)
                elif base_name in ['bvp', 'spo2']:
                    # Oximeter数据（不包括hr）
                    target_folder = os.path.join(segment_folder, 'Oximeter')
                    # 创建目标文件夹（如果不存在）
                    if not os.path.exists(target_folder):
                        os.makedirs(target_folder)
                    # 保存切分后的文件（不带段号后缀）
                    segment_path = os.path.join(target_folder, csv_file)
                else:
                    # 其他文件（包括hr）保留在output根目录，生成分段文件
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


def convert_biopac_to_csv(input_file, output_dir='output', utc_offset=8, auto_segment=True):
    """转换Biopac数据到CSV格式"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 获取输入文件所在目录
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
    
    # 自动进行时间段切分
    if auto_segment:
        segments = get_segment_timestamps(input_dir)
        if segments:
            segment_csv_files(output_dir, input_dir, segments)
        else:
            print("\nNo numbered folders with timestamps found for segmentation.")


# ==================== 可视化模块 ====================
def bandpass_filter(data, lowcut=0.5, highcut=3, fs=30, order=3):
    """0.5-3Hz 带通滤波"""
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
    """分析真值数据分布的拼接柱状图"""
    # 获取非sensor数据
    biopac_data = {
        name: df for name, df in dfs_filt.items()
        if name.split('-')[0] not in sensor_mapping and not df.empty
    }
    
    if not biopac_data:
        print("No Biopac data found for distribution analysis.")
        return
    
    n_datasets = len(biopac_data)
    if n_datasets == 0:
        return
    
    # 计算子图布局
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
    
    # 确保axes是数组格式
    if n_datasets == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    
    for idx, (name, df) in enumerate(biopac_data.items()):
        ax = axes[idx]
        
        # 获取数据列（第二列通常是数值列）
        if df.shape[1] > 1:
            data_column = df.iloc[:, 1].values
            
            # 移除异常值（使用IQR方法）
            Q1 = np.percentile(data_column, 25)
            Q3 = np.percentile(data_column, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # 过滤数据
            filtered_data = data_column[(data_column >= lower_bound) & (data_column <= upper_bound)]
            
            # 生成柱状图
            n_bins = min(30, max(10, int(np.sqrt(len(filtered_data)))))
            counts, bins, patches = ax.hist(filtered_data, bins=n_bins, 
                                          color=colors[idx % len(colors)], 
                                          alpha=0.7, edgecolor='black', linewidth=0.5)
            
            # 设置标题和标签
            clean_name = name.replace('_', ' ').title()
            ax.set_title(f'Distribution of {clean_name}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Value', fontsize=10)
            ax.set_ylabel('Number', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # 添加统计信息
            mean_val = np.mean(filtered_data)
            std_val = np.std(filtered_data)
            median_val = np.median(filtered_data)
            
            # 在图上添加垂直线显示统计信息
            ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8, linewidth=2, label=f'Mean: {mean_val:.2f}')
            ax.axvline(median_val, color='green', linestyle='--', alpha=0.8, linewidth=2, label=f'Median: {median_val:.2f}')
            
            # 添加文本框显示统计信息
            stats_text = f'N: {len(filtered_data)}\nMean: {mean_val:.2f}\nStd: {std_val:.2f}\nMedian: {median_val:.2f}'
            ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, 
                   fontsize=9, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # 添加图例
            ax.legend(loc='upper left', fontsize=8)
            
            print(f"  {name}: {len(filtered_data)} data points, Mean={mean_val:.2f}, Std={std_val:.2f}")
    
    # 隐藏多余的子图
    for idx in range(n_datasets, len(axes)):
        axes[idx].axis('off')
    
    # 设置总标题
    plt.suptitle('Biopac Data Distribution Analysis', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save:
        plt.savefig('biopac_distribution_analysis.png', dpi=300, bbox_inches='tight')
        print("Saved: biopac_distribution_analysis.png")
    
    plt.show()


def plot_combined(dfs_filt, save=True):
    """真值 + 血氧总图"""
    plt.figure(figsize=(14, 8))

    # 全局时间起点
    all_ts = np.concatenate([
        df['timestamp'].values
        for name, df in dfs_filt.items()
        if not df.empty and name.split('-')[0] not in sensor_mapping
    ])
    
    if len(all_ts) == 0:
        print("No non-sensor data found for combined plot.")
        plt.close()
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
    plt.show()


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

    # 如果只有一行，确保axes是二维的
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # 计算全局起始时间
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

    plt.show()


def plot_channels_separately(dfs_filt, save=True):
    """可穿戴子图"""
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


def plot_all_channels_overlay_filtered(dfs_filt, save=True):
    """带滤波的五个传感器每个通道叠加总图"""
    # 获取传感器数据并进行深拷贝以避免修改原数据
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
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
    
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


def plot_psd_analysis(dfs_filt, save=True):
    """PSD 频谱图"""
    # Sensor 单通道 PSD
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

    # Sensor 聚合网格 PSD
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


def run_visualization(base_dir='output', mode='both'):
    """运行可视化流程 - 修改版本"""
    print("\n" + "="*50)
    print("开始可视化处理...")
    print("="*50)
    
    # 获取父目录路径（与output同级）
    parent_dir = os.path.dirname(os.path.abspath(base_dir))
    
    # 1. 先扫描存在的段文件夹
    available_segments = []
    for segment in range(1, 12):
        segment_dir = os.path.join(parent_dir, str(segment))
        if os.path.exists(segment_dir):
            # 检查是否有HUB或Oximeter文件夹
            hub_path = os.path.join(segment_dir, 'HUB')
            oximeter_path = os.path.join(segment_dir, 'Oximeter')
            if os.path.exists(hub_path) or os.path.exists(oximeter_path):
                available_segments.append(segment)
    
    if not available_segments:
        print("没有找到任何包含HUB或Oximeter数据的段文件夹")
        return
        
    print(f"找到 {len(available_segments)} 个数据段: {available_segments}")
    
    # 创建visualization主文件夹（与output同级）
    vis_dir = os.path.join(parent_dir, 'visualization')
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
        print(f"Created visualization directory: {vis_dir}")
    
    # 保存当前工作目录
    original_cwd = os.getcwd()
    
    # 获取Biopac数据文件（output文件夹中的直接数据）
    biopac_files = [f for f in os.listdir(base_dir) 
                    if f.endswith('.csv') and not f.startswith('sensor') and '-' not in f]
    
    # 2. 处理每个段的数据（分别输入时间戳）
    segments_processed = 0
    for segment in available_segments:
        segment_dir = os.path.join(parent_dir, str(segment))
        
        print(f"\n" + "="*50)
        print(f"处理段 {segment} 数据...")
        print("="*50)
        
        # 为每个段单独输入时间戳
        print(f"请为段 {segment} 输入时间窗口:")
        try:
            start_ts = float(input(f"Enter start timestamp for segment {segment} (Unix seconds): "))
            end_ts = float(input(f"Enter end timestamp for segment {segment} (Unix seconds): "))
        except ValueError:
            print(f"无效的时间戳输入，跳过段 {segment}")
            continue
        
        # 创建段级visualization文件夹
        segment_vis_dir = os.path.join(vis_dir, str(segment))
        if not os.path.exists(segment_vis_dir):
            os.makedirs(segment_vis_dir)
        
        # 2.1 处理Biopac数据（为每个段生成）
        if biopac_files:
            print(f"\n处理段 {segment} - Biopac数据...")
            
            biopac_dfs = {}
            for file in biopac_files:
                name = file[:-4]
                try:
                    df = pd.read_csv(os.path.join(base_dir, file))
                    biopac_dfs[name] = df.dropna().reset_index(drop=True)
                except Exception as e:
                    print(f"Error reading {file}: {e}")
            
            if biopac_dfs:
                # 过滤时间窗口
                biopac_filt = filter_window(biopac_dfs, start_ts, end_ts)
                biopac_original = copy.deepcopy(biopac_filt)
                
                # 检查过滤后是否有数据
                has_data = any(not df.empty for df in biopac_filt.values())
                if has_data:
                    # 应用滤波
                    apply_filtering(biopac_filt)
                    
                    # 创建Biopac子文件夹
                    biopac_dir = os.path.join(segment_vis_dir, 'Biopac')
                    if not os.path.exists(biopac_dir):
                        os.makedirs(biopac_dir)
                    
                    os.chdir(biopac_dir)
                    
                    # 生成Biopac图表
                    print(f"  生成段 {segment} Biopac图表...")
                    
                    # 新增：分布分析图
                    plot_biopac_distribution_analysis(biopac_original, save=True)
                    if os.path.exists("biopac_distribution_analysis.png"):
                        os.rename("biopac_distribution_analysis.png", f"biopac_distribution_analysis_seg{segment}.png")
                    
                    # 无滤波图
                    plot_combined(biopac_original, save=True)
                    if os.path.exists("combined_excluding_sensors.png"):
                        os.rename("combined_excluding_sensors.png", f"biopac_combined_no_filter_seg{segment}.png")
                    
                    plot_subplots(biopac_original, save=True)
                    if os.path.exists("subplots_excluding_sensors.png"):
                        os.rename("subplots_excluding_sensors.png", f"biopac_subplots_no_filter_seg{segment}.png")
                    
                    # 滤波图
                    plot_combined(biopac_filt, save=True)
                    if os.path.exists("combined_excluding_sensors.png"):
                        os.rename("combined_excluding_sensors.png", f"biopac_combined_filtered_seg{segment}.png")
                    
                    plot_subplots(biopac_filt, save=True)
                    if os.path.exists("subplots_excluding_sensors.png"):
                        os.rename("subplots_excluding_sensors.png", f"biopac_subplots_filtered_seg{segment}.png")
                    
                    os.chdir(original_cwd)
                    print(f"  段 {segment} Biopac图表已保存至: {biopac_dir}")
                else:
                    print(f"  段 {segment} Biopac数据在指定时间窗口内无数据")
        
        # 2.2 处理Oximeter数据
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
                        print(f"  Loaded {file} with {len(df)} rows")
                    except Exception as e:
                        print(f"  Error reading {file}: {e}")
                
                if oximeter_dfs:
                    # 过滤时间窗口
                    oximeter_filt = filter_window(oximeter_dfs, start_ts, end_ts)
                    oximeter_original = copy.deepcopy(oximeter_filt)
                    
                    # 检查过滤后是否有数据
                    has_data = any(not df.empty for df in oximeter_filt.values())
                    if has_data:
                        # 应用滤波
                        apply_filtering(oximeter_filt)
                        
                        # 创建Oximeter子文件夹
                        oximeter_vis_dir = os.path.join(segment_vis_dir, 'Oximeter')
                        if not os.path.exists(oximeter_vis_dir):
                            os.makedirs(oximeter_vis_dir)
                        
                        os.chdir(oximeter_vis_dir)
                        
                        # 生成Oximeter图表
                        print(f"  生成段 {segment} Oximeter图表...")
                        
                        # 无滤波图
                        plot_combined(oximeter_original, save=True)
                        if os.path.exists("combined_excluding_sensors.png"):
                            os.rename("combined_excluding_sensors.png", f"oximeter_combined_no_filter_seg{segment}.png")
                        
                        plot_subplots(oximeter_original, save=True)
                        if os.path.exists("subplots_excluding_sensors.png"):
                            os.rename("subplots_excluding_sensors.png", f"oximeter_subplots_no_filter_seg{segment}.png")
                        
                        # 滤波图
                        plot_combined(oximeter_filt, save=True)
                        if os.path.exists("combined_excluding_sensors.png"):
                            os.rename("combined_excluding_sensors.png", f"oximeter_combined_filtered_seg{segment}.png")
                        
                        plot_subplots(oximeter_filt, save=True)
                        if os.path.exists("subplots_excluding_sensors.png"):
                            os.rename("subplots_excluding_sensors.png", f"oximeter_subplots_filtered_seg{segment}.png")
                        
                        os.chdir(original_cwd)
                        print(f"  段 {segment} Oximeter图表已保存至: {oximeter_vis_dir}")
                    else:
                        print(f"  段 {segment} Oximeter数据在指定时间窗口内无数据")
        
        # 2.3 处理HUB数据（sensor数据）
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
                        print(f"  Loaded {file} with {len(df)} rows")
                    except Exception as e:
                        print(f"  Error reading {file}: {e}")
                
                if hub_dfs:
                    # 过滤时间窗口
                    hub_filt = filter_window(hub_dfs, start_ts, end_ts)
                    hub_original = copy.deepcopy(hub_filt)
                    
                    # 检查过滤后是否有数据
                    has_data = any(not df.empty for df in hub_filt.values())
                    if has_data:
                        # 创建HUB子文件夹
                        hub_vis_dir = os.path.join(segment_vis_dir, 'HUB')
                        if not os.path.exists(hub_vis_dir):
                            os.makedirs(hub_vis_dir)
                        
                        os.chdir(hub_vis_dir)
                        
                        # 生成HUB图表
                        print(f"  生成段 {segment} HUB图表...")
                        
                        plot_channels_grid(hub_original, save=True)
                        if os.path.exists("channels_grid.png"):
                            os.rename("channels_grid.png", f"hub_channels_grid_seg{segment}.png")
                        
                        plot_all_channels_overlay_filtered(hub_original, save=True)
                        if os.path.exists("all_channels_overlay_filtered.png"):
                            os.rename("all_channels_overlay_filtered.png", f"hub_channels_overlay_seg{segment}.png")
                        
                        plot_psd_analysis(hub_filt, save=True)
                        if os.path.exists("psd_sensors.png"):
                            os.rename("psd_sensors.png", f"hub_psd_sensors_seg{segment}.png")
                        if os.path.exists("psd_sensor_grid.png"):
                            os.rename("psd_sensor_grid.png", f"hub_psd_grid_seg{segment}.png")
                        
                        os.chdir(original_cwd)
                        print(f"  段 {segment} HUB图表已保存至: {hub_vis_dir}")
                    else:
                        print(f"  段 {segment} HUB数据在指定时间窗口内无数据")
        
        segments_processed += 1
    
    print(f"\n" + "="*60)
    print(f"可视化完成! 处理了 {segments_processed} 个段")
    print(f"所有图表已保存至: {vis_dir}")
    print(f"文件夹结构:")
    print(f"├── {vis_dir}/")
    for i in available_segments:
        segment_path = os.path.join(vis_dir, str(i))
        if os.path.exists(segment_path):
            print(f"│   ├── {i}/")
            if os.path.exists(os.path.join(segment_path, 'Biopac')):
                print(f"│   │   ├── Biopac/")
            if os.path.exists(os.path.join(segment_path, 'HUB')):
                print(f"│   │   ├── HUB/")
            if os.path.exists(os.path.join(segment_path, 'Oximeter')):
                print(f"│   │   └── Oximeter/")
    print("="*60)
    
    # 在可视化流程最后添加心率分析
    print("\n" + "="*60)
    print("步骤 3: 心率分析")
    print("="*60)
    
    # 获取父目录路径以便查找output文件夹
    parent_dir = os.path.dirname(os.path.abspath(base_dir))
    output_path = os.path.join(parent_dir, 'output')
    
    # 检查output文件夹是否存在
    if os.path.exists(output_path):
        calculate_average_hr(output_path)
    else:
        # 如果在父目录没找到，尝试使用传入的base_dir
        calculate_average_hr(base_dir)
    
    print("\n" + "="*60)
    print("所有分析步骤已完成！")
    print("="*60)


def apply_filtering(dfs_filt):
    """对数据应用滤波"""
    for name, df in dfs_filt.items():
        if df.empty:
            continue

        ts = df['timestamp'].values
        # 只保留唯一的时间戳，去掉重复值
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
        print(f"{name}: 采样率 ≈ {fs:.1f} Hz, Nyquist={fs/2:.1f} Hz")

        # 检查采样率是否足够高以支持滤波
        nyquist = fs / 2
        if nyquist <= 1.5:
            print(f"{name}: 采样率太低 (Nyquist={nyquist:.1f} Hz <= 1.5 Hz)，跳过滤波")
            continue
        
        # 对于低采样率信号，调整滤波参数
        if fs < 10:
            highcut = min(3.0, 0.9 * nyquist)
            print(f"{name}: 调整滤波范围为 0.5-{highcut:.1f} Hz")
        else:
            highcut = 3.0

        for col in df.columns:
            if col == 'timestamp':
                continue
            try:
                dfs_filt[name][col] = bandpass_filter(
                    df[col].values,
                    lowcut=0.5,
                    highcut=highcut,
                    fs=fs)
            except Exception as e:
                print(f"{name} {col}: 滤波失败 - {str(e)}")


def calculate_average_hr(output_dir='output'):
    """计算平均心率"""
    print("\n" + "="*50)
    print("心率分析")
    print("="*50)
    
    file_number = input("请输入HR文件编号 (1-11,或输入 'all' 分析所有文件): ")
    
    if file_number.lower() == 'all':
        # 分析所有HR文件
        hr_files = []
        for i in range(1, 12):
            filename = f"hr-{i}.csv"
            filepath = os.path.join(output_dir, filename)
            if os.path.exists(filepath):
                hr_files.append((i, filepath))
        
        if not hr_files:
            print("错误:没有找到任何HR文件")
            return
        
        print(f"\n找到 {len(hr_files)} 个HR文件")
        print("-" * 50)
        
        all_results = []
        for num, filepath in hr_files:
            try:
                df = pd.read_csv(filepath)
                if 'hr' in df.columns:
                    average_hr = df['hr'].mean()
                    all_results.append({
                        'segment': num,
                        'file': f"hr-{num}.csv",
                        'data_points': len(df),
                        'avg_hr': average_hr
                    })
                    print(f"段 {num}: 平均心率 = {average_hr:.2f} BPM ({len(df)} 数据点)")
            except Exception as e:
                print(f"段 {num}: 读取错误 - {str(e)}")
        
        if all_results:
            print("-" * 50)
            print("\n汇总统计:")
            total_hr = sum(r['avg_hr'] for r in all_results)
            overall_avg = total_hr / len(all_results)
            print(f"所有段的平均心率: {overall_avg:.2f} BPM")
            
            # 找出最高和最低
            max_hr = max(all_results, key=lambda x: x['avg_hr'])
            min_hr = min(all_results, key=lambda x: x['avg_hr'])
            print(f"最高平均心率: 段 {max_hr['segment']} - {max_hr['avg_hr']:.2f} BPM")
            print(f"最低平均心率: 段 {min_hr['segment']} - {min_hr['avg_hr']:.2f} BPM")
    
    else:
        # 分析单个文件
        try:
            num = int(file_number)
            if num < 1 or num > 11:
                print("错误:请输入1-11之间的数字")
                return
        except ValueError:
            print("错误：请输入有效的数字或 'all'")
            return
        
        # 构建文件路径
        filename = f"hr-{num}.csv"
        filepath = os.path.join(output_dir, filename)
        
        # 检查文件是否存在
        if not os.path.exists(filepath):
            print(f"错误：文件 {filepath} 不存在")
            return
        
        try:
            # 读取CSV文件
            df = pd.read_csv(filepath)
            
            # 检查文件格式
            if 'hr' not in df.columns:
                print("错误：文件中没有找到'hr'列")
                return
            
            # 计算平均心率
            average_hr = df['hr'].mean()
            
            # 输出结果
            print(f"\n文件: {filename}")
            print(f"总数据点: {len(df)}")
            print(f"平均心率: {average_hr:.2f} BPM")
            
            # 额外统计
            print(f"最小心率: {df['hr'].min():.2f} BPM")
            print(f"最大心率: {df['hr'].max():.2f} BPM")
            print(f"心率标准差: {df['hr'].std():.2f} BPM")
            
        except Exception as e:
            print(f"读取文件时出错: {str(e)}")


# ==================== 心率分析模块 ====================


# ==================== 主程序 ====================
def main():
    """主程序入口"""
    parser = argparse.ArgumentParser(
        description='综合生理信号处理脚本 - Biopac同步与可视化',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('input_file', 
                        help='Biopac导出的CSV文件路径')
    parser.add_argument('-o', '--output', default='output', 
                        help='输出目录 (默认: output)')
    parser.add_argument('-t', '--timezone', type=int, default=8, 
                        help='UTC时差,单位为小时 (默认: 8)')
    parser.add_argument('--vis-mode', choices=['combined', 'subplots', 'both'], 
                        default='both', help='可视化模式 (默认: both)')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("生理信号处理系统 - Biopac同步与可视化")
    print("="*60)
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input_file):
        print(f"错误：输入文件 {args.input_file} 不存在")
        return
    
    try:
        # Step 1: Biopac数据同步与切片
        print("\n" + "="*50)
        print("步骤 1: Biopac数据同步与切片")
        print("="*50)
        convert_biopac_to_csv(args.input_file, args.output, args.timezone)
        
        # Step 2: 数据可视化
        print("\n" + "="*50)
        print("步骤 2: 数据可视化")
        print("="*50)
        run_visualization(args.output, args.vis_mode)
        
        print("\n" + "="*60)
        print("所有处理步骤已完成！")
        print(f"输出文件保存在: {args.output}/")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n\n用户中断处理")
    except Exception as e:
        print(f"\n处理过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
