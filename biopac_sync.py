import csv
import os
import re
from datetime import datetime, timezone, timedelta
import argparse


def parse_biopac_header(lines, utc_offset=8):
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
    """
    获取所有数字命名文件夹中的时间段信息
    返回：[(segment_num, start_ts, end_ts), ...]
    """
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
    """
    根据时间段切分已生成的标准CSV文件
    """
    if not segments:
        print("No segments found, skipping segmentation.")
        return
    
    print(f"\nSegmenting files based on {len(segments)} time periods...")
    
    # 查找所有生成的CSV文件
    csv_files = [f for f in os.listdir(output_dir) if f.endswith('.csv')]
    
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
                # 保存切分后的文件
                segment_filename = f"{base_name}-{seg_num}.csv"
                segment_path = os.path.join(output_dir, segment_filename)
                
                with open(segment_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(segment_rows)
                
                print(f"  Segment {seg_num}: {len(segment_rows)} rows -> {segment_filename}")
            else:
                print(f"  Segment {seg_num}: No data in time range")


def convert_biopac_to_csv(input_file, output_dir='output', utc_offset=8, auto_segment=True):
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


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('input_file', help='Biopac exported CSV file')
    p.add_argument('-o','--output',default='output', help='Output directory')
    p.add_argument('-t','--timezone',type=int,default=8, help='UTC offset in hours')
    p.add_argument('--no-segment', action='store_true', help='Disable automatic segmentation')
    a=p.parse_args()
    convert_biopac_to_csv(a.input_file, a.output, a.timezone, auto_segment=not a.no_segment)
