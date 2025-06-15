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
        '血压':'bp','心输出量':'cardiac_output','全身血管阻力':'systemic_vascular_resistance','心脏指数':'cardiac_index',
        'Systolic_BP':'systolic_bp','Diastolic_BP':'diastolic_bp','Mean_BP':'mean_bp','Heart_Rate':'hr'
    }
    for k,v in abbrev.items():
        if k in safe:
            return v
    return safe.lower()


def convert_biopac_to_csv(input_file, output_dir='output', utc_offset=8):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
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

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('input_file'); p.add_argument('-o','--output',default='output'); p.add_argument('-t','--timezone',type=int,default=8)
    a=p.parse_args(); convert_biopac_to_csv(a.input_file,a.output,a.timezone)