# physio-signal-tools

## 概述
仓库包含两个Python脚本，用于处理和可视化生理信号数据：
- `biopac_sync.py`: 将BIOPAC导出的csv文件转换为标准格式
- `visualization.py`: 对齐并可视化多通道生理信号

## 环境
```bash
pip install pandas numpy matplotlib
```

## 使用流程

### 1. 数据转换 (`biopac_sync.py`)

**输入：**
```bash
python biopac_sync.py input_file.csv
```

**输出：**
- 为每个通道生成独立的CSV文件（如`bp.csv`, `hr.csv`等）
- 每个文件包含两列：`timestamp`（Unix时间戳）和信号值

### 2. 数据可视化 (`visualization.py`)

**输入：**
```bash
python visualization.py
```

**交互步骤：**
1. 运行脚本后，输入时间窗口：
   ```
   请输入开始时间戳 (Unix 秒)：1640995200.123
   请输入结束时间戳 (Unix 秒)：1640995800.456
   ```
2. 脚本自动对齐所有通道数据并生成可视化图表

**输出：**
- 在屏幕显示对齐后的多通道波形图
- 保存为`combined_excluding_sensors.png`,`subplots_excluding_sensors.png`,`channels_grid.png`

## 工作流程

1. **准备数据**: 从BIOPAC导出csv格式数，放入同文件夹
2. **转换数据**: `python biopac_sync.py data.csv`
3. **检查输出**: 确认`output/`目录下生成的csv文件
4. **可视化**: `python visualization.py`，输入想可视化的时间段
5. **分析结果**: 查看生成的`combined_excluding_sensors.png`等图表

## 文件结构
```
project/
├── biopac_sync.py      # 数据转换脚本
├── visualization.py    # 可视化脚本
├── input_data.csv      # 原始BIOPAC数据
├── output/             # 转换后的CSV文件
│   ├── bp.csv
│   ├── hr.csv
│   └── ...
├── combined_excluding_sensors.png # 真值+血氧总图
├── subplots_excluding_sensors.png # 真值+血氧拼接子图
└── channels_grid.png # 可穿戴总图
```
