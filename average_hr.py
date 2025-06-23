import pandas as pd
import os

def calculate_average_hr():
    file_number = input("请输入HR文件编号 (1-11): ")
    
    # 验证输入
    try:
        num = int(file_number)
        if num < 1 or num > 11:
            print("错误：请输入1-11之间的数字")
            return
    except ValueError:
        print("错误：请输入有效的数字")
        return
    
    # 构建文件路径
    filename = f"hr{num}.csv"
    filepath = os.path.join("output", filename)
    
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
        print(f"文件: {filename}")
        print(f"总数据点: {len(df)}")
        print(f"平均心率: {average_hr:.2f} BPM")
        
    except Exception as e:
        print(f"读取文件时出错: {str(e)}")

if __name__ == "__main__":
    calculate_average_hr()
