import pandas as pd
import argparse
import sys

def main():
    # 1. 创建参数解析器
    parser = argparse.ArgumentParser(description="统计 CSV 文件中 Ensemble 列的 True/False 占比")
    
    # 2. 添加参数：'file_path'
    # type=str 表示输入必须是字符串，help 是帮助信息
    parser.add_argument('file_path', type=str, help='CSV 文件的路径')
    
    # 3. 解析参数
    args = parser.parse_args()
    
    # 获取用户输入的文件路径
    file_path = args.file_path
    
    print(f"正在处理文件: {file_path}")

    try:
        # 读取 CSV 文件
        df = pd.read_csv(file_path)

        # 检查是否存在 'Ensemble' 列
        if 'Ensemble' not in df.columns:
            print("错误: CSV 文件中未找到 'Ensemble' 列。")
            return

        # 统计 'Ensemble' 列中 True 和 False 的占比
        percentage = df['Ensemble'].value_counts(normalize=True)

        # 打印结果
        print("\nTrue/False 占比:")
        print(percentage)

        # 如果想看百分比格式 (例如 50%)
        print("\n百分比格式:")
        print(percentage.mul(100).round(2).astype(str) + '%')

        # 如果想看具体的数量
        print("\n具体数量:")
        print(df['Ensemble'].value_counts())

    except FileNotFoundError:
        print(f"错误: 找不到文件 '{file_path}'，请检查路径是否正确。")
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    main()