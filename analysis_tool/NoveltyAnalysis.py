import pandas as pd
import numpy as np
from Bio import SeqIO, Align
from Bio.Align import substitution_matrices
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import argparse  # 新增：用于解析命令行参数

# ================= 配置区域 =================
# 数据库路径 (硬编码)
DATABASE_FILE = "/t9k/mnt/AMP-rebuilt/data/sequence.fasta"

# 其他配置
CPU_CORES = -1  # -1 表示使用所有可用 CPU 核心

# ================= 核心分析函数 =================

def load_sequences(file_path):
    """读取 FASTA 文件并返回序列列表"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件未找到: {file_path}")
    return [str(record.seq) for record in SeqIO.parse(file_path, "fasta")]

def setup_aligner():
    """配置序列比对器 (与 MMCD 论文一致使用 BLOSUM62)"""
    aligner = Align.PairwiseAligner()
    try:
        aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    except Exception as e:
        print(f"警告: 无法加载 BLOSUM62 矩阵 ({e})，将使用默认计分。")
    return aligner

def analyze_single_sequence(gen_seq, db_sequences, aligner):
    """
    计算单个生成序列相对于整个数据库的两种指标：
    1. Max Score: 最佳匹配得分 (用于检测是否只是单纯复制/泄露)
    2. Mean Score: 平均匹配得分 (MMCD 论文使用的指标)
    """
    scores = []
    # 遍历数据库计算得分
    for db_seq in db_sequences:
        score = aligner.score(gen_seq, db_seq)
        scores.append(score)
    
    scores_arr = np.array(scores)
    return {
        "sequence": gen_seq,
        "max_score": np.max(scores_arr),
        "mean_score": np.mean(scores_arr)
    }

# ================= 主程序 =================

def main():
    # --- 1. 参数解析 ---
    parser = argparse.ArgumentParser(description="AMP Novelty Analysis Script")
    parser.add_argument("input_fasta", type=str, help="Path to the generated FASTA file")
    args = parser.parse_args()

    generated_file = args.input_fasta
    
    # --- 2. 自动设置输出路径 ---
    # 获取输入文件的目录和不带后缀的文件名
    file_dir = os.path.dirname(generated_file)
    file_name = os.path.basename(generated_file)
    base_name = os.path.splitext(file_name)[0]
    
    # 自动生成输出路径 (保存在与输入文件相同的目录下，添加后缀)
    output_csv = os.path.join(file_dir, f"{base_name}_novelty_analysis_results.csv")
    output_img = os.path.join(file_dir, f"{base_name}_novelty_distribution_enhanced.png")

    print(f"Input File:  {generated_file}")
    print(f"Output CSV:  {output_csv}")
    print(f"Output Img:  {output_img}")
    print("-" * 30)

    start_time = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] 开始加载数据...")

    # --- 3. 加载数据 ---
    try:
        db_seqs = load_sequences(DATABASE_FILE)
        gen_seqs = load_sequences(generated_file)
        print(f"数据库序列数: {len(db_seqs)}")
        print(f"生成序列数: {len(gen_seqs)}")
    except Exception as e:
        print(f"错误: {e}")
        return

    # --- 4. 初始化比对器 ---
    aligner = setup_aligner()

    # --- 5. 并行计算 ---
    print(f"[{time.strftime('%H:%M:%S')}] 开始并行分析 (Cores: {CPU_CORES})...")
    
    results = Parallel(n_jobs=CPU_CORES, verbose=5)(
        delayed(analyze_single_sequence)(seq, db_seqs, aligner) 
        for seq in gen_seqs
    )

    # --- 6. 整理并保存结果 ---
    df_results = pd.DataFrame(results)
    
    # 确保输出目录存在 (通常输入文件存在目录就存在，但为了保险)
    if file_dir and not os.path.exists(file_dir):
        os.makedirs(file_dir, exist_ok=True)
        
    df_results.to_csv(output_csv, index=False)
    print(f"分析结果已保存至: {output_csv}")

    # --- 7. 统计摘要 ---
    print("\n" + "="*30)
    print("分析摘要 (Analysis Summary)")
    print("="*30)
    print(f"Max Score (Leakage Metric) Mean: {df_results['max_score'].mean():.4f}")
    print(f"Mean Score (Paper Metric) Mean:  {df_results['mean_score'].mean():.4f}")
    print("-" * 30)
    
    # 简单的“泄露”预警
    high_similarity_count = (df_results['max_score'] > 100).sum()
    print(f"高相似度序列数 (Max Score > 100): {high_similarity_count}")

    # --- 8. 绘图 ---
    print(f"[{time.strftime('%H:%M:%S')}] 正在绘制图表...")
    plt.figure(figsize=(14, 6))
    sns.set_style("whitegrid")

    # 子图 1: Max Score
    plt.subplot(1, 2, 1)
    sns.histplot(df_results['max_score'], kde=True, color='skyblue', bins=30)
    plt.title(f'Distribution of Max Alignment Scores\n({base_name})', fontsize=12)
    plt.xlabel('Max BLOSUM62 Score (vs Database)', fontsize=10)
    plt.ylabel('Count', fontsize=10)
    plt.axvline(df_results['max_score'].mean(), color='red', linestyle='--', label=f'Mean: {df_results["max_score"].mean():.2f}')
    plt.legend()

    # 子图 2: Mean Score
    plt.subplot(1, 2, 2)
    sns.histplot(df_results['mean_score'], kde=True, color='salmon', bins=30)
    plt.title(f'Distribution of Mean Alignment Scores\n({base_name})', fontsize=12)
    plt.xlabel('Mean BLOSUM62 Score (vs Database)', fontsize=10)
    plt.ylabel('Count', fontsize=10)
    plt.axvline(df_results['mean_score'].mean(), color='red', linestyle='--', label=f'Mean: {df_results["mean_score"].mean():.2f}')
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_img, dpi=300)
    print(f"图表已保存至: {output_img}")

    print(f"[{time.strftime('%H:%M:%S')}] 完成。总耗时: {time.time() - start_time:.2f} 秒")

if __name__ == "__main__":
    main()