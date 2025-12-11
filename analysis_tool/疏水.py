import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio import SeqIO
import math
import os
import argparse
import sys

# ================= 配置区域 =================

# 1. 永久硬编码的参考文件 (始终会被加载)
REF_FILE = "/t9k/mnt/AMP-rebuilt/data/sequence.fasta"

# 2. 默认结果保存路径 (可通过命令行参数 --outdir 修改)
DEFAULT_OUTPUT_DIR = "/t9k/mnt/AMP-rebuilt/output/fted-t5/results_analysis/" 

# 3. 颜色映射 (Key = 文件名不带后缀)
# 注意：如果你传入的新文件名为 my_new_file.fasta，它的 Key 就是 'my_new_file'。
# 如果没有在这里定义颜色，脚本会使用默认的灰色。
COLOR_MAP = {
    'generated': '#1f77b4',  # 蓝色
    'positive': '#2ca02c',   # 绿色
    'negative': '#d62728',   # 红色
    'random': '#ff7f0e',     # 橙色
    'sequence': '#2ca02c',   # 参考文件 sequence.fasta 通常作为 positive 或 baseline，这里设为绿色
}

# ===========================================

def calculate_hydrophobic_moment(seq, angle=100):
    """
    计算疏水矩 (Eisenberg scale, 100度 alpha-helix).
    """
    hydrophobicity_scale = {
        'A': 0.62, 'R': -2.53, 'N': -0.78, 'D': -0.90, 'C': 0.29,
        'Q': -0.85, 'E': -0.74, 'G': 0.48, 'H': -0.40, 'I': 1.38,
        'L': 1.06, 'K': -1.50, 'M': 0.64, 'F': 1.19, 'P': 0.12,
        'S': -0.18, 'T': -0.05, 'W': 0.81, 'Y': 0.26, 'V': 1.08
    }
    seq = seq.upper()
    h_vals = [hydrophobicity_scale.get(aa, 0.0) for aa in seq]
    
    rad = math.radians(angle)
    sum_cos = sum(h * math.cos(i * rad) for i, h in enumerate(h_vals))
    sum_sin = sum(h * math.sin(i * rad) for i, h in enumerate(h_vals))
        
    l = len(seq)
    if l == 0: return 0.0
    return math.sqrt(sum_cos**2 + sum_sin**2) / l

def load_fasta_files(file_paths):
    """读取 fasta 文件并返回字典"""
    data_dict = {}
    AA20 = set("ACDEFGHIKLMNPQRSTVWY")
    
    for path in file_paths:
        # 路径去重检查，防止重复加载
        if not os.path.exists(path):
            print(f"[警告] 文件不存在，跳过: {path}")
            continue
            
        filename = os.path.basename(path)
        label = os.path.splitext(filename)[0]
        
        # 读取并简单的预处理（只保留标准氨基酸组成的序列）
        seqs = []
        try:
            for record in SeqIO.parse(path, "fasta"):
                s = str(record.seq).upper()
                # 简单的过滤：确保只包含标准氨基酸
                if all(c in AA20 for c in s) and len(s) > 0:
                    seqs.append(s)
            
            data_dict[label] = seqs
            print(f"[加载] {label} ({filename}): {len(seqs)} 条序列")
        except Exception as e:
            print(f"[错误] 读取文件 {filename} 失败: {e}")
        
    return data_dict

def calc_essential_features(dict_data):
    """计算 DataFrame"""
    rows = []
    print("\n正在计算理化性质...")
    for label, seqs in dict_data.items():
        for seq in seqs:
            ana = ProteinAnalysis(seq)
            rows.append({
                "Label": label,
                "Sequence": seq,
                "Hydrophobicity": ana.gravy(),
                "HydrophobicMoment": calculate_hydrophobic_moment(seq)
            })
    return pd.DataFrame(rows)

def plot_hydrophobicity_and_moment_violins(df, label_col="Label", save_dir="."):
    """绘制并保存图片"""
    features = ("Hydrophobicity", "HydrophobicMoment")
    titles = ["Global hydrophobicity", "Global hydrophobic moment"]
    
    # 准备颜色
    labels_present = df[label_col].unique()
    # 如果 label 在 COLOR_MAP 中找不到，默认使用深灰色
    pal = [COLOR_MAP.get(l, '#555555') for l in labels_present]

    # 设置绘图风格
    sns.set(style="white", rc={"axes.facecolor": (0,0,0,0)})
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=False)

    for ax, feat, title in zip(axes, features, titles):
        # 小提琴图
        sns.violinplot(
            data=df, x=label_col, y=feat,
            order=labels_present, palette=pal,
            cut=0, inner=None, linewidth=1.2, width=0.8, ax=ax
        )
        
        # 自定义须线 (Range) 和 均值线 (Mean)
        xticks = ax.get_xticks()
        for x, lab in zip(xticks, labels_present):
            vals = df.loc[df[label_col] == lab, feat].dropna().values
            if vals.size == 0: continue
            
            # Range line (Min to Max)
            ax.vlines(x, np.min(vals), np.max(vals), color="black", lw=1.5, zorder=3)
            # Mean line
            mean_val = np.mean(vals)
            ax.hlines(mean_val, x - 0.15, x + 0.15, color="black", lw=2.0, ls="--", zorder=4)

        ax.set_ylabel(title, fontsize=12)
        ax.set_xlabel("")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    
    # --- 保存图片 ---
    png_path = os.path.join(save_dir, "analysis_violin_plot.png")
    pdf_path = os.path.join(save_dir, "analysis_violin_plot.pdf")
    
    print(f"\n[保存] 图片已保存至:\n  - {png_path}\n  - {pdf_path}")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    
    # 显示图片
    try:
        # 检查是否在交互式环境中，避免服务器无头模式报错
        if sys.stdout.isatty():
             plt.show()
    except Exception:
        pass

def main():
    # 1. 设置参数解析
    parser = argparse.ArgumentParser(description="AMP Physicochemical Analysis Script")
    parser.add_argument(
        'input_files', 
        nargs='*', 
        help='Additional FASTA files to analyze (space separated)'
    )
    parser.add_argument(
        '--outdir', 
        default=DEFAULT_OUTPUT_DIR, 
        help=f'Directory to save results (default: {DEFAULT_OUTPUT_DIR})'
    )
    
    args = parser.parse_args()

    # 2. 准备输出目录
    output_dir = args.outdir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建输出目录: {output_dir}")

    # 3. 构建目标文件列表 (参考文件 + 命令行参数文件)
    target_files = []
    
    # A. 添加硬编码的参考文件
    if os.path.exists(REF_FILE):
        target_files.append(REF_FILE)
        print(f"[信息] 已添加参考文件: {REF_FILE}")
    else:
        print(f"[警告] 硬编码的参考文件未找到: {REF_FILE}")

    # B. 添加命令行传入的文件
    for f in args.input_files:
        if os.path.exists(f):
            # 简单的去重检查
            if f != REF_FILE: 
                target_files.append(f)
        else:
            print(f"[警告] 参数指定的文件不存在: {f}")

    # C. 如果没有文件（既没参考也没参数），且不是帮助模式，则生成模拟数据
    if not target_files:
        print("\n[注意] 未找到任何有效输入文件。")
        print("正在生成模拟数据用于演示...\n")
        
        from Bio.SeqRecord import SeqRecord
        from Bio.Seq import Seq
        # 生成两个模拟文件
        mock_files = ['generated.fasta', 'positive.fasta']
        for fname in mock_files:
            records = []
            for i in range(100):
                s = "".join(np.random.choice(list("ACDEFGHIKLMNPQRSTVWY"), 20))
                records.append(SeqRecord(Seq(s), id=f"{fname}_{i}", description=""))
            
            SeqIO.write(records, fname, "fasta")
            target_files.append(fname)
            print(f"  [模拟] 已生成: {fname}")

    # 4. 加载数据
    dict_data = load_fasta_files(target_files)
    if not dict_data:
        print("错误: 没有有效数据可处理。")
        return

    # 5. 计算特征
    df = calc_essential_features(dict_data)

    # 6. 保存 CSV 数据
    csv_path = os.path.join(output_dir, "analysis_features.csv")
    df.to_csv(csv_path, index=False)
    print(f"[保存] 特征数据表已保存至: {csv_path}")

    # 7. 绘图并保存
    plot_hydrophobicity_and_moment_violins(df, save_dir=output_dir)

if __name__ == '__main__':
    main()