import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from scipy.stats import gaussian_kde

def analyze_charge_heatmap_style(fasta_file, target_pH=7.4):
    charges = []
    
    print(f"正在分析文件: {fasta_file} ...")
    
    try:
        for record in SeqIO.parse(fasta_file, "fasta"):
            seq_str = str(record.seq).upper().replace("*", "").replace("X", "")
            if not seq_str: continue
            try:
                analyser = ProteinAnalysis(seq_str)
                charge = analyser.charge_at_pH(target_pH)
                charges.append(charge)
            except ValueError:
                continue
    except FileNotFoundError:
        print(f"找不到文件: {fasta_file}")
        return

    if not charges:
        print("无有效数据。")
        return

    charges_np = np.array(charges)
    
    # 统计区间 [+2, +7]
    in_range_mask = (charges_np >= 2.0) & (charges_np <= 7.0)
    count_in_range = np.sum(in_range_mask)
    percentage = (count_in_range / len(charges_np)) * 100
    
    print(f"统计结果: [+2.0, +7.0] 区间占比: {percentage:.2f}%")

    # 绘图
    plot_density_heatmap(charges_np, target_pH, percentage)

def plot_density_heatmap(data, ph, range_pct):
    # 创建画布
    fig, ax = plt.subplots(figsize=(12, 7)) #稍微增加一点高度
    
    # --- A. 计算核密度 (KDE) ---
    if len(data) < 2 or np.std(data) == 0:
        print("数据不足，无法绘图")
        return

    kde = gaussian_kde(data)
    x_grid = np.linspace(data.min() - 3, data.max() + 3, 1000)
    y_grid = kde(x_grid)
    
    # --- B. 绘制渐变填充 ---
    ax.plot(x_grid, y_grid, color='white', linewidth=1, alpha=0.5)
    
    im = ax.imshow(np.vstack([y_grid, y_grid]), cmap='turbo', aspect='auto', 
                   extent=[x_grid.min(), x_grid.max(), 0, y_grid.max()],
                   origin='lower', alpha=0.9)

    poly_verts = [(x_grid[0], 0)] + list(zip(x_grid, y_grid)) + [(x_grid[-1], 0)]
    poly = Polygon(poly_verts, facecolor='none')
    ax.add_patch(poly)
    im.set_clip_path(poly)

    # --- C. 标注与修饰 ---
    ax.axvline(2, color='white', linestyle=':', linewidth=1.5, alpha=0.8)
    ax.axvline(7, color='white', linestyle=':', linewidth=1.5, alpha=0.8)
    
    mid_idx = (np.abs(x_grid - 4.5)).argmin()
    text_height = y_grid[mid_idx]
    
    ax.text(4.5, text_height * 0.5, f"Range [+2, +7]\n{range_pct:.2f}%", 
            color='white', ha='center', va='center', fontweight='bold',
            bbox=dict(facecolor='black', alpha=0.3, edgecolor='none', boxstyle='round'))

    mean_val = np.mean(data)
    ax.axvline(mean_val, color='cyan', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
    ax.axvline(0, color='white', linestyle='-', linewidth=1, alpha=0.5, label='Neutral (0)')

    # 设置样式
    ax.set_facecolor('#1e1e1e')
    fig.patch.set_facecolor('#1e1e1e')
    
    ax.set_title(f'Protein Net Charge Density (pH {ph})', color='white', fontsize=16, pad=20)
    ax.set_xlabel('Net Charge', color='white', fontsize=12)
    ax.set_ylabel('Probability Density', color='white', fontsize=12)
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    
    for spine in ax.spines.values():
        spine.set_visible(False)
        
    ax.legend(facecolor='#333333', labelcolor='white', loc='upper right')

    # --- D. 底部布局修复 (核心修改) ---
    
    # 1. 调整主图布局，给底部留出 20% 的空间 (bottom=0.2)
    # rect = [left, bottom, right, top]
    plt.tight_layout(rect=[0, 0.2, 1, 1]) 
    
    # 2. 手动添加底部热力条坐标轴
    # [left, bottom, width, height] (坐标基于画布 0.0-1.0)
    # 将 bottom 设为 0.08，给下方的文字留出足够的空隙
    ax_divider = plt.axes([0.125, 0.08, 0.775, 0.03]) 
    
    cb = plt.colorbar(im, cax=ax_divider, orientation='horizontal')
    cb.set_label('Density Intensity', color='white', fontsize=10, labelpad=5) # labelpad 增加文字距离
    cb.outline.set_visible(False)
    cb.ax.xaxis.set_tick_params(color='white', labelcolor='white')
    
    out_file = 'charge_heatmap_style_fixed.png'
    plt.savefig(out_file, dpi=300, facecolor='#1e1e1e')
    print(f"图表已保存为: {out_file}")
    plt.show()

if __name__ == "__main__":
    input_file = "test.fasta"
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    analyze_charge_heatmap_style(input_file)