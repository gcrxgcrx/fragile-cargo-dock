import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ==================== 配置 ====================
BASE = Path("runs/env_001/ablation_score_only_v4")
OUTPUT = "search_trajectory_score_only.png"
TARGET_SCORE = 200.0
MAX_ROUNDS = 10  # 实验设定的最大轮数

# ==================== 数据提取 ====================
def parse_memory(md_file):
    """从 reward_memory.md 中提取每轮的 score 列表"""
    scores = []
    with open(md_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    header_found = False
    for line in lines:
        if line.startswith("|---"):
            header_found = True
            continue
        if not header_found or not line.startswith("|"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 5:
            continue
        try:
            iter_num = int(parts[1])
            score = float(parts[3])   # score 列
        except (ValueError, IndexError):
            continue
        # 插入到对应迭代位置（确保顺序）
        while len(scores) < iter_num:
            scores.append(None)
        scores[iter_num - 1] = score
    return scores

def collect_all_seeds():
    """收集所有 seed 的分数轨迹，并返回填充好的数组和最佳分数"""
    seed_scores = {}
    best_scores = {}
    for seed_dir in sorted(BASE.glob("seed_*")):
        seed = seed_dir.name.split("_")[-1]
        md_file = seed_dir / "memory" / "reward_memory.md"
        if not md_file.exists():
            print(f"Warning: {md_file} not found, skipping seed {seed}")
            continue
        scores = parse_memory(md_file)
        seed_scores[seed] = scores
        if scores:
            best_scores[seed] = max(s for s in scores if s is not None)
        else:
            best_scores[seed] = None
        print(f"Seed {seed}: rounds={len(scores)}, best={best_scores[seed]:.2f}" if best_scores[seed] is not None else f"Seed {seed}: no data")
    return seed_scores, best_scores

# ==================== 绘图 ====================
def plot_trajectories(seed_scores, best_scores):
    seeds = sorted(seed_scores.keys(), key=lambda x: int(x))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(seeds)))

    # 将不等长的轨迹填充到相同长度，以便计算统计量
    max_len = max(len(scores) for scores in seed_scores.values())
    array = np.full((len(seeds), max_len), np.nan)
    for i, seed in enumerate(seeds):
        scores = seed_scores[seed]
        array[i, :len(scores)] = scores

    # 绘图
    fig, ax = plt.subplots(figsize=(10, 6))

    # 绘制每个 seed 的独立轨迹（半透明）
    for i, seed in enumerate(seeds):
        scores = seed_scores[seed]
        x = np.arange(1, len(scores) + 1)
        ax.plot(x, scores, color=colors[i], alpha=0.6, linewidth=1, label=f"Seed {seed} (best={best_scores[seed]:.1f})")

    # 绘制均值 ± 标准差
    mean_curve = np.nanmean(array, axis=0)
    std_curve = np.nanstd(array, axis=0)
    x_mean = np.arange(1, max_len + 1)
    ax.plot(x_mean, mean_curve, color='black', linewidth=2.5, label='Mean')
    ax.fill_between(x_mean, mean_curve - std_curve, mean_curve + std_curve,
                    color='black', alpha=0.15, label='±1 Std')

    # 目标线
    ax.axhline(y=TARGET_SCORE, color='red', linestyle='--', linewidth=1.5, label=f'Target ({TARGET_SCORE})')

    ax.set_xlabel('Iteration', fontsize=13)
    ax.set_ylabel('External Evaluation Reward', fontsize=13)
    ax.set_title('Ablation: Score-Only Feedback — Search Trajectory', fontsize=14)
    ax.legend(loc='lower right', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, max_len)
    plt.tight_layout()
    plt.savefig(OUTPUT, dpi=200)
    plt.show()
    print(f"Figure saved to {OUTPUT}")

if __name__ == "__main__":
    seed_scores, best_scores = collect_all_seeds()
    plot_trajectories(seed_scores, best_scores)