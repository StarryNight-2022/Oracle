# 该文件负责线性拟合指定模型的runtime与length_of_output_token_ids
# 将所有点绘制为散点，接着绘制一个拟合直线，显示直线方程与R^2误差

import os
import json
from typing import List, Union
from pathlib import Path
import argparse

def read_jsonl(path: Union[str,Path]) -> List[dict]:
    p = Path(path)
    recs: List[dict] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except Exception:
                # 跳过异常行
                continue
    return recs


def read_jsonl_from_dir(dir_path: Path, pattern: str = "*.jsonl", recursive: bool = False) -> List[dict]:
    files: List[Path]
    if recursive:
        files = [p for p in dir_path.rglob(pattern) if p.is_file()]
    else:
        files = [p for p in dir_path.glob(pattern) if p.is_file()]
    files = sorted(files, key=lambda x: str(x))
    out: List[dict] = []
    for fp in files:
        out.extend(read_jsonl(fp))
    return out


if __name__ == "__main__":
    llm_model = "Qwen3-0.6B-no-thinking"
    parser = argparse.ArgumentParser(description="从目录提取 x=输出tokens, y=运行时，并进行线性拟合")
    parser.add_argument("--inputs", type=Path, help="一个或多个路径：目录或 .jsonl 文件")
    parser.add_argument("--out", type=Path, default=Path("Runtime/outputs/figures"), help="输出图片目录")
    parser.add_argument("--remove-x-outliers", type=float, default=None,
                        help="(可选) 若设置为 0-1 之间的浮点数 p，则移除 x 大于 x 的第 p 分位数 的点。例: 0.99 会移除 x 最大的 1% 点。")
    args = parser.parse_args()

    input_dir = Path(args.inputs)
    print(f"Reading data from {input_dir}...")

    data = read_jsonl_from_dir(input_dir)
    
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 进行线性拟合
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    # 原始数组（未 reshape），便于按 x 进行过滤
    x_all = np.array([rec["length_of_output_token_ids"] for rec in data])
    y_all = np.array([rec["runtime"] for rec in data])

    # 可选：根据 x 的上分位数移除异常值（仅当用户通过 --remove-x-outliers 提供值时启用）
    # 默认 removed_count 为 0，便于后续在标题中显示
    removed_count = 0
    if args.remove_x_outliers is not None:
        p = float(args.remove_x_outliers)
        if not (0.0 < p < 1.0):
            raise ValueError("--remove-x-outliers must be a float between 0 and 1 (exclusive).")
        thresh = np.quantile(x_all, p)
        mask = x_all <= thresh
        removed_count = int(np.count_nonzero(~mask))
        print(f"remove-x-outliers: threshold={thresh}, removed {removed_count} / {len(x_all)} points")
        x_filtered = x_all[mask]
        y_filtered = y_all[mask]
    else:
        x_filtered = x_all
        y_filtered = y_all

    # reshape 为 scikit-learn 要求的列向量
    x = x_filtered.reshape(-1, 1)
    y = y_filtered
    model = LinearRegression()
    model.fit(x, y)
    y_pred = model.predict(x)
    r2 = r2_score(y, y_pred)
    slope = model.coef_[0]
    intercept = model.intercept_
    equation = f"y = {slope:.4f}x + {intercept:.4f}\nR² = {r2:.4f}"
    
    # 绘制散点图和拟合直线 (使用 fig/ax，图例放在图外，避免与方程文本重叠)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x, y, color='blue', label='Data Points')
    # 为保证拟合直线按 x 的顺序连线，先对 x 进行排序
    try:
        sort_idx = np.argsort(x.flatten())
        x_sorted = x.flatten()[sort_idx]
        y_pred_sorted = y_pred[sort_idx]
    except Exception:
        x_sorted = x.flatten()
        y_pred_sorted = y_pred
    ax.plot(x_sorted, y_pred_sorted, color='red', label='Fitted Line')
    ax.set_xlabel('Length of Output Token IDs')
    ax.set_ylabel('Runtime (seconds)')
    # 在标题中添加被移除的 outliers 数量（若未启用移除则为 0）
    ax.set_title(f'{llm_model}: Runtime vs Length of Output Token IDs (removed_outliers={removed_count})')

    # 在图内靠上方放置拟合方程文本（axes 坐标），并用半透明背景
    ax.text(0.02, 0.98, equation, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # 将图例放到绘图区域的外侧（右边），避免与文本重叠；为图例预留空间
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)

    # 调整布局以给右侧图例留出空间
    fig.tight_layout(rect=[0, 0, 0.85, 1])

    output_path = out_dir / "runtime_vs_length_of_output_token_ids.png"
    fig.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Figure saved to {output_path}")
    