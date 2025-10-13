from __future__ import annotations
#!/usr/bin/env python3
"""
示例：输入多个目录路径（每个目录下每条数据是一个 .jsonl 文件），按目录聚合并绘图。

python Runtime/scripts/plot_runtime_correlation.py Runtime/outputs/outputs-qwen3-0.6B/train_temp_0_no_think Runtime/outputs/outputs-qwen3-1.7B/train_temp_0_no_think Runtime/outputs/outputs-qwen3-4B/train_temp_0_no_think Runtime/outputs/outputs-qwen3-8B/train_temp_0_no_think Runtime/outputs/outputs-qwen3-14B/train_temp_0_no_think --out Runtime/outputs  --anno temp_0_no_think

也兼容直接传入 .jsonl 文件路径（将被视为单独的一组）。
默认 x 使用 index，y 将在 duration_seconds/runtime 等字段中自动检测，亦可通过参数指定。
"""
import json
from pathlib import Path
from typing import List, Tuple, Dict
import argparse
import numpy as np
import matplotlib.pyplot as plt

# ----------------- 内置配置（按需修改） -----------------

SCATTER_ALPHA: float = 0.4
SCATTER_SIZE: float = 8

LINE_WIDTH: float = 1.2
LINE_ALPHA: float = 0.8

# ------------------------------------------------------

def read_jsonl(path: str | Path) -> List[dict]:
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

def pick_y_field(records: List[dict], preferred: str | None = None) -> str | None:
    """在记录中选择 y 字段。
    优先使用 preferred（若存在），否则尝试以下候选：
    duration_seconds, runtime, runtime_seconds, elapsed, time
    若均不存在返回 None。
    """
    if not records:
        return preferred
    if preferred and any((preferred in r) and (r.get(preferred) is not None) for r in records):
        return preferred
    candidates = ["duration_seconds", "runtime", "runtime_seconds", "elapsed", "time"]
    for c in candidates:
        if any((c in r) and (r.get(c) is not None) for r in records):
            return c
    return preferred

def extract_xy_field(records: List[dict], x_field: str, y_field: str) -> Tuple[np.ndarray, np.ndarray]:
    xs: List[float] = []
    ys: List[float] = []
    for r in records:
        x = r.get(x_field)
        y = r.get(y_field)
        if x is None or y is None:
            continue
        try:
            x_val = float(x)
            y_val = float(y)
        except Exception:
            continue
        if not np.isfinite(x_val) or not np.isfinite(y_val):
            continue
        xs.append(x_val)
        ys.append(y_val)
    return np.asarray(xs), np.asarray(ys)

def extract_xy_dict(
    records: List[dict],
    x_field: str,
    y_field: str,
    aggregate: str = "last",
) -> Dict[float, float]:
    """提取并按 x 聚合，返回 {x: y} 字典。

    - 过滤 None、无法转为数值、非有限值。
    - 对相同 x 的多条记录，根据 aggregate 聚合 y：
      - mean: 取均值（默认）
      - median: 取中位数
      - first: 第一条
      - last: 最后一条
    """
    from collections import defaultdict

    buckets: Dict[float, List[float]] = defaultdict(list)
    for r in records:
        x = r.get(x_field)
        y = r.get(y_field)
        if x is None or y is None:
            continue
        try:
            x_val = float(x)
            y_val = float(y)
        except Exception:
            continue
        if not np.isfinite(x_val) or not np.isfinite(y_val):
            continue
        buckets[x_val].append(y_val)

    out: Dict[float, float] = {}
    if aggregate not in {"mean", "median", "first", "last"}:
        aggregate = "mean"
    for x_val, ys in buckets.items():
        if not ys:
            continue
        if aggregate == "mean":
            out[x_val] = float(np.mean(ys))
        elif aggregate == "median":
            out[x_val] = float(np.median(ys))
        elif aggregate == "first":
            out[x_val] = float(ys[0])
        elif aggregate == "last":
            out[x_val] = float(ys[-1])
    return out

def plot_y_vs_y(yx: np.ndarray, yy: np.ndarray, label_x: str, label_y: str, out_path: Path, title: str | None = None) -> None:
    """绘制 yx 为横轴、yy 为纵轴的散点图，并添加线性拟合与 y=x 参考线。"""
    if yx.size == 0 or yy.size == 0:
        return

    # 线性拟合：yy ≈ b*yx + a
    b, a = np.polyfit(yx, yy, deg=1)
    yy_pred = b * yx + a
    ss_res = float(np.sum((yy - yy_pred) ** 2))
    ss_tot = float(np.sum((yy - np.mean(yy)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # 拟合线
    x_min, x_max = float(np.min(yx)), float(np.max(yx))
    x_line = np.linspace(x_min, x_max, 200)
    y_line = b * x_line + a

    # y=x 参考线范围
    diag_min = min(x_min, float(np.min(yy)))
    diag_max = max(x_max, float(np.max(yy)))
    diag = np.linspace(diag_min, diag_max, 200)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.scatter(yx, yy, s=20, alpha=0.5, c="#56d022", label="paired points")
    plt.plot(x_line, y_line, color="#93f357", linewidth=2, label=f"fit: {label_y} ~ {label_x} (R²={r2:.3f})")
    plt.plot(diag, diag, color="#45b2fa", linestyle="--", linewidth=1.5, alpha=0.8, label="y = x")
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    if title:
        plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_y_vs_y_filtered(yx: np.ndarray, yy: np.ndarray, label_x: str, label_y: str, out_path: Path, title: str | None = None) -> None:
    """绘制 yx 为横轴、yy 为纵轴的散点图，并添加线性拟合与 y=x 参考线。"""
    if yx.size == 0 or yy.size == 0:
        return

    # 线性拟合：yy ≈ b*yx + a
    filtered_yx, filtered_yy, _ = filter_by_mean_multiplier(yx, yy, multiplier=6.0, drop_non_finite=True)
    b, a = np.polyfit(filtered_yx, filtered_yy, deg=1)
    yy_pred = b * filtered_yx + a
    ss_res = float(np.sum((filtered_yy - yy_pred) ** 2))
    ss_tot = float(np.sum((filtered_yy - np.mean(filtered_yy)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # 拟合线
    x_min, x_max = float(np.min(filtered_yx)), float(np.max(filtered_yx))
    x_line = np.linspace(x_min, x_max, 200)
    y_line = b * x_line + a

    # y=x 参考线范围
    diag_min = min(x_min, float(np.min(filtered_yy)))
    diag_max = max(x_max, float(np.max(filtered_yy)))
    diag = np.linspace(diag_min, diag_max, 200)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.scatter(filtered_yx, filtered_yy, s=20, alpha=0.5, c="#56d022", label="paired points")
    plt.plot(x_line, y_line, color="#93f357", linewidth=2, label=f"fit: {label_y} ~ {label_x} (R²={r2:.3f})")
    plt.plot(diag, diag, color="#45b2fa", linestyle="--", linewidth=1.5, alpha=0.8, label="y = x")
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    if title:
        plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def filter_by_mean_multiplier(
    yx: np.ndarray,
    yy: np.ndarray,
    multiplier: float = 6.0,
    drop_non_finite: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """根据均值阈值过滤点：删除 yx 或 yy 中超过其各自均值 `multiplier` 倍的样本。

    参数:
    - yx, yy: 成对的一维数组，长度必须一致。
    - multiplier: 阈值倍数，默认 6.0。
    - drop_non_finite: 是否先移除非有限值(Inf/NaN)，默认 True。

    返回:
    - yx_f, yy_f: 过滤后的数组副本（保持对应关系）。
    - mask: 布尔掩码，表示原始样本中哪些被保留。

    说明:
    - 若均值<=0(极端情况)，则仅根据非有限值过滤，不做倍数过滤。
    - 这是一个简单的“大于均值倍数”型粗略异常值剔除规则。
    """
    if yx.shape != yy.shape:
        raise ValueError("yx 和 yy 的形状必须一致")

    yx = np.asarray(yx, dtype=float)
    yy = np.asarray(yy, dtype=float)

    base_mask = np.ones_like(yx, dtype=bool)
    if drop_non_finite:
        base_mask &= np.isfinite(yx) & np.isfinite(yy)

    if not np.any(base_mask):
        return yx[base_mask], yy[base_mask], base_mask

    mx = float(np.mean(yx[base_mask]))
    my = float(np.mean(yy[base_mask]))

    if mx > 0 and my > 0 and multiplier > 0:
        thr_x = mx * multiplier
        thr_y = my * multiplier
        range_mask = (yx <= thr_x) & (yy <= thr_y)
        mask = base_mask & range_mask
    else:
        # 均值非正或 multiplier 无效时，仅返回去除非有限值后的数据
        mask = base_mask

    return yx[mask], yy[mask], mask


def plot_runtime_correlation(
    data,
    out_path: Path,
    labels_override: List[str] | None = None,
    anno: str = "temp_0_no_think",
    x_field: str = "index",
    y_field: str = "runtime",
    aggregate: str = "last",
) -> None:
    """
    data: List[Tuple[str, List[dict]]]  每个元素为 (组名, 该组的记录列表)
    out_path: 输出目录
    labels_override: 若提供，优先使用该标签列表（按输入顺序对齐）
    """
    data_groups: List[Tuple[str, Dict[float, float]]] = []

    default_colors = ["#fd4444", "#54ad4a", "#5fb2ee", "#ff9d4e", "#c055c7", "#d8eb49", "#23ecce"]
    default_markers = ["o", "s", "^", "v", "D", "x", "+"]
    default_size_labels = ["0.6B","1.7B","4B","8B","14B"]

    # 聚合出每组的 {x: y}
    for name, recs in data:
        y_resolved = pick_y_field(recs, preferred=y_field)
        if y_resolved is None:
            print(f"[warn] 组 {name} 未找到可用 y 字段，跳过该组。")
            data_groups.append((name, {}))
            continue
        dict_xy = extract_xy_dict(recs, x_field, y_resolved, aggregate=aggregate)
        print(f"[info] 组 {name}: 使用字段 x='{x_field}', y='{y_resolved}', 有效点数={len(dict_xy)}")
        data_groups.append((name, dict_xy))

    ########################################
    # one model vs next model (e.g. 0.6B vs 1.7B, 1.7B vs 4B, etc.)
    ########################################
    # 解析标签：优先外部覆盖，其次默认型号标签，最后使用组名
    resolved_labels: List[str] = []
    for i, (name, _dict) in enumerate(data_groups):
        if labels_override and i < len(labels_override):
            resolved_labels.append(labels_override[i])
        elif i < len(default_size_labels):
            resolved_labels.append(default_size_labels[i])
        else:
            resolved_labels.append(Path(name).name)

    for i, (name, dict1) in enumerate(data_groups):
        # 只与下一组配对
        if i >= len(data_groups) - 1:
            break
        dict2 = data_groups[i + 1][1]
        if not dict1 or not dict2:
            continue

        # 取相同的 x，并按 x 升序组织 y1、y2 列表
        common_x = sorted(set(dict1.keys()) & set(dict2.keys()))
        if not common_x:
            continue
        y1_list = np.asarray([dict1[xk] for xk in common_x], dtype=float)
        y2_list = np.asarray([dict2[xk] for xk in common_x], dtype=float)

        # 安全文件名片段
        def _slug(s: str) -> str:
            return (
                str(s)
                .replace("/", "_")
                .replace("\\", "_")
                .replace(" ", "_")
            )

        plot_y_vs_y_filtered(
            y1_list,
            y2_list,
            label_x=resolved_labels[i] + " runtime(s)",
            label_y=resolved_labels[i + 1] + " runtime(s)",
            out_path=out_path / f"filtered_{anno}_runtime_corr_{_slug(resolved_labels[i])}_vs_{_slug(resolved_labels[i+1])}.png",
            title=f"(filtered) Runtime correlation: {resolved_labels[i]} vs {resolved_labels[i+1]}",
        )

    ########################################
    # 0.6B vs all other models (e.g. 0.6B vs 1.7B, 0.6B vs 4B, etc.)
    ########################################

    for i, (name, dict2) in enumerate(data_groups):
        # 与第 0 组(基准：0.6B)配对，跳过 i==0 以外的组
        if i == 0:
            continue
        dict1 = data_groups[0][1]
        if not dict1 or not dict2:
            continue

        # 取相同的 x，并按 x 升序组织 y1、y2 列表
        common_x = sorted(set(dict1.keys()) & set(dict2.keys()))
        if not common_x:
            print(f"[warn] 组 {name} 与基准组无公共 index，跳过。")
            continue
        y1_list = np.asarray([dict1[xk] for xk in common_x], dtype=float)
        y2_list = np.asarray([dict2[xk] for xk in common_x], dtype=float)

        # without outliers
        # 生成更友好的文件名片段
        def _slug(s: str) -> str:
            return (
                str(s)
                .replace("/", "_")
                .replace("\\", "_")
                .replace(" ", "_")
            )

        label_i = resolved_labels[i] if i < len(resolved_labels) else f"group_{i}"
        plot_y_vs_y_filtered(
            y1_list,
            y2_list,
            label_x="0.6B runtime(s)",
            label_y=label_i + " runtime(s)",
            out_path=out_path / f"filtered_{anno}_runtime_corr_0.6B_vs_{_slug(label_i)}.png",
            title=f"(filtered) Runtime correlation: 0.6B vs {label_i}",
        )


def main():
    # 读取并提取多组数据（输入支持目录或文件）
    parser = argparse.ArgumentParser(description="从多个目录/文件收集 JSONL 记录并绘制运行时相关性")
    parser.add_argument("inputs", nargs="+", type=Path, help="一个或多个路径：目录或 .jsonl 文件")
    parser.add_argument("--out", type=Path, default=Path("Runtime/outputs"), help="输出图片目录")
    parser.add_argument("--anno", type=str, default="temp_0_no_think", help="输入参数标记，用于输出文件名")
    args = parser.parse_args()

    # 聚合输入为多组记录
    groups: List[Tuple[str, List[dict]]] = []
    for p in args.inputs:
        if p.is_dir():
            recs = read_jsonl_from_dir(p)
            if not recs:
                print(f"[warn] 目录为空或无匹配文件：{p}")
                continue
            groups.append((str(p), recs))
        elif p.is_file():
            recs = read_jsonl(p)
            if not recs:
                print(f"[warn] 文件无有效记录：{p}")
            groups.append((str(p), recs))
        else:
            print(f"[warn] 路径不存在：{p}")

    if not groups:
        print("未收集到任何数据，退出。")
        return

    # 执行绘图
    out_dir = Path(args.out)
    plot_runtime_correlation(
        groups,
        out_dir,
        labels_override=["0.6B","1.7B","4B","8B","14B"],
        anno=args.anno,
        x_field="index",
        y_field="runtime",
    )



if __name__ == "__main__":
    main()

