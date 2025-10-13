from __future__ import annotations
#!/usr/bin/env python3
"""
Plot length_of_prompt_token_ids vs duration_seconds from a JSONL log.

Inputs (JSONL fields per record):
  - index
  - duration_seconds
  - prompt
  - length_of_prompt_token_ids
  - length_of_output_token_ids
  - full_response
  - source_file

Generates:
  - scatter_with_regression.png: Scatter plot with sklearn LinearRegression fit curve
  - mean_duration_by_promptlen.png: Line plot of mean duration per unique prompt length

Default input file: Runtime/GSM8k/outputs-qwen3-8B/train_runtime_log.jsonl
Default output dir: Runtime/GSM8k/outputs-qwen3-8B/plots
"""
"""
python Runtime/scripts/plot_x_vs_duration.py --input Runtime/GSM8k/outputs-qwen3-8B/train_0.2_temp_0.1.jsonl  --out-dir Runtime/GSM8k/outputs-qwen3-8B/plots_0.2_temp_0.1 --degree 3 --type 8B
"""



import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures


DEFAULT_INPUT = Path("Runtime/GSM8k/outputs-qwen3-8B/train_runtime_log.jsonl")
DEFAULT_OUTPUT_DIR = Path("Runtime/GSM8k/outputs-qwen3-8B/plots")


def read_jsonl(path: Path) -> List[dict]:
    recs: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except Exception:
                # skip malformed lines
                continue
    return recs

def extract_xy_field(records: List[dict], x_field: str, y_field: str = "duration_seconds") -> Tuple[np.ndarray, np.ndarray]:
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
        if x_val < 0 or y_val < 0:
            continue
        xs.append(x_val)
        ys.append(y_val)
    return np.asarray(xs), np.asarray(ys)


def plot_simple_scatter(x: np.ndarray, y: np.ndarray, out_dir: Path, filename: str, x_label: str, y_label: str, title: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, s=8, alpha=0.4)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def polynomial_regression_and_plot(
    x: np.ndarray,
    y: np.ndarray,
    degree: int,
    out_dir: Path,
    filename: str,
    x_label: str,
    y_label: str,
    title: str,
) -> tuple[str, Path, float]:
    """Fit polynomial regression y ~ poly(x, degree) and plot curve over scatter.

    Returns (expression_str, out_path, r2_score).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename

    # Prepare polynomial features
    x_col = x.reshape(-1, 1)
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(x_col)

    # Fit linear model on polynomial features
    model = LinearRegression()
    model.fit(X_poly, y)
    r2 = float(model.score(X_poly, y))

    # Build readable expression: y = b0 + b1*x + b2*x^2 + ...
    intercept = float(model.intercept_) # 截距
    coefs = model.coef_.ravel() # 系数 从1次方逐渐增大
    terms: list[str] = [f"{intercept:.6g}"]
    for i, c in enumerate(coefs, start=1):
        sign = " + " if c >= 0 else " - "
        mag = abs(float(c))
        if i == 1:
            term = f"{sign}{mag:.3g} x"
        else:
            term = f"{sign}{mag:.3g} x^{i}"
        terms.append(term)
    expression = "y = " + "".join(terms)

    # Prepare smooth curve for plotting
    x_line = np.linspace(float(x.min()), float(x.max()), 400).reshape(-1, 1)
    X_line_poly = poly.transform(x_line)
    y_line = model.predict(X_line_poly)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, s=8, alpha=0.35, label="data")
    plt.plot(x_line.ravel(), y_line, color="darkorange", linewidth=2, label=f"poly degree {degree} (R²={r2:.4f})")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.annotate(f"{expression}", xy=(0.01, 0.95), xycoords="axes fraction", fontsize=10, ha="left", va="top")
    plt.savefig(out_path, dpi=150)
    plt.close()

    return expression, out_path, r2


def linear_regression_and_plot(
    x: np.ndarray,
    y: np.ndarray,
    out_dir: Path,
    filename: str,
    x_label: str,
    y_label: str,
    title: str,
) -> tuple[str, Path, float]:
    """Fit linear regression y ~ a + b*x, plot scatter and fitted line.

    Returns (expression_str, out_path, r2_score).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename

    X = x.reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, y)
    r2 = float(model.score(X, y))

    intercept = float(model.intercept_)
    slope = float(model.coef_.ravel()[0])
    expression = f"y = {intercept:.6g} + {slope:.6g} x"

    # line for plotting
    x_line = np.linspace(float(x.min()), float(x.max()), 200)
    y_line = model.predict(x_line.reshape(-1, 1))

    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, s=8, alpha=0.4, label="data")
    plt.plot(x_line, y_line, color="crimson", linewidth=2, label=f"Linear fit (R²={r2:.4f})")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.annotate(expression, xy=(0.01, 0.95), xycoords="axes fraction", fontsize=10, ha="left", va="top")
    plt.savefig(out_path, dpi=150)
    plt.close()

    return expression, out_path, r2


def main():
    parser = argparse.ArgumentParser(description="Plot prompt length vs duration from JSONL")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help=f"Input JSONL (default: {DEFAULT_INPUT})")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help=f"Output dir for plots (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--degree", type=int, default=2, help="Polynomial degree for regression on input tokens vs runtime")
    parser.add_argument("--type", type=str, default="0.6B", help="model type: 0.6B/1.7B/4B/8B/14B")
    args = parser.parse_args()

    records = read_jsonl(args.input)

    # 1) prompt tokens vs duration (with regression)
    x1, y1 = extract_xy_field(records, "length_of_prompt_token_ids", "duration_seconds")
    # scatter_prompt = plot_simple_scatter(
    #     x1,
    #     y1,
    #     args.out_dir,
    #     filename=f"Qwen3-{args.type}_input_tokens_vs_runtime.png",
    #     x_label="input tokens",
    #     y_label="runtime(s)",
    #     title=f"Qwen3-{args.type} input tokens vs runtime (scatter)",
    # )
    # Polynomial regression fit and plot
    expr, poly_path, poly_r2 = polynomial_regression_and_plot(
        x1,
        y1,
        degree=args.degree,
        out_dir=args.out_dir,
        filename=f"Qwen3-{args.type}_input_tokens_vs_runtime_poly_deg{args.degree}.png",
        x_label="input tokens",
        y_label="runtime(s)",
        title=f"Qwen3-{args.type} input tokens vs runtime (poly deg {args.degree})",
    )
    print(expr)
    print(f"Poly R^2={poly_r2:.4f} -> {poly_path}")

    # # 2) output tokens vs duration (simple scatter)
    x2, y2 = extract_xy_field(records, "length_of_output_token_ids", "duration_seconds")

    # scatter_output = plot_simple_scatter(
    #     x2,
    #     y2,
    #     args.out_dir,
    #     filename=f"Qwen3-{args.type}_output_tokens_vs_runtime.png",
    #     x_label="output tokens",
    #     y_label="runtime(s)",
    #     title=f"Qwen3-{args.type} output tokens vs runtime (scatter)",
    # )
    # Fit linear regression on output tokens vs runtime and save plot
    lin_expr, lin_path, lin_r2 = linear_regression_and_plot(
        x2,
        y2,
        out_dir=args.out_dir,
        filename=f"Qwen3-{args.type}_output_tokens_vs_runtime_linear.png",
        x_label="output tokens",
        y_label="runtime(s)",
        title=f"Qwen3-{args.type} output tokens vs runtime (linear fit)",
    )
    print(lin_expr)
    print(f"Linear R^2={lin_r2:.4f} -> {lin_path}")

    # # 3) prompt length vs duration (simple scatter)
    # x3, y3 = extract_xy_field(records, "length_of_question", "duration_seconds")
    # scatter_index = plot_simple_scatter(
    #     x3,
    #     y3,
    #     args.out_dir,
    #     filename=f"Qwen3-{args.type}_prompt_length_vs_runtime.png",
    #     x_label="prompt length",
    #     y_label="runtime(s)",
    #     title=f"Qwen3-{args.type} prompt length vs runtime (scatter)",
    # )



if __name__ == "__main__":
    main()
