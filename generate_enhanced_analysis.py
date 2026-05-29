from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
MAIN_RESULTS = ROOT / "results" / "contamination_aware_train1000_test300_with_clean"
TRAIN100_SCALE_RESULTS = ROOT / "results" / "train100_model_on_same300test_as_train1000"
TRAIN1000_SCALE_RESULTS = ROOT / "results" / "train1000_model_reeval_same300test"
CONTAM40_RESULTS = ROOT / "results" / "contamination_aware_contam40_test50_existing"
OUTPUT_DIR = ROOT / "results" / "enhanced_analysis"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"

CORE_METHODS = [
    "Unweighted Count Beam",
    "Read-weighted Count Beam",
    "Read-weighted MPNN Beam",
    "Read-weighted MPNN+ReadAux Beam",
]

TRAIN_SIZE_METHODS = [
    "Read-weighted Count Beam",
    "Read-weighted MPNN Beam",
    "Read-weighted MPNN+ReadAux Beam",
]

TRAIN_SIZE_FIGURE_NAMES = {
    "Read-weighted Count Beam": "train_size_read_weighted_count_beam.png",
    "Read-weighted MPNN Beam": "train_size_read_weighted_mpnn_beam.png",
    "Read-weighted MPNN+ReadAux Beam": "train_size_mpnn_readaux_beam.png",
}

PLOT_COLORS = [
    (38, 99, 201),
    (227, 126, 42),
    (49, 153, 96),
    (196, 64, 82),
    (119, 90, 176),
]

PLOT_LABELS = {
    "Unweighted Count Beam": "Unweighted Count",
    "Read-weighted Count Beam": "Read-weighted Count",
    "Read-weighted MPNN Beam": "MPNN",
    "Read-weighted MPNN+ReadAux Beam": "MPNN+ReadAux",
    "Read-weighted MPNN": "MPNN",
    "Read-weighted MPNN+ReadAux": "MPNN+ReadAux",
}


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"缺少结果文件: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def write_csv_rows(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def as_float(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value in {"", None}:
        return default
    return float(value)


def as_int(row: Dict[str, str], key: str, default: int = 0) -> int:
    value = row.get(key, "")
    if value in {"", None}:
        return default
    return int(float(value))


def fmt3(value: float) -> str:
    return f"{value:.3f}"


def fmt1(value: float) -> str:
    return f"{value:.1f}"


def rate_label(rate: float) -> str:
    return f"{rate * 100:.0f}%"


def method_index(rows: Iterable[Dict[str, str]]) -> Dict[Tuple[float, str], Dict[str, str]]:
    indexed: Dict[Tuple[float, str], Dict[str, str]] = {}
    for row in rows:
        indexed[(round(as_float(row, "contamination_rate"), 4), row["method"])] = row
    return indexed


def available_rates(rows: Iterable[Dict[str, str]]) -> List[float]:
    return sorted({round(as_float(row, "contamination_rate"), 4) for row in rows})


def build_main_metrics(summary_rows: Sequence[Dict[str, str]]) -> List[Dict[str, object]]:
    indexed = method_index(summary_rows)
    rows: List[Dict[str, object]] = []
    for rate in available_rates(summary_rows):
        for method in CORE_METHODS:
            row = indexed.get((rate, method))
            if row is None:
                continue
            rows.append(
                {
                    "污染率": rate_label(rate),
                    "方法": method,
                    "样本数": as_int(row, "samples"),
                    "SRR": fmt3(as_float(row, "sequence_recovery_rate")),
                    "BRR": fmt3(as_float(row, "base_recovery_rate")),
                    "AvgED": fmt3(as_float(row, "avg_edit_distance")),
                    "平均节点数": fmt1(as_float(row, "avg_nodes")),
                    "平均边数": fmt1(as_float(row, "avg_edges")),
                }
            )
    return rows


def pivot_main_metric_rows(rows: Sequence[Dict[str, object]], metric: str) -> List[Dict[str, object]]:
    by_rate: Dict[str, Dict[str, object]] = defaultdict(dict)
    for row in rows:
        by_rate[str(row["污染率"])][str(row["方法"])] = row.get(metric, "")

    output: List[Dict[str, object]] = []
    for rate in sorted(by_rate, key=lambda value: float(value.rstrip("%"))):
        item: Dict[str, object] = {"污染率": rate}
        for method in CORE_METHODS:
            item[method] = by_rate[rate].get(method, "")
        output.append(item)
    return output


def build_read_weight_ablation(summary_rows: Sequence[Dict[str, str]]) -> List[Dict[str, object]]:
    indexed = method_index(summary_rows)
    rows: List[Dict[str, object]] = []
    for rate in available_rates(summary_rows):
        unweighted = indexed.get((rate, "Unweighted Count Beam"))
        weighted = indexed.get((rate, "Read-weighted Count Beam"))
        if unweighted is None or weighted is None:
            continue
        unweighted_edges = as_float(unweighted, "avg_edges")
        weighted_edges = as_float(weighted, "avg_edges")
        rows.append(
            {
                "污染率": rate_label(rate),
                "Unweighted Count Beam SRR": fmt3(as_float(unweighted, "sequence_recovery_rate")),
                "Read-weighted Count Beam SRR": fmt3(as_float(weighted, "sequence_recovery_rate")),
                "SRR提升": fmt3(
                    as_float(weighted, "sequence_recovery_rate") - as_float(unweighted, "sequence_recovery_rate")
                ),
                "Unweighted BRR": fmt3(as_float(unweighted, "base_recovery_rate")),
                "Read-weighted BRR": fmt3(as_float(weighted, "base_recovery_rate")),
                "Unweighted AvgED": fmt3(as_float(unweighted, "avg_edit_distance")),
                "Read-weighted AvgED": fmt3(as_float(weighted, "avg_edit_distance")),
                "AvgED变化": fmt3(as_float(weighted, "avg_edit_distance") - as_float(unweighted, "avg_edit_distance")),
                "Unweighted 平均边数": fmt1(unweighted_edges),
                "Read-weighted 平均边数": fmt1(weighted_edges),
                "边数减少比例": fmt3((unweighted_edges - weighted_edges) / max(unweighted_edges, 1e-9)),
            }
        )
    return rows


def build_mpnn_contribution(summary_rows: Sequence[Dict[str, str]]) -> List[Dict[str, object]]:
    indexed = method_index(summary_rows)
    rows: List[Dict[str, object]] = []
    for rate in available_rates(summary_rows):
        count_row = indexed.get((rate, "Read-weighted Count Beam"))
        mpnn_row = indexed.get((rate, "Read-weighted MPNN Beam"))
        aux_row = indexed.get((rate, "Read-weighted MPNN+ReadAux Beam"))
        if count_row is None or mpnn_row is None or aux_row is None:
            continue
        rows.append(
            {
                "污染率": rate_label(rate),
                "Count Beam SRR": fmt3(as_float(count_row, "sequence_recovery_rate")),
                "Count Beam BRR": fmt3(as_float(count_row, "base_recovery_rate")),
                "Count Beam AvgED": fmt3(as_float(count_row, "avg_edit_distance")),
                "MPNN Beam SRR": fmt3(as_float(mpnn_row, "sequence_recovery_rate")),
                "MPNN Beam BRR": fmt3(as_float(mpnn_row, "base_recovery_rate")),
                "MPNN Beam AvgED": fmt3(as_float(mpnn_row, "avg_edit_distance")),
                "MPNN+ReadAux SRR": fmt3(as_float(aux_row, "sequence_recovery_rate")),
                "MPNN+ReadAux BRR": fmt3(as_float(aux_row, "base_recovery_rate")),
                "MPNN+ReadAux AvgED": fmt3(as_float(aux_row, "avg_edit_distance")),
                "MPNN相对Count的SRR提升": fmt3(
                    as_float(mpnn_row, "sequence_recovery_rate") - as_float(count_row, "sequence_recovery_rate")
                ),
                "ReadAux相对MPNN的AvgED变化": fmt3(
                    as_float(aux_row, "avg_edit_distance") - as_float(mpnn_row, "avg_edit_distance")
                ),
            }
        )
    return rows


def build_read_aux_analysis(
    main_read_rows: Sequence[Dict[str, str]], contam40_read_rows: Sequence[Dict[str, str]]
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for source, test_clusters, source_rows in [
        ("1000训练簇/300测试簇", 300, main_read_rows),
        ("40%补充测试/50测试簇", 50, contam40_read_rows),
    ]:
        for row in source_rows:
            if row.get("model") != "Read-weighted MPNN+ReadAux":
                continue
            rate = round(as_float(row, "contamination_rate"), 4)
            if rate == 0:
                continue
            rows.append(
                {
                    "污染率": rate_label(rate),
                    "测试来源": source,
                    "测试簇数": test_clusters,
                    "Precision": fmt3(as_float(row, "read_precision")),
                    "Recall": fmt3(as_float(row, "read_recall")),
                    "Read-F1": fmt3(as_float(row, "read_f1")),
                    "Accuracy": fmt3(as_float(row, "read_accuracy")),
                    "Read loss": fmt3(as_float(row, "read_loss")),
                }
            )
    return sorted(rows, key=lambda item: float(str(item["污染率"]).rstrip("%")))


def build_edge_f1_rows(
    main_edge_rows: Sequence[Dict[str, str]], contam40_edge_rows: Sequence[Dict[str, str]]
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for source, test_clusters, source_rows in [
        ("1000训练簇/300测试簇", 300, main_edge_rows),
        ("40%补充测试/50测试簇", 50, contam40_edge_rows),
    ]:
        for row in source_rows:
            rows.append(
                {
                    "污染率": rate_label(round(as_float(row, "contamination_rate"), 4)),
                    "模型": row["model"],
                    "测试来源": source,
                    "测试簇数": test_clusters,
                    "Precision": fmt3(as_float(row, "precision")),
                    "Recall": fmt3(as_float(row, "recall")),
                    "Edge-F1": fmt3(as_float(row, "edge_f1")),
                    "Loss": fmt3(as_float(row, "loss")),
                }
            )
    return sorted(rows, key=lambda item: (float(str(item["污染率"]).rstrip("%")), str(item["模型"])))


def build_train_size_comparison(
    train100_rows: Sequence[Dict[str, str]], train1000_rows: Sequence[Dict[str, str]]
) -> List[Dict[str, object]]:
    train100_index = method_index(train100_rows)
    train1000_index = method_index(train1000_rows)
    rows: List[Dict[str, object]] = []
    for rate in available_rates(train1000_rows):
        if rate > 0.3:
            continue
        for method in [
            "Read-weighted Count Beam",
            "Read-weighted MPNN Beam",
            "Read-weighted MPNN+ReadAux Beam",
        ]:
            small = train100_index.get((rate, method))
            large = train1000_index.get((rate, method))
            if small is None or large is None:
                continue
            rows.append(
                {
                    "污染率": rate_label(rate),
                    "方法": method,
                    "100训练簇/300测试簇 SRR": fmt3(as_float(small, "sequence_recovery_rate")),
                    "1000训练簇/300测试簇 SRR": fmt3(as_float(large, "sequence_recovery_rate")),
                    "SRR变化": fmt3(
                        as_float(large, "sequence_recovery_rate") - as_float(small, "sequence_recovery_rate")
                    ),
                    "100训练簇/300测试簇 BRR": fmt3(as_float(small, "base_recovery_rate")),
                    "1000训练簇/300测试簇 BRR": fmt3(as_float(large, "base_recovery_rate")),
                    "100训练簇/300测试簇 AvgED": fmt3(as_float(small, "avg_edit_distance")),
                    "1000训练簇/300测试簇 AvgED": fmt3(as_float(large, "avg_edit_distance")),
                    "AvgED变化": fmt3(
                        as_float(large, "avg_edit_distance") - as_float(small, "avg_edit_distance")
                    ),
                    "备注": "两组结果使用同一批300个测试簇。",
                }
            )
    return rows


def build_train_size_tables(rows: Sequence[Dict[str, object]]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    train100_rows: List[Dict[str, object]] = []
    train1000_rows: List[Dict[str, object]] = []
    for row in rows:
        train100_rows.append(
            {
                "训练规模": "100训练簇/300测试簇",
                "污染率": row["污染率"],
                "方法": row["方法"],
                "SRR": row["100训练簇/300测试簇 SRR"],
                "BRR": row["100训练簇/300测试簇 BRR"],
                "AvgED": row["100训练簇/300测试簇 AvgED"],
            }
        )
        train1000_rows.append(
            {
                "训练规模": "1000训练簇/300测试簇",
                "污染率": row["污染率"],
                "方法": row["方法"],
                "SRR": row["1000训练簇/300测试簇 SRR"],
                "BRR": row["1000训练簇/300测试簇 BRR"],
                "AvgED": row["1000训练簇/300测试簇 AvgED"],
            }
        )
    return train100_rows, train1000_rows


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "/System/Library/Fonts/PingFang.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def draw_line_chart(
    path: Path,
    title: str,
    y_label: str,
    series: Dict[str, List[Tuple[float, float]]],
    y_min: float | None = None,
    y_max: float | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1280, 760
    margin_left, margin_right, margin_top, margin_bottom = 110, 330, 90, 105
    plot_left = margin_left
    plot_top = margin_top
    plot_right = width - margin_right
    plot_bottom = height - margin_bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(30)
    label_font = load_font(21)
    tick_font = load_font(18)
    legend_font = load_font(18)

    all_points = [point for values in series.values() for point in values]
    xs = sorted({point[0] for point in all_points})
    ys = [point[1] for point in all_points]
    if y_min is None:
        y_min = min(ys)
    if y_max is None:
        y_max = max(ys)
    if math.isclose(y_min, y_max):
        y_min -= 0.05
        y_max += 0.05
    padding = (y_max - y_min) * 0.08
    y_min = max(0.0, y_min - padding)
    y_max = min(1.0 if max(ys) <= 1.0 else y_max + padding, y_max + padding)

    title_width, _ = text_size(draw, title, title_font)
    draw.text(((width - title_width) / 2, 24), title, fill=(28, 34, 43), font=title_font)
    draw.rectangle((plot_left, plot_top, plot_right, plot_bottom), outline=(185, 192, 203), width=2)

    for tick in range(6):
        ratio = tick / 5
        y = plot_bottom - ratio * (plot_bottom - plot_top)
        value = y_min + ratio * (y_max - y_min)
        draw.line((plot_left, y, plot_right, y), fill=(230, 234, 240), width=1)
        draw.text((20, y - 11), f"{value:.2f}", fill=(82, 91, 105), font=tick_font)

    if len(xs) == 1:
        x_positions = {xs[0]: (plot_left + plot_right) / 2}
    else:
        x_positions = {
            x: plot_left + index * (plot_right - plot_left) / (len(xs) - 1) for index, x in enumerate(xs)
        }
    for x in xs:
        px = x_positions[x]
        draw.line((px, plot_bottom, px, plot_bottom + 8), fill=(95, 104, 118), width=2)
        label = rate_label(x)
        label_width, _ = text_size(draw, label, tick_font)
        draw.text((px - label_width / 2, plot_bottom + 16), label, fill=(62, 72, 86), font=tick_font)

    def map_y(value: float) -> float:
        return plot_bottom - (value - y_min) / max(y_max - y_min, 1e-9) * (plot_bottom - plot_top)

    for index, (name, values) in enumerate(series.items()):
        color = PLOT_COLORS[index % len(PLOT_COLORS)]
        points = [(x_positions[x], map_y(y)) for x, y in values]
        if len(points) > 1:
            draw.line(points, fill=color, width=4, joint="curve")
        for px, py in points:
            draw.ellipse((px - 6, py - 6, px + 6, py + 6), fill=color, outline="white", width=2)

        legend_x = plot_right + 36
        legend_y = plot_top + 18 + index * 42
        draw.line((legend_x, legend_y + 12, legend_x + 32, legend_y + 12), fill=color, width=5)
        draw.ellipse((legend_x + 10, legend_y + 4, legend_x + 24, legend_y + 18), fill=color)
        draw.text((legend_x + 46, legend_y), name, fill=(40, 48, 61), font=legend_font)

    draw.text((plot_left, height - 44), "污染率", fill=(45, 53, 67), font=label_font)
    draw.text((18, plot_top - 36), y_label, fill=(45, 53, 67), font=label_font)
    image.save(path)


def draw_grouped_bar_chart(
    path: Path,
    title: str,
    y_label: str,
    rows: Sequence[Dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1280, 760
    margin_left, margin_right, margin_top, margin_bottom = 120, 260, 90, 110
    plot_left = margin_left
    plot_top = margin_top
    plot_right = width - margin_right
    plot_bottom = height - margin_bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(30)
    label_font = load_font(21)
    tick_font = load_font(18)
    legend_font = load_font(18)
    title_width, _ = text_size(draw, title, title_font)
    draw.text(((width - title_width) / 2, 24), title, fill=(28, 34, 43), font=title_font)

    rates = [str(row["污染率"]) for row in rows]
    unweighted_values = [float(str(row["Unweighted 平均边数"])) for row in rows]
    weighted_values = [float(str(row["Read-weighted 平均边数"])) for row in rows]
    y_max = max(unweighted_values + weighted_values) * 1.12
    draw.rectangle((plot_left, plot_top, plot_right, plot_bottom), outline=(185, 192, 203), width=2)
    for tick in range(6):
        ratio = tick / 5
        y = plot_bottom - ratio * (plot_bottom - plot_top)
        value = ratio * y_max
        draw.line((plot_left, y, plot_right, y), fill=(230, 234, 240), width=1)
        draw.text((28, y - 11), f"{value:.0f}", fill=(82, 91, 105), font=tick_font)

    group_width = (plot_right - plot_left) / max(len(rates), 1)
    bar_width = group_width * 0.28
    colors = [(38, 99, 201), (227, 126, 42)]
    for index, rate in enumerate(rates):
        center = plot_left + group_width * (index + 0.5)
        for offset, value, color in [
            (-bar_width * 0.6, unweighted_values[index], colors[0]),
            (bar_width * 0.6, weighted_values[index], colors[1]),
        ]:
            left = center + offset - bar_width / 2
            right = center + offset + bar_width / 2
            top = plot_bottom - value / y_max * (plot_bottom - plot_top)
            draw.rectangle((left, top, right, plot_bottom), fill=color)
        label_width, _ = text_size(draw, rate, tick_font)
        draw.text((center - label_width / 2, plot_bottom + 16), rate, fill=(62, 72, 86), font=tick_font)

    legend_x = plot_right + 34
    for idx, name in enumerate(["Unweighted graph", "Read-weighted graph"]):
        legend_y = plot_top + 18 + idx * 42
        draw.rectangle((legend_x, legend_y + 3, legend_x + 30, legend_y + 23), fill=colors[idx])
        draw.text((legend_x + 44, legend_y), name, fill=(40, 48, 61), font=legend_font)

    draw.text((plot_left, height - 44), "污染率", fill=(45, 53, 67), font=label_font)
    draw.text((18, plot_top - 36), y_label, fill=(45, 53, 67), font=label_font)
    image.save(path)


def train_size_metric_series(
    rows: Sequence[Dict[str, object]], method: str, metric: str
) -> List[Tuple[float, float]]:
    values = []
    for row in rows:
        if row["方法"] != method:
            continue
        rate = float(str(row["污染率"]).rstrip("%")) / 100.0
        values.append((rate, float(row[metric])))
    return sorted(values)


def draw_train_size_method_chart(
    path: Path,
    method: str,
    train100_rows: Sequence[Dict[str, object]],
    train1000_rows: Sequence[Dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1420, 760
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(30)
    label_font = load_font(20)
    tick_font = load_font(17)
    legend_font = load_font(18)

    title = f"{PLOT_LABELS.get(method, method)} 训练规模对比"
    title_width, _ = text_size(draw, title, title_font)
    draw.text(((width - title_width) / 2, 24), title, fill=(28, 34, 43), font=title_font)

    legend_items = [
        ("100训练簇/300测试簇", (38, 99, 201)),
        ("1000训练簇/300测试簇", (227, 126, 42)),
    ]
    legend_x = 455
    legend_y = 72
    for index, (label, color) in enumerate(legend_items):
        x = legend_x + index * 260
        draw.line((x, legend_y + 12, x + 32, legend_y + 12), fill=color, width=5)
        draw.ellipse((x + 10, legend_y + 4, x + 24, legend_y + 18), fill=color)
        draw.text((x + 46, legend_y), label, fill=(40, 48, 61), font=legend_font)

    panels = [
        {
            "title": "SRR",
            "metric": "SRR",
            "rect": (95, 135, 660, 625),
            "y_min": 0.0,
            "y_max": 1.0,
        },
        {
            "title": "AvgED",
            "metric": "AvgED",
            "rect": (805, 135, 1370, 625),
            "y_min": 0.0,
            "y_max": None,
        },
    ]

    for panel in panels:
        left, top, right, bottom = panel["rect"]
        metric = str(panel["metric"])
        series_100 = train_size_metric_series(train100_rows, method, metric)
        series_1000 = train_size_metric_series(train1000_rows, method, metric)
        all_points = series_100 + series_1000
        xs = sorted({point[0] for point in all_points})
        ys = [point[1] for point in all_points]
        y_min = float(panel["y_min"])
        if panel["y_max"] is None:
            y_max = max(ys) * 1.15 if ys else 1.0
            if y_max <= 0:
                y_max = 1.0
        else:
            y_max = float(panel["y_max"])

        draw.rectangle((left, top, right, bottom), outline=(185, 192, 203), width=2)
        panel_title = str(panel["title"])
        panel_title_width, _ = text_size(draw, panel_title, label_font)
        draw.text(((left + right - panel_title_width) / 2, top - 38), panel_title, fill=(45, 53, 67), font=label_font)

        for tick in range(6):
            ratio = tick / 5
            y = bottom - ratio * (bottom - top)
            value = y_min + ratio * (y_max - y_min)
            draw.line((left, y, right, y), fill=(230, 234, 240), width=1)
            label = f"{value:.2f}" if metric == "SRR" else f"{value:.1f}"
            draw.text((left - 70, y - 10), label, fill=(82, 91, 105), font=tick_font)

        if len(xs) == 1:
            x_positions = {xs[0]: (left + right) / 2}
        else:
            x_positions = {x: left + index * (right - left) / (len(xs) - 1) for index, x in enumerate(xs)}

        for x in xs:
            px = x_positions[x]
            draw.line((px, bottom, px, bottom + 8), fill=(95, 104, 118), width=2)
            label = rate_label(x)
            label_width, _ = text_size(draw, label, tick_font)
            draw.text((px - label_width / 2, bottom + 16), label, fill=(62, 72, 86), font=tick_font)

        def map_y(value: float) -> float:
            return bottom - (value - y_min) / max(y_max - y_min, 1e-9) * (bottom - top)

        for values, color in [(series_100, legend_items[0][1]), (series_1000, legend_items[1][1])]:
            points = [(x_positions[x], map_y(y)) for x, y in values]
            if len(points) > 1:
                draw.line(points, fill=color, width=4, joint="curve")
            for px, py in points:
                draw.ellipse((px - 6, py - 6, px + 6, py + 6), fill=color, outline="white", width=2)

        x_label_width, _ = text_size(draw, "污染率", label_font)
        draw.text(((left + right - x_label_width) / 2, height - 50), "污染率", fill=(45, 53, 67), font=label_font)

    image.save(path)


def series_from_summary(
    rows: Sequence[Dict[str, str]], methods: Sequence[str], metric: str
) -> Dict[str, List[Tuple[float, float]]]:
    indexed = method_index(rows)
    result: Dict[str, List[Tuple[float, float]]] = {}
    for method in methods:
        values = []
        for rate in available_rates(rows):
            row = indexed.get((rate, method))
            if row is not None:
                values.append((rate, as_float(row, metric)))
        result[PLOT_LABELS.get(method, method)] = values
    return result


def series_from_metric_rows(rows: Sequence[Dict[str, str]], metric: str) -> Dict[str, List[Tuple[float, float]]]:
    grouped: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for row in rows:
        grouped[PLOT_LABELS.get(row["model"], row["model"])].append(
            (round(as_float(row, "contamination_rate"), 4), as_float(row, metric))
        )
    return {name: sorted(values) for name, values in grouped.items()}


def edit_operations(reference: str, prediction: str) -> List[Dict[str, object]]:
    n, m = len(reference), len(prediction)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if reference[i - 1] == prediction[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )

    ops: List[Dict[str, object]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and reference[i - 1] == prediction[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append({"type": "sub", "pos": i, "ref": reference[i - 1], "pred": prediction[j - 1]})
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append({"type": "del", "pos": i, "ref": reference[i - 1], "pred": ""})
            i -= 1
        else:
            ops.append({"type": "ins", "pos": i + 1, "ref": "", "pred": prediction[j - 1]})
            j -= 1
    return list(reversed(ops))


def merge_positions(positions: Sequence[int]) -> str:
    if not positions:
        return "无"
    sorted_positions = sorted(set(positions))
    spans = []
    start = previous = sorted_positions[0]
    for pos in sorted_positions[1:]:
        if pos == previous + 1:
            previous = pos
            continue
        spans.append((start, previous))
        start = previous = pos
    spans.append((start, previous))
    labels = [str(a) if a == b else f"{a}-{b}" for a, b in spans[:4]]
    if len(spans) > 4:
        labels.append("...")
    return "、".join(labels)


def summarize_alignment(reference: str, prediction: str) -> Dict[str, str]:
    ops = edit_operations(reference, prediction)
    sub_count = sum(1 for op in ops if op["type"] == "sub")
    del_count = sum(1 for op in ops if op["type"] == "del")
    ins_count = sum(1 for op in ops if op["type"] == "ins")
    positions = [int(op["pos"]) for op in ops]
    if not ops:
        location = "预测序列与参考序列完全一致。"
        risk = "未观察到 bridge/bypass 风险。"
    else:
        location = (
            f"错误主要位于参考序列第 {merge_positions(positions)} 位附近；"
            f"替换 {sub_count} 个、删除 {del_count} 个、插入 {ins_count} 个。"
        )
        if del_count >= max(sub_count, ins_count) + 2:
            risk = "错误以删除为主，可能存在绕过真实片段的 bypass 类风险。"
        elif ins_count >= max(sub_count, del_count) + 2:
            risk = "错误以插入为主，可能存在引入额外片段的 bridge 类风险。"
        elif sub_count >= max(del_count, ins_count) + 2:
            risk = "错误以局部替换为主，bridge/bypass 风险不是主要表现。"
        else:
            risk = "错误类型混合，可能同时包含局部替换和 bridge/bypass 路径干扰。"
    return {
        "错误集中位置": location,
        "bridge/bypass判断": risk,
    }


def contaminant_support_text(rate: float, avg_read_weight: float) -> str:
    if rate <= 0:
        return "该案例没有人工加入污染 reads。"
    if avg_read_weight < 0.74:
        return "平均 read_weight 较低，说明簇内 reads 可靠性分化明显，错误路径存在被污染 reads 支持的风险。"
    if avg_read_weight < 0.80:
        return "平均 read_weight 中等偏低，污染 reads 可能参与了错误路径竞争。"
    return "平均 read_weight 较高，污染 reads 影响相对有限，局部图结构错误更值得关注。"


def choose_case_studies(detail_rows: Sequence[Dict[str, str]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[float, str], Dict[str, Dict[str, str]]] = defaultdict(dict)
    for row in detail_rows:
        key = (round(as_float(row, "contamination_rate"), 4), row["cluster_id"])
        grouped[key][row["method"]] = row

    selected: List[Tuple[str, str, Dict[str, Dict[str, str]], str, str]] = []
    used_keys: set[Tuple[float, str]] = set()

    def add_best(
        title: str,
        bad_method: str,
        good_method: str,
        predicate,
        score,
    ) -> None:
        candidates = []
        for key, methods in grouped.items():
            if key in used_keys:
                continue
            bad = methods.get(bad_method)
            good = methods.get(good_method)
            if bad is None or good is None:
                continue
            if predicate(key[0], bad, good):
                candidates.append((score(key[0], bad, good), key, methods))
        if not candidates:
            return
        _, key, methods = sorted(candidates, key=lambda item: item[0], reverse=True)[0]
        used_keys.add(key)
        selected.append((title, str(key[1]), methods, bad_method, good_method))

    add_best(
        "Count Beam 失败但 MPNN 成功",
        "Read-weighted Count Beam",
        "Read-weighted MPNN Beam",
        lambda rate, bad, good: rate > 0 and as_float(bad, "edit_distance") > 0 and as_float(good, "exact") == 1.0,
        lambda rate, bad, good: (rate, as_float(bad, "edit_distance")),
    )
    add_best(
        "Count Beam 失败但 ReadAux 成功",
        "Read-weighted Count Beam",
        "Read-weighted MPNN+ReadAux Beam",
        lambda rate, bad, good: rate > 0 and as_float(bad, "edit_distance") > 0 and as_float(good, "exact") == 1.0,
        lambda rate, bad, good: (rate, as_float(bad, "edit_distance")),
    )
    add_best(
        "MPNN 未完全正确但 ReadAux 降低 AvgED",
        "Read-weighted MPNN Beam",
        "Read-weighted MPNN+ReadAux Beam",
        lambda rate, bad, good: rate > 0
        and as_float(bad, "edit_distance") > 0
        and as_float(good, "edit_distance") < as_float(bad, "edit_distance"),
        lambda rate, bad, good: (as_float(bad, "edit_distance") - as_float(good, "edit_distance"), rate),
    )

    rows: List[Dict[str, object]] = []
    for title, cluster_id, methods, bad_method, good_method in selected:
        bad = methods[bad_method]
        good = methods[good_method]
        reference = bad["reference"]
        rate = round(as_float(bad, "contamination_rate"), 4)
        avg_read_weight = as_float(good, "avg_read_weight", as_float(bad, "avg_read_weight", 1.0))
        alignment = summarize_alignment(reference, bad["prediction"])
        rows.append(
            {
                "案例": title,
                "cluster_id": cluster_id,
                "污染率": rate_label(rate),
                "参考序列长度": len(reference),
                "主要失败方法": bad_method,
                "主要改进方法": good_method,
                "失败方法ED": as_int(bad, "edit_distance"),
                "改进方法ED": as_int(good, "edit_distance"),
                "Count Beam ED": as_int(methods.get("Read-weighted Count Beam", {}), "edit_distance"),
                "MPNN ED": as_int(methods.get("Read-weighted MPNN Beam", {}), "edit_distance"),
                "ReadAux ED": as_int(methods.get("Read-weighted MPNN+ReadAux Beam", {}), "edit_distance"),
                "错误集中位置": alignment["错误集中位置"],
                "bridge/bypass判断": alignment["bridge/bypass判断"],
                "污染reads支持判断": contaminant_support_text(rate, avg_read_weight),
                "参考序列": reference,
                "失败方法预测": bad["prediction"],
                "改进方法预测": good["prediction"],
            }
        )
    return rows


def markdown_table(rows: Sequence[Dict[str, object]], columns: Sequence[str]) -> str:
    if not rows:
        return "暂无可用数据。\n"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        values = [str(row.get(column, "")) for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def build_markdown_report(
    main_srr_rows: Sequence[Dict[str, object]],
    main_brr_rows: Sequence[Dict[str, object]],
    main_avged_rows: Sequence[Dict[str, object]],
    read_weight_rows: Sequence[Dict[str, object]],
    mpnn_rows: Sequence[Dict[str, object]],
    read_aux_rows: Sequence[Dict[str, object]],
    train100_metric_rows: Sequence[Dict[str, object]],
    train1000_metric_rows: Sequence[Dict[str, object]],
    case_rows: Sequence[Dict[str, object]],
) -> str:
    main_columns = ["污染率", *CORE_METHODS]
    train_size_columns = ["污染率", "方法", "SRR", "BRR", "AvgED"]
    lines = [
        "# 污染感知 DNA 存储序列重构增强分析",
        "",
        "本文件把大规模实验结果整理为更适合课程报告使用的形式。重点不是只看 SRR，而是同时观察 SRR、BRR 和 AvgED：SRR 表示完全重构成功率，BRR 表示碱基层面的恢复比例，AvgED 表示平均编辑距离。某些方法 SRR 接近时，AvgED 仍可能不同，因此三者需要一起展示。",
        "",
        "## 1. 主结果三指标表",
        "",
        "### 1.1 SRR 主表",
        "",
        markdown_table(main_srr_rows, main_columns),
        "### 1.2 BRR 主表",
        "",
        markdown_table(main_brr_rows, main_columns),
        "### 1.3 AvgED 主表",
        "",
        markdown_table(main_avged_rows, main_columns),
        "## 2. Read-weighted graph 消融实验",
        "",
        "该实验回答 read_weight 是否真正有用。结果显示，read-weighted graph 在所有污染率下均提升 Count Beam SRR，同时显著减少平均边数，说明基于 read reliability 的加权构图不只是形式上的改动，而是实际改变了图结构并抑制了污染 reads 对路径搜索的干扰。",
        "",
        markdown_table(
            read_weight_rows,
            [
                "污染率",
                "Unweighted Count Beam SRR",
                "Read-weighted Count Beam SRR",
                "SRR提升",
                "Unweighted 平均边数",
                "Read-weighted 平均边数",
                "边数减少比例",
            ],
        ),
        "## 3. Edge-aware MPNN 贡献实验",
        "",
        "该实验回答深度学习模型带来了什么。Read-weighted Count Beam 是较强的非学习基线；在此基础上，Edge-aware MPNN 进一步学习边是否应该出现在真实路径中，因此 SRR 明显提升、AvgED 明显下降。整体上，read-weighted graph 是基础，MPNN 是主要性能提升来源。",
        "",
        markdown_table(
            mpnn_rows,
            [
                "污染率",
                "Count Beam SRR",
                "Count Beam AvgED",
                "MPNN Beam SRR",
                "MPNN Beam AvgED",
                "MPNN+ReadAux SRR",
                "MPNN+ReadAux AvgED",
                "MPNN相对Count的SRR提升",
            ],
        ),
        "## 4. ReadAux 辅助任务分析",
        "",
        "ReadAux 不一定在每个污染率下都提升 SRR，但它能非常准确地识别污染 reads。通俗地说，模型不仅在恢复目标序列，也学会了判断哪些 reads 更可能来自外部污染。",
        "",
        markdown_table(read_aux_rows, ["污染率", "测试来源", "Precision", "Recall", "Read-F1", "Accuracy"]),
        "## 5. 100 训练簇 vs 1000 训练簇",
        "",
        "该组结果使用同一批 300 个测试簇，用于观察训练簇数量变化对重构性能、边判别能力和污染读段识别能力的影响。",
        "",
        "### 5.1 100 训练簇结果表",
        "",
        markdown_table(train100_metric_rows, train_size_columns),
        "### 5.2 1000 训练簇结果表",
        "",
        markdown_table(train1000_metric_rows, train_size_columns),
        "### 5.3 方法级训练规模对比图",
        "",
        "每张图对应一个方法，左侧展示 SRR，右侧展示 AvgED，用于观察 100 训练簇和 1000 训练簇在不同污染率下的差异。",
        "",
        "![Read-weighted Count Beam 训练规模对比](figures/train_size_read_weighted_count_beam.png)",
        "",
        "![Read-weighted MPNN Beam 训练规模对比](figures/train_size_read_weighted_mpnn_beam.png)",
        "",
        "![MPNN+ReadAux Beam 训练规模对比](figures/train_size_mpnn_readaux_beam.png)",
        "",
        "## 6. 曲线图",
        "",
        "![污染率-SRR](figures/srr_by_contamination.png)",
        "",
        "![污染率-AvgED](figures/avged_by_contamination.png)",
        "",
        "![污染率-Edge-F1](figures/edge_f1_by_contamination.png)",
        "",
        "![污染率-Read-F1](figures/read_f1_by_contamination.png)",
        "",
        "![污染率-平均边数](figures/avg_edges_by_contamination.png)",
        "",
        "## 7. 代表性簇案例分析",
        "",
        "案例分析基于预测序列与参考序列的编辑差异，以及该簇的污染率、平均 read_weight 等信息。bridge/bypass 判断是根据插入、删除和局部错误分布得到的启发式分析，用于帮助解释错误类型。",
        "",
        markdown_table(
            case_rows,
            [
                "案例",
                "cluster_id",
                "污染率",
                "参考序列长度",
                "主要失败方法",
                "主要改进方法",
                "失败方法ED",
                "改进方法ED",
                "错误集中位置",
                "bridge/bypass判断",
                "污染reads支持判断",
            ],
        ),
    ]
    return "\n".join(lines)


def write_case_markdown(case_rows: Sequence[Dict[str, object]]) -> None:
    lines = [
        "# 代表性簇案例分析",
        "",
        "本文件选择少量典型簇，观察不同方法在同一簇上的错误差异。这里的 bridge/bypass 判断来自编辑差异的启发式分析，主要用于辅助说明错误形态。",
        "",
    ]
    for row in case_rows:
        lines.extend(
            [
                f"## {row['案例']}",
                "",
                f"- cluster_id：{row['cluster_id']}",
                f"- 污染率：{row['污染率']}",
                f"- 参考序列长度：{row['参考序列长度']}",
                f"- 主要失败方法：{row['主要失败方法']}，ED={row['失败方法ED']}",
                f"- 主要改进方法：{row['主要改进方法']}，ED={row['改进方法ED']}",
                f"- 错误集中位置：{row['错误集中位置']}",
                f"- bridge/bypass 判断：{row['bridge/bypass判断']}",
                f"- 污染 reads 支持判断：{row['污染reads支持判断']}",
                "",
            ]
        )
    (OUTPUT_DIR / "case_studies.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    main_summary = read_csv_rows(MAIN_RESULTS / "method_summary.csv")
    main_edge = read_csv_rows(MAIN_RESULTS / "edge_metrics.csv")
    main_read = read_csv_rows(MAIN_RESULTS / "read_contamination_metrics.csv")
    main_detail = read_csv_rows(MAIN_RESULTS / "method_detail.csv")
    train100_summary = read_csv_rows(TRAIN100_SCALE_RESULTS / "method_summary.csv")
    train1000_scale_summary = read_csv_rows(TRAIN1000_SCALE_RESULTS / "method_summary.csv")
    contam40_summary = read_csv_rows(CONTAM40_RESULTS / "method_summary.csv")
    contam40_edge = read_csv_rows(CONTAM40_RESULTS / "edge_metrics.csv")
    contam40_read = read_csv_rows(CONTAM40_RESULTS / "read_contamination_metrics.csv")

    main_metrics = build_main_metrics(main_summary)
    read_weight_rows = build_read_weight_ablation(main_summary)
    mpnn_rows = build_mpnn_contribution(main_summary)
    read_aux_rows = build_read_aux_analysis(main_read, contam40_read)
    edge_f1_rows = build_edge_f1_rows(main_edge, contam40_edge)
    train_size_rows = build_train_size_comparison(train100_summary, train1000_scale_summary)
    case_rows = choose_case_studies(main_detail)
    main_srr_rows = pivot_main_metric_rows(main_metrics, "SRR")
    main_brr_rows = pivot_main_metric_rows(main_metrics, "BRR")
    main_avged_rows = pivot_main_metric_rows(main_metrics, "AvgED")
    train100_metric_rows, train1000_metric_rows = build_train_size_tables(train_size_rows)

    for old_name in [
        "main_metrics_srr_brr_avged.csv",
        "train_size_comparison.csv",
        "train_size_srr_comparison.csv",
        "train_size_avged_comparison.csv",
    ]:
        old_path = TABLE_DIR / old_name
        if old_path.exists():
            old_path.unlink()

    write_csv_rows(
        TABLE_DIR / "main_metrics_srr.csv",
        main_srr_rows,
        ["污染率", *CORE_METHODS],
    )
    write_csv_rows(
        TABLE_DIR / "main_metrics_brr.csv",
        main_brr_rows,
        ["污染率", *CORE_METHODS],
    )
    write_csv_rows(
        TABLE_DIR / "main_metrics_avged.csv",
        main_avged_rows,
        ["污染率", *CORE_METHODS],
    )
    write_csv_rows(
        TABLE_DIR / "read_weighted_graph_ablation.csv",
        read_weight_rows,
        [
            "污染率",
            "Unweighted Count Beam SRR",
            "Read-weighted Count Beam SRR",
            "SRR提升",
            "Unweighted BRR",
            "Read-weighted BRR",
            "Unweighted AvgED",
            "Read-weighted AvgED",
            "AvgED变化",
            "Unweighted 平均边数",
            "Read-weighted 平均边数",
            "边数减少比例",
        ],
    )
    write_csv_rows(
        TABLE_DIR / "edge_aware_mpnn_contribution.csv",
        mpnn_rows,
        [
            "污染率",
            "Count Beam SRR",
            "Count Beam BRR",
            "Count Beam AvgED",
            "MPNN Beam SRR",
            "MPNN Beam BRR",
            "MPNN Beam AvgED",
            "MPNN+ReadAux SRR",
            "MPNN+ReadAux BRR",
            "MPNN+ReadAux AvgED",
            "MPNN相对Count的SRR提升",
            "ReadAux相对MPNN的AvgED变化",
        ],
    )
    write_csv_rows(
        TABLE_DIR / "read_aux_analysis.csv",
        read_aux_rows,
        ["污染率", "测试来源", "测试簇数", "Precision", "Recall", "Read-F1", "Accuracy", "Read loss"],
    )
    write_csv_rows(
        TABLE_DIR / "edge_f1_by_contamination.csv",
        edge_f1_rows,
        ["污染率", "模型", "测试来源", "测试簇数", "Precision", "Recall", "Edge-F1", "Loss"],
    )
    write_csv_rows(
        TABLE_DIR / "train_size_100_clusters.csv",
        train100_metric_rows,
        [
            "训练规模",
            "污染率",
            "方法",
            "SRR",
            "BRR",
            "AvgED",
        ],
    )
    write_csv_rows(
        TABLE_DIR / "train_size_1000_clusters.csv",
        train1000_metric_rows,
        [
            "训练规模",
            "污染率",
            "方法",
            "SRR",
            "BRR",
            "AvgED",
        ],
    )
    write_csv_rows(
        TABLE_DIR / "case_studies.csv",
        case_rows,
        [
            "案例",
            "cluster_id",
            "污染率",
            "参考序列长度",
            "主要失败方法",
            "主要改进方法",
            "失败方法ED",
            "改进方法ED",
            "Count Beam ED",
            "MPNN ED",
            "ReadAux ED",
            "错误集中位置",
            "bridge/bypass判断",
            "污染reads支持判断",
            "参考序列",
            "失败方法预测",
            "改进方法预测",
        ],
    )

    draw_line_chart(
        FIGURE_DIR / "srr_by_contamination.png",
        "污染率 - SRR",
        "SRR",
        series_from_summary(main_summary, CORE_METHODS, "sequence_recovery_rate"),
        y_min=0.0,
        y_max=1.0,
    )
    draw_line_chart(
        FIGURE_DIR / "avged_by_contamination.png",
        "污染率 - AvgED",
        "AvgED",
        series_from_summary(main_summary, CORE_METHODS, "avg_edit_distance"),
        y_min=0.0,
    )
    draw_line_chart(
        FIGURE_DIR / "edge_f1_by_contamination.png",
        "污染率 - Edge-F1",
        "Edge-F1",
        series_from_metric_rows(main_edge, "edge_f1"),
        y_min=0.97,
        y_max=1.0,
    )
    draw_line_chart(
        FIGURE_DIR / "read_f1_by_contamination.png",
        "污染率 - Read-F1",
        "Read-F1",
        {"MPNN+ReadAux": [(float(str(row["污染率"]).rstrip("%")) / 100.0, float(row["Read-F1"])) for row in read_aux_rows]},
        y_min=0.95,
        y_max=1.0,
    )
    draw_grouped_bar_chart(
        FIGURE_DIR / "avg_edges_by_contamination.png",
        "污染率 - 平均边数",
        "平均边数",
        read_weight_rows,
    )
    for method in TRAIN_SIZE_METHODS:
        draw_train_size_method_chart(
            FIGURE_DIR / TRAIN_SIZE_FIGURE_NAMES[method],
            method,
            train100_metric_rows,
            train1000_metric_rows,
        )

    write_case_markdown(case_rows)
    report = build_markdown_report(
        main_srr_rows=main_srr_rows,
        main_brr_rows=main_brr_rows,
        main_avged_rows=main_avged_rows,
        read_weight_rows=read_weight_rows,
        mpnn_rows=mpnn_rows,
        read_aux_rows=read_aux_rows,
        train100_metric_rows=train100_metric_rows,
        train1000_metric_rows=train1000_metric_rows,
        case_rows=case_rows,
    )
    (OUTPUT_DIR / "analysis_report.md").write_text(report, encoding="utf-8")

    print(f"增强分析已生成: {OUTPUT_DIR}")
    print(f"表格目录: {TABLE_DIR}")
    print(f"图片目录: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
