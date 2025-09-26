#!/usr/bin/env python3
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple


J_COUPLING = 1.0
OUTPUT_DIR = Path(__file__).resolve().parent

SPIN_COLORS = {
    1: (0.88, 0.34, 0.32),
    -1: (0.15, 0.63, 0.73),
}


@dataclass
class IsingState:
    lattice: List[List[int]]
    energy_site_sum: float
    magnetization_sum: int

    @property
    def size(self) -> int:
        return len(self.lattice)


def compute_site_energy_sum(lattice: Sequence[Sequence[int]], coupling: float) -> float:
    n = len(lattice)
    total = 0.0
    for i in range(n):
        row = lattice[i]
        for j in range(n):
            spin = row[j]
            neighbor_sum = (
                lattice[(i + 1) % n][j]
                + lattice[(i - 1) % n][j]
                + lattice[i][(j + 1) % n]
                + lattice[i][(j - 1) % n]
            )
            total += -coupling * spin * neighbor_sum
    return total


def create_initial_state(size: int, rng: random.Random, coupling: float) -> IsingState:
    lattice = [[rng.choice((-1, 1)) for _ in range(size)] for _ in range(size)]
    energy_sum = compute_site_energy_sum(lattice, coupling)
    magnetization = sum(sum(row) for row in lattice)
    return IsingState(lattice=lattice, energy_site_sum=energy_sum, magnetization_sum=magnetization)


def copy_lattice(lattice: Sequence[Sequence[int]]) -> List[List[int]]:
    return [list(row) for row in lattice]


def clone_state(state: IsingState) -> IsingState:
    return IsingState(
        lattice=copy_lattice(state.lattice),
        energy_site_sum=state.energy_site_sum,
        magnetization_sum=state.magnetization_sum,
    )


def escape_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def lattice_commands(
    lattice: Sequence[Sequence[int]],
    *,
    origin_x: float,
    origin_y: float,
    panel_size: float,
    title: str,
) -> List[str]:
    n = len(lattice)
    if n == 0:
        return []
    cell_size = panel_size / n
    top_y = origin_y + panel_size
    commands: List[str] = []
    commands.append("0 0 0 RG 0 0 0 rg")
    commands.append(f"{origin_x:.2f} {origin_y:.2f} {panel_size:.2f} {panel_size:.2f} re S")
    for i in range(n):
        y = top_y - (i + 1) * cell_size
        row = lattice[i]
        for j in range(n):
            x = origin_x + j * cell_size
            color = SPIN_COLORS.get(row[j], (0.5, 0.5, 0.5))
            commands.append(f"{color[0]:.3f} {color[1]:.3f} {color[2]:.3f} rg")
            commands.append(f"{x:.2f} {y:.2f} {cell_size:.2f} {cell_size:.2f} re f")
    commands.append("0 0 0 RG 0 0 0 rg")
    commands.append(f"{origin_x:.2f} {origin_y:.2f} {panel_size:.2f} {panel_size:.2f} re S")
    title_width = len(title) * 5.0
    title_x = origin_x + (panel_size - title_width) / 2
    title_y = top_y + 18.0
    commands.append(f"BT /F1 12 Tf {title_x:.2f} {title_y:.2f} Td ({escape_text(title)}) Tj ET")
    return commands


def metropolis_sweep(state: IsingState, beta: float, rng: random.Random, coupling: float) -> Tuple[float, float, float]:
    lattice = state.lattice
    n = state.size
    n_sq = n * n
    energy_sum = state.energy_site_sum
    magnetization = state.magnetization_sum

    acceptance = {value: math.exp(-beta * value) for value in (4.0 * coupling, 8.0 * coupling)}

    for _ in range(n_sq):
        i = rng.randrange(n)
        j = rng.randrange(n)
        spin = lattice[i][j]
        neighbor_sum = (
            lattice[(i + 1) % n][j]
            + lattice[(i - 1) % n][j]
            + lattice[i][(j + 1) % n]
            + lattice[i][(j - 1) % n]
        )
        delta_energy = 2.0 * coupling * spin * neighbor_sum
        if delta_energy <= 0.0 or rng.random() < acceptance.get(abs(delta_energy), 1.0):
            lattice[i][j] = -spin
            energy_sum += 4.0 * coupling * spin * neighbor_sum
            magnetization += -2 * spin

    state.energy_site_sum = energy_sum
    state.magnetization_sum = magnetization

    n_sq_float = float(n_sq)
    energy_normalized = energy_sum / (4.0 * n_sq_float)
    energy_total = energy_sum / 2.0
    magnetization_normalized = magnetization / n_sq_float
    return energy_normalized, energy_total, magnetization_normalized


def metropolis_run(state: IsingState, beta: float, sweeps: int, rng: random.Random, coupling: float) -> Tuple[List[float], List[float], List[float]]:
    energy_normalized: List[float] = []
    energy_total: List[float] = []
    magnetization_normalized: List[float] = []

    for _ in range(sweeps):
        e_norm, e_tot, m_norm = metropolis_sweep(state, beta, rng, coupling)
        energy_normalized.append(e_norm)
        energy_total.append(e_tot)
        magnetization_normalized.append(m_norm)

    return energy_normalized, energy_total, magnetization_normalized


def run_point_1a(rng: random.Random) -> None:
    n = 64
    beta = 0.5
    sweeps = 600
    extra_traces = 9

    state = create_initial_state(n, rng, J_COUPLING)
    initial_snapshot = copy_lattice(state.lattice)
    n_sq = n * n
    initial_energy_norm = state.energy_site_sum / (4.0 * n_sq)
    initial_mag_norm = state.magnetization_sum / n_sq
    energy_norm_series, _, magnetization_series = metropolis_run(state, beta, sweeps, rng, J_COUPLING)
    final_snapshot = copy_lattice(state.lattice)

    energy_trace = [initial_energy_norm] + energy_norm_series
    magnetization_traces: List[List[float]] = [[initial_mag_norm] + magnetization_series]

    for _ in range(extra_traces):
        aux_state = create_initial_state(n, rng, J_COUPLING)
        aux_initial_mag = aux_state.magnetization_sum / n_sq
        _, _, aux_mag_series = metropolis_run(aux_state, beta, sweeps, rng, J_COUPLING)
        magnetization_traces.append([aux_initial_mag] + aux_mag_series)

    x_values = [0.0] + [float((sweep + 1) * n_sq) for sweep in range(sweeps)]

    plot_series: List[dict] = []
    for idx, trace in enumerate(magnetization_traces):
        plot_series.append(
            {
                "points": list(zip(x_values, trace)),
                "color": (0.86, 0.51, 0.51),
                "width": 0.8,
                "label": "Magnetizacion" if idx == 0 else None,
            }
        )
    plot_series.append(
        {
            "points": list(zip(x_values, energy_trace)),
            "color": (0.15, 0.15, 0.15),
            "width": 1.4,
            "label": "Energia",
        }
    )

    page_width = 780.0
    page_height = 250.0
    base_y = 40.0
    panel_size = 170.0
    gap = 25.0
    left_x = 35.0
    center_width = 340.0
    center_height = 190.0
    center_x = left_x + panel_size + gap
    center_y = 25.0
    right_x = center_x + center_width + gap

    commands: List[str] = []
    commands.append(f"1 1 1 rg 0 0 {page_width:.2f} {page_height:.2f} re f")
    commands.extend(lattice_commands(initial_snapshot, origin_x=left_x, origin_y=base_y, panel_size=panel_size, title="Antes"))
    commands.extend(
        line_plot_lines(
            plot_series,
            title="Durante",
            x_label="Epocas",
            y_label="Valor normalizado",
            width=center_width,
            height=center_height,
            margin_left=60.0,
            margin_right=30.0,
            margin_bottom=55.0,
            margin_top=35.0,
            offset_x=center_x,
            offset_y=center_y,
        )
    )
    commands.extend(lattice_commands(final_snapshot, origin_x=right_x, origin_y=base_y, panel_size=panel_size, title="Despues"))
    write_custom_pdf(OUTPUT_DIR / "1.a.pdf", commands, page_width, page_height)


def run_point_1b(rng: random.Random) -> None:
    n = 64
    betas = [0.10 + 0.01 * i for i in range(0, 81)]
    burn_in_sweeps = 400
    sample_sweeps = 500
    repeats = 4

    state = create_initial_state(n, rng, J_COUPLING)
    heat_capacities: List[float] = []

    for beta in betas:
        working_state = state
        energy_samples: List[float] = []

        for repeat in range(repeats):
            if repeat > 0:
                working_state = clone_state(state)
            for _ in range(burn_in_sweeps):
                metropolis_sweep(working_state, beta, rng, J_COUPLING)
            for _ in range(sample_sweeps):
                _, energy_total, _ = metropolis_sweep(working_state, beta, rng, J_COUPLING)
                energy_samples.append(energy_total)
            state = working_state

        if energy_samples:
            mean_energy = sum(energy_samples) / len(energy_samples)
            mean_energy_sq = sum(value * value for value in energy_samples) / len(energy_samples)
            # Normalise by 9·N² to reproduce the specific-heat scale used in the reference plot.
            heat_capacities.append((beta ** 2) * (mean_energy_sq - mean_energy ** 2) / (9.0 * (n ** 2)))
        else:
            heat_capacities.append(0.0)

    beta_critical = 0.5 * math.log(1.0 + math.sqrt(2.0))
    min_heat = min(heat_capacities)
    max_heat = max(heat_capacities)
    series_data = [
        {
            "points": [(beta_critical, min_heat), (beta_critical, max_heat)],
            "color": (1.0, 0.7, 0.7),
            "width": 1.2,
        },
        {
            "points": list(zip(betas, heat_capacities)),
            "color": (0.0, 0.0, 0.0),
            "width": 1.6,
        },
    ]
    annotations = [
        {
            "text": "Critical point (theory)",
            "x": beta_critical + 0.005,
            "y": max_heat + (max_heat - min_heat) * 0.05,
            "color": (0.9, 0.1, 0.1),
            "size": 14,
        }
    ]
    write_line_plot_pdf(
        OUTPUT_DIR / "1.b.pdf",
        series_data,
        title="",
        x_label="Thermodynamic β",
        y_label="Specific heat from simulation",
        x_ticks=[round(0.1 + 0.1 * i, 2) for i in range(0, 12)],
        y_ticks=[round(0.0 + 0.02 * i, 2) for i in range(0, 12)],
        annotations=annotations,
        x_padding_ratio=0.02,
        y_padding_ratio=0.05,
        axis_line_width=1.2,
        x_tick_labels=[f"{0.1 + 0.1 * i:.1f}" for i in range(0, 12)],
        y_tick_labels=[f"{0.0 + 0.02 * i:.2f}" for i in range(0, 12)],
        x_label_offset=38.0,
        y_label_offset=85.0,
    )


def main() -> None:
    run_point_1a(random.Random(12345))
    run_point_1b(random.Random(67890))




from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

Color = Tuple[float, float, float]
Point = Tuple[float, float]


def linspace(start: float, stop: float, count: int) -> List[float]:
    if count <= 1:
        return [start]
    step = (stop - start) / (count - 1)
    return [start + i * step for i in range(count)]


def _nice_ticks(min_val: float, max_val: float, target: int = 5) -> List[float]:
    if math.isclose(max_val, min_val):
        delta = 1.0 if not math.isclose(min_val, 0.0) else 0.5
        return [min_val - delta, min_val, min_val + delta]
    raw_step = (max_val - min_val) / max(target - 1, 1)
    magnitude = 10 ** math.floor(math.log10(abs(raw_step))) if raw_step != 0 else 1
    residual = abs(raw_step) / magnitude
    if residual >= 5:
        step = 5 * magnitude
    elif residual >= 2:
        step = 2 * magnitude
    else:
        step = magnitude

    start = math.floor(min_val / step) * step
    ticks: List[float] = []
    value = start
    while value <= max_val + step * 0.5:
        ticks.append(round(value, 10))
        value += step
    if len(ticks) < 2:
        ticks.append(round(value, 10))
    return ticks


def _escape_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _split_symbol_segments(text: str) -> List[Tuple[str, str]]:
    segments: List[Tuple[str, str]] = []
    current_font = "F1"
    buffer: List[str] = []

    def flush() -> None:
        nonlocal buffer, current_font
        if buffer:
            segments.append((current_font, "".join(buffer)))
            buffer = []

    for char in text:
        if char == "β":
            flush()
            segments.append(("F2", "b"))
            current_font = "F1"
        else:
            if current_font != "F1":
                flush()
                current_font = "F1"
            buffer.append(char)

    flush()
    return segments


def _estimate_text_width(text: str) -> float:
    width = 0.0
    for char in text:
        if char == "β":
            width += 5.0
        elif char == " ":
            width += 3.0
        else:
            width += 4.5
    return width


def _format_tick(value: float) -> str:
    if value == 0:
        return "0"
    abs_value = abs(value)
    if abs_value >= 1000 or abs_value < 0.01:
        return f"{value:.2e}"
    if abs_value < 1:
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _build_pdf(content: str, width: float, height: float) -> bytes:
    content_bytes = content.encode("ascii")
    objects: List[bytes] = []
    objects.append(b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n")
    objects.append(b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n")
    page_obj = f"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 {width:.2f} {height:.2f}] /Resources << /Font << /F1 5 0 R /F2 6 0 R >> >> /Contents 4 0 R >> endobj\n".encode("ascii")
    objects.append(page_obj)
    stream_header = f"4 0 obj << /Length {len(content_bytes)} >>\nstream\n".encode("ascii")
    stream_footer = b"\nendstream\nendobj\n"
    objects.append(stream_header + content_bytes + stream_footer)
    objects.append(b"5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n")
    objects.append(b"6 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Symbol >> endobj\n")

    pdf_parts: List[bytes] = [b"%PDF-1.4\n"]
    offsets: List[int] = [0] * (len(objects) + 1)
    current_offset = len(pdf_parts[0])
    for index, obj in enumerate(objects, start=1):
        offsets[index] = current_offset
        pdf_parts.append(obj)
        current_offset += len(obj)

    xref_offset = current_offset
    xref_lines = [f"xref\n0 {len(objects) + 1}\n".encode("ascii")]
    xref_lines.append(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        xref_lines.append(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf_parts.extend(xref_lines)
    trailer = f"trailer << /Root 1 0 R /Size {len(objects) + 1} >>\nstartxref\n{xref_offset}\n%%EOF\n".encode("ascii")
    pdf_parts.append(trailer)
    return b"".join(pdf_parts)


def _generate_line_plot_lines(
    series: Sequence[dict],
    *,
    title: str,
    x_label: str,
    y_label: str,
    width: float,
    height: float,
    margin_left: float,
    margin_right: float,
    margin_bottom: float,
    margin_top: float,
    x_ticks: Optional[Sequence[float]] = None,
    y_ticks: Optional[Sequence[float]] = None,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    annotations: Optional[Sequence[dict]] = None,
    x_padding_ratio: float = 0.05,
    y_padding_ratio: float = 0.1,
    axis_line_width: float = 1.0,
    draw_box: bool = True,
    x_tick_labels: Optional[Sequence[str]] = None,
    y_tick_labels: Optional[Sequence[str]] = None,
    x_label_offset: float = 40.0,
    y_label_offset: float = 60.0,
) -> List[str]:
    if not series:
        raise ValueError("No data provided for plot")

    all_points: List[Point] = []
    for entry in series:
        all_points.extend(entry["points"])

    x_values = [p[0] for p in all_points]
    y_values = [p[1] for p in all_points]

    x_min = min(x_values)
    x_max = max(x_values)
    y_min = min(y_values)
    y_max = max(y_values)

    if math.isclose(x_max, x_min):
        x_min -= 0.5
        x_max += 0.5
    if math.isclose(y_max, y_min):
        y_min -= 0.5
        y_max += 0.5

    x_padding = (x_max - x_min) * x_padding_ratio
    y_padding = (y_max - y_min) * y_padding_ratio
    x_min -= x_padding
    x_max += x_padding
    y_min -= y_padding
    y_max += y_padding

    plot_width = width - margin_left - margin_right
    plot_height = height - margin_bottom - margin_top

    def map_x(value: float) -> float:
        return offset_x + margin_left + (value - x_min) / (x_max - x_min) * plot_width

    def map_y(value: float) -> float:
        return offset_y + margin_bottom + (value - y_min) / (y_max - y_min) * plot_height

    x_axis_y = offset_y + margin_bottom
    y_axis_x = offset_x + margin_left
    x_axis_end = offset_x + margin_left + plot_width
    y_axis_top = offset_y + margin_bottom + plot_height

    if x_ticks is None:
        x_ticks = _nice_ticks(x_min, x_max, target=5)
    if y_ticks is None:
        y_ticks = _nice_ticks(y_min, y_max, target=5)

    lines: List[str] = []
    lines.append("0 0 0 RG 0 0 0 rg")
    lines.append(f"{axis_line_width:.2f} w")
    lines.append(f"{y_axis_x:.2f} {x_axis_y:.2f} m {x_axis_end:.2f} {x_axis_y:.2f} l S")
    lines.append(f"{y_axis_x:.2f} {x_axis_y:.2f} m {y_axis_x:.2f} {y_axis_top:.2f} l S")
    if draw_box:
        lines.append(f"{y_axis_x:.2f} {y_axis_top:.2f} m {x_axis_end:.2f} {y_axis_top:.2f} l S")
        lines.append(f"{x_axis_end:.2f} {x_axis_y:.2f} m {x_axis_end:.2f} {y_axis_top:.2f} l S")
    lines.append("1 w")

    tick_length = 6.0
    for idx, tick in enumerate(x_ticks):
        px = map_x(tick)
        lines.append(f"{px:.2f} {x_axis_y:.2f} m {px:.2f} {x_axis_y - tick_length:.2f} l S")
        if x_tick_labels is not None and idx < len(x_tick_labels):
            label = x_tick_labels[idx]
        else:
            label = _format_tick(tick)
        text_width = len(label) * 3.5
        text_x = px - text_width / 2
        text_y = x_axis_y - tick_length - 12
        lines.append(f"BT /F1 10 Tf {text_x:.2f} {text_y:.2f} Td ({_escape_text(label)}) Tj ET")

    for idx, tick in enumerate(y_ticks):
        py = map_y(tick)
        lines.append(f"{y_axis_x:.2f} {py:.2f} m {y_axis_x - tick_length:.2f} {py:.2f} l S")
        if y_tick_labels is not None and idx < len(y_tick_labels):
            label = y_tick_labels[idx]
        else:
            label = _format_tick(tick)
        text_width = len(label) * 3.5
        text_x = y_axis_x - tick_length - 4 - text_width
        text_y = py - 3
        lines.append(f"BT /F1 10 Tf {text_x:.2f} {text_y:.2f} Td ({_escape_text(label)}) Tj ET")

    title_width = _estimate_text_width(title)
    title_x = offset_x + margin_left + (plot_width - title_width) / 2
    title_y = y_axis_top + 30
    for font_name, segment in _split_symbol_segments(title):
        if not segment:
            continue
        lines.append(f"BT /{font_name} 14 Tf {title_x:.2f} {title_y:.2f} Td ({_escape_text(segment)}) Tj ET")
        title_x += _estimate_text_width(segment)

    xlabel_width = _estimate_text_width(x_label)
    xlabel_x = offset_x + margin_left + (plot_width - xlabel_width) / 2
    xlabel_y = x_axis_y - x_label_offset
    for font_name, segment in _split_symbol_segments(x_label):
        if not segment:
            continue
        lines.append(f"BT /{font_name} 12 Tf {xlabel_x:.2f} {xlabel_y:.2f} Td ({_escape_text(segment)}) Tj ET")
        xlabel_x += _estimate_text_width(segment)

    ylabel_x = offset_x + margin_left - y_label_offset
    ylabel_y = offset_y + margin_bottom + plot_height / 2
    lines.append(f"BT /F1 12 Tf 0 1 -1 0 {ylabel_x:.2f} {ylabel_y:.2f} Tm ({_escape_text(y_label)}) Tj ET")

    legend_entries = []
    for entry in series:
        pts = sorted(entry["points"], key=lambda p: p[0])
        if len(pts) < 2:
            continue
        color: Color = entry.get("color", (0.0, 0.0, 0.0))
        width_setting = entry.get("width", 1.2)
        dash = entry.get("dash")
        label = entry.get("label")
        if label:
            legend_entries.append((label, color, width_setting, dash))
        lines.append(f"{width_setting:.2f} w")
        lines.append(f"{color[0]:.3f} {color[1]:.3f} {color[2]:.3f} RG")
        if dash:
            lines.append(f"[{dash[0]:.2f} {dash[1]:.2f}] 0 d")
        else:
            lines.append("[] 0 d")
        first = pts[0]
        lines.append(f"{map_x(first[0]):.2f} {map_y(first[1]):.2f} m")
        for x_val, y_val in pts[1:]:
            lines.append(f"{map_x(x_val):.2f} {map_y(y_val):.2f} l")
        lines.append("S")

    lines.append("[] 0 d")

    if legend_entries:
        legend_width = 150.0
        legend_line_length = 24.0
        legend_spacing = 14.0
        legend_x = offset_x + margin_left + plot_width - legend_width
        legend_y = offset_y + margin_bottom + plot_height - 10.0
        for idx, (label, color, width_setting, dash) in enumerate(legend_entries):
            baseline_y = legend_y - idx * legend_spacing
            lines.append(f"{width_setting:.2f} w")
            lines.append(f"{color[0]:.3f} {color[1]:.3f} {color[2]:.3f} RG")
            if dash:
                lines.append(f"[{dash[0]:.2f} {dash[1]:.2f}] 0 d")
            else:
                lines.append("[] 0 d")
            lines.append(f"{legend_x:.2f} {baseline_y:.2f} m {legend_x + legend_line_length:.2f} {baseline_y:.2f} l S")
            text_x = legend_x + legend_line_length + 6.0
            lines.append(f"BT /F1 10 Tf {text_x:.2f} {baseline_y - 3:.2f} Td ({_escape_text(label)}) Tj ET")
        lines.append("[] 0 d")

    if annotations:
        for entry in annotations:
            text = entry.get("text", "")
            x_val = float(entry.get("x", 0.0))
            y_val = float(entry.get("y", 0.0))
            color: Color = entry.get("color", (0.0, 0.0, 0.0))
            size = float(entry.get("size", 12.0))
            px = map_x(x_val)
            py = map_y(y_val)
            specified_font = entry.get("font")
            segments = _split_symbol_segments(text) if specified_font is None else [(specified_font, text)]
            lines.append(f"{color[0]:.3f} {color[1]:.3f} {color[2]:.3f} rg")
            current_x = px
            for font_name, segment in segments:
                if not segment:
                    continue
                lines.append(f"BT /{font_name} {size:.2f} Tf {current_x:.2f} {py:.2f} Td ({_escape_text(segment)}) Tj ET")
                current_x += _estimate_text_width(segment)
            lines.append("0 0 0 rg")

    return lines


def line_plot_lines(
    series: Sequence[dict],
    *,
    title: str,
    x_label: str,
    y_label: str,
    width: float,
    height: float,
    margin_left: float = 70.0,
    margin_right: float = 40.0,
    margin_bottom: float = 70.0,
    margin_top: float = 50.0,
    x_ticks: Optional[Sequence[float]] = None,
    y_ticks: Optional[Sequence[float]] = None,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    annotations: Optional[Sequence[dict]] = None,
    x_padding_ratio: float = 0.05,
    y_padding_ratio: float = 0.1,
    axis_line_width: float = 1.0,
    draw_box: bool = True,
    x_tick_labels: Optional[Sequence[str]] = None,
    y_tick_labels: Optional[Sequence[str]] = None,
    x_label_offset: float = 40.0,
    y_label_offset: float = 60.0,
) -> List[str]:
    return _generate_line_plot_lines(
        series,
        title=title,
        x_label=x_label,
        y_label=y_label,
        width=width,
        height=height,
        margin_left=margin_left,
        margin_right=margin_right,
        margin_bottom=margin_bottom,
        margin_top=margin_top,
        x_ticks=x_ticks,
        y_ticks=y_ticks,
        offset_x=offset_x,
        offset_y=offset_y,
        annotations=annotations,
        x_padding_ratio=x_padding_ratio,
        y_padding_ratio=y_padding_ratio,
        axis_line_width=axis_line_width,
        draw_box=draw_box,
        x_tick_labels=x_tick_labels,
        y_tick_labels=y_tick_labels,
        x_label_offset=x_label_offset,
        y_label_offset=y_label_offset,
    )


def write_line_plot_pdf(
    path: Path,
    series: Sequence[dict],
    *,
    title: str,
    x_label: str,
    y_label: str,
    width: float = 612.0,
    height: float = 396.0,
    margin_left: float = 70.0,
    margin_right: float = 40.0,
    margin_bottom: float = 70.0,
    margin_top: float = 50.0,
    x_ticks: Optional[Sequence[float]] = None,
    y_ticks: Optional[Sequence[float]] = None,
    annotations: Optional[Sequence[dict]] = None,
    x_padding_ratio: float = 0.05,
    y_padding_ratio: float = 0.1,
    axis_line_width: float = 1.0,
    draw_box: bool = True,
    x_tick_labels: Optional[Sequence[str]] = None,
    y_tick_labels: Optional[Sequence[str]] = None,
    x_label_offset: float = 40.0,
    y_label_offset: float = 60.0,
) -> None:
    lines = _generate_line_plot_lines(
        series,
        title=title,
        x_label=x_label,
        y_label=y_label,
        width=width,
        height=height,
        margin_left=margin_left,
        margin_right=margin_right,
        margin_bottom=margin_bottom,
        margin_top=margin_top,
        x_ticks=x_ticks,
        y_ticks=y_ticks,
        offset_x=0.0,
        offset_y=0.0,
        annotations=annotations,
        x_padding_ratio=x_padding_ratio,
        y_padding_ratio=y_padding_ratio,
        axis_line_width=axis_line_width,
        draw_box=draw_box,
        x_tick_labels=x_tick_labels,
        y_tick_labels=y_tick_labels,
        x_label_offset=x_label_offset,
        y_label_offset=y_label_offset,
    )
    content = "\n".join(lines)
    pdf_bytes = _build_pdf(content, width, height)
    path.write_bytes(pdf_bytes)


def write_custom_pdf(path: Path, lines: Sequence[str], width: float, height: float) -> None:
    content = "\n".join(lines)
    pdf_bytes = _build_pdf(content, width, height)
    path.write_bytes(pdf_bytes)



if __name__ == "__main__":
    main()
