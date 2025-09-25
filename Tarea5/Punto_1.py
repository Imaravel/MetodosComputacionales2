#!/usr/bin/env python3
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

from pdf_plot import line_plot_lines, linspace, write_custom_pdf, write_line_plot_pdf

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


if __name__ == "__main__":
    main()
