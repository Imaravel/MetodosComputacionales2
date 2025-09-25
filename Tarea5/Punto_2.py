#!/usr/bin/env python3
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from pdf_plot import write_line_plot_pdf

A_SOURCE = 1000.0
B_REMOVAL = 20.0
HALF_LIFE_U_MIN = 23.4
HALF_LIFE_NP_DAYS = 2.36
MINUTES_PER_DAY = 24.0 * 60.0

HALF_LIFE_U_DAYS = HALF_LIFE_U_MIN / MINUTES_PER_DAY
LAMBDA_U = math.log(2.0) / HALF_LIFE_U_DAYS
LAMBDA_NP = math.log(2.0) / HALF_LIFE_NP_DAYS

INITIAL_STATE = (10.0, 10.0, 10.0)
T_MAX = 30.0
OUTPUT_DIR = Path(__file__).resolve().parent


@dataclass
class DeterministicResult:
    time: List[float]
    U: List[float]
    Np: List[float]
    Pu: List[float]
    equilibrium_time: float | None
    equilibrium_state: Tuple[float, float, float] | None


def derivatives(state: Sequence[float]) -> Tuple[float, float, float]:
    U, Np, Pu = state
    dU = A_SOURCE - LAMBDA_U * U
    dNp = LAMBDA_U * U - LAMBDA_NP * Np
    dPu = LAMBDA_NP * Np - B_REMOVAL * Pu
    return dU, dNp, dPu


def rk4_step(state: Sequence[float], dt: float) -> List[float]:
    k1 = derivatives(state)
    state_k2 = [state[i] + 0.5 * dt * k1[i] for i in range(3)]
    k2 = derivatives(state_k2)
    state_k3 = [state[i] + 0.5 * dt * k2[i] for i in range(3)]
    k3 = derivatives(state_k3)
    state_k4 = [state[i] + dt * k3[i] for i in range(3)]
    k4 = derivatives(state_k4)
    next_state = [
        max(state[i] + dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]), 0.0)
        for i in range(3)
    ]
    return next_state


def simulate_deterministic(dt: float, t_max: float) -> DeterministicResult:
    time = 0.0
    state = list(INITIAL_STATE)
    times = [time]
    U_values = [state[0]]
    Np_values = [state[1]]
    Pu_values = [state[2]]

    equilibrium_time: float | None = None
    equilibrium_state: Tuple[float, float, float] | None = None
    stable_time_accum = 0.0
    tolerance = 1e-3

    while time < t_max - 1e-12:
        step = min(dt, t_max - time)
        state = rk4_step(state, step)
        time += step
        times.append(time)
        U_values.append(state[0])
        Np_values.append(state[1])
        Pu_values.append(state[2])

        deriv = derivatives(state)
        max_der = max(abs(deriv[0]), abs(deriv[1]), abs(deriv[2]))
        if max_der < tolerance:
            stable_time_accum += step
            if equilibrium_time is None and stable_time_accum >= 1.0:
                equilibrium_time = time
                equilibrium_state = (state[0], state[1], state[2])
        else:
            stable_time_accum = 0.0

    return DeterministicResult(
        time=times,
        U=U_values,
        Np=Np_values,
        Pu=Pu_values,
        equilibrium_time=equilibrium_time,
        equilibrium_state=equilibrium_state,
    )


def compute_sigma(state: Sequence[float]) -> Tuple[float, float, float]:
    U, Np, Pu = state
    return (
        math.sqrt(max(0.0, A_SOURCE + LAMBDA_U * U)),
        math.sqrt(max(0.0, LAMBDA_U * U + LAMBDA_NP * Np)),
        math.sqrt(max(0.0, LAMBDA_NP * Np + B_REMOVAL * Pu)),
    )


def compute_mu(state: Sequence[float]) -> Tuple[float, float, float]:
    U, Np, Pu = state
    return (
        A_SOURCE - LAMBDA_U * U,
        LAMBDA_U * U - LAMBDA_NP * Np,
        LAMBDA_NP * Np - B_REMOVAL * Pu,
    )


def sde_rk2_step(state: Sequence[float], dt: float, rng: random.Random) -> List[float]:
    mu_current = compute_mu(state)
    sigma_current = compute_sigma(state)
    sqrt_dt = math.sqrt(dt)
    noises = [rng.gauss(0.0, 1.0) for _ in range(3)]
    signs = [rng.choice((-1.0, 1.0)) for _ in range(3)]

    k1 = []
    for i in range(3):
        noise_term = sqrt_dt * (noises[i] + signs[i]) * sigma_current[i]
        k1.append(dt * mu_current[i] + noise_term)

    intermediate = [max(state[i] + k1[i], 0.0) for i in range(3)]
    mu_intermediate = compute_mu(intermediate)
    sigma_intermediate = compute_sigma(intermediate)

    k2 = []
    for i in range(3):
        noise_term = sqrt_dt * (noises[i] + signs[i]) * sigma_intermediate[i]
        k2.append(dt * mu_intermediate[i] + noise_term)

    return [
        max(state[i] + 0.5 * (k1[i] + k2[i]), 0.0)
        for i in range(3)
    ]


def simulate_sde(dt: float, t_max: float, rng: random.Random) -> List[Tuple[float, float, float, float]]:
    time = 0.0
    state = list(INITIAL_STATE)
    results = [(time, state[0], state[1], state[2])]

    while time < t_max - 1e-12:
        step = min(dt, t_max - time)
        state = sde_rk2_step(state, step, rng)
        time += step
        results.append((time, state[0], state[1], state[2]))

    return results


def simulate_gillespie(rng: random.Random, t_max: float) -> List[Tuple[float, float, float, float]]:
    time = 0.0
    state = [INITIAL_STATE[0], INITIAL_STATE[1], INITIAL_STATE[2]]
    results = [(time, state[0], state[1], state[2])]

    while time < t_max - 1e-12:
        rates = [
            A_SOURCE,
            LAMBDA_U * state[0],
            LAMBDA_NP * state[1],
            B_REMOVAL * state[2],
        ]
        total_rate = sum(rates)
        if total_rate <= 0.0:
            break
        tau = rng.expovariate(total_rate)
        time += tau
        if time > t_max:
            break
        selection = rng.random() * total_rate
        cumulative = 0.0
        cumulative += rates[0]
        if selection < cumulative:
            state[0] += 1.0
        else:
            cumulative += rates[1]
            if selection < cumulative:
                if state[0] > 0.0:
                    state[0] -= 1.0
                state[1] += 1.0
            else:
                cumulative += rates[2]
                if selection < cumulative:
                    if state[1] > 0.0:
                        state[1] -= 1.0
                    state[2] += 1.0
                else:
                    if state[2] > 0.0:
                        state[2] -= 1.0
        results.append((min(time, t_max), state[0], state[1], state[2]))

    if results[-1][0] < t_max:
        results.append((t_max, state[0], state[1], state[2]))

    return results


def downsample(points: Sequence[Tuple[float, ...]], max_points: int = 600) -> List[Tuple[float, ...]]:
    if len(points) <= max_points:
        return list(points)
    step = (len(points) - 1) / float(max_points - 1)
    result: List[Tuple[float, ...]] = []
    position = 0.0
    for _ in range(max_points - 1):
        idx = int(round(position))
        result.append(points[idx])
        position += step
    result.append(points[-1])
    return result


def extract_plot_series(times: Sequence[float], values: Sequence[float]) -> List[Tuple[float, float]]:
    return list(zip(times, values))


def create_point_series(data: Sequence[Tuple[float, float, float, float]], index: int) -> List[Tuple[float, float]]:
    return [(entry[0], entry[index]) for entry in data]


def crosses_threshold(series: Sequence[Tuple[float, float]], threshold: float) -> bool:
    return any(value >= threshold for _, value in series)


def estimate_probability(method: str, generator, trial_count: int, threshold: float) -> Dict[str, float]:
    hits = 0
    for _ in range(trial_count):
        trajectory = generator()
        if crosses_threshold(create_point_series(trajectory, 3), threshold):
            hits += 1
    probability = hits / trial_count
    error = math.sqrt(probability * (1.0 - probability) / trial_count)
    return {
        "method": method,
        "trials": trial_count,
        "hits": hits,
        "probability": probability,
        "uncertainty": error,
    }


def main() -> None:
    deterministic = simulate_deterministic(dt=0.001, t_max=T_MAX)

    if deterministic.equilibrium_time is not None and deterministic.equilibrium_state is not None:
        state_info = (
            f"Equilibrio detectado en t≈{deterministic.equilibrium_time:.2f} dias"
            f" con U≈{deterministic.equilibrium_state[0]:.2f},"
            f" Np≈{deterministic.equilibrium_state[1]:.2f},"
            f" Pu≈{deterministic.equilibrium_state[2]:.2f}"
        )
    else:
        state_info = "No se detecto equilibrio en la ventana de 30 dias"
    print(state_info)

    deterministic_series = [
        {
            "points": downsample(extract_plot_series(deterministic.time, deterministic.U)),
            "color": (0.2, 0.4, 0.8),
            "width": 1.2,
            "label": "U determinista",
        },
        {
            "points": downsample(extract_plot_series(deterministic.time, deterministic.Np)),
            "color": (0.9, 0.5, 0.1),
            "width": 1.2,
            "label": "Np determinista",
        },
        {
            "points": downsample(extract_plot_series(deterministic.time, deterministic.Pu)),
            "color": (0.2, 0.7, 0.3),
            "width": 1.2,
            "label": "Pu determinista",
        },
    ]
    write_line_plot_pdf(
        OUTPUT_DIR / "2.a.pdf",
        deterministic_series,
        title="2.a Evolucion determinista de isotopos",
        x_label="Tiempo (dias)",
        y_label="Cantidad",
    )

    rng_stochastic = random.Random(202510)
    dt_sde = 0.05
    sde_trajectories = [simulate_sde(dt_sde, T_MAX, random.Random(rng_stochastic.randrange(1, 10_000_000))) for _ in range(5)]
    pu_deterministic_series = {
        "points": downsample(extract_plot_series(deterministic.time, deterministic.Pu)),
        "color": (0.1, 0.1, 0.1),
        "width": 1.5,
        "label": "Pu determinista",
    }
    sde_series = [
        {
            "points": downsample(create_point_series(traj, 3)),
            "color": (0.6, 0.6, 0.9),
            "width": 0.9,
            "label": f"Trayectoria SDE {idx + 1}",
        }
        for idx, traj in enumerate(sde_trajectories)
    ]
    write_line_plot_pdf(
        OUTPUT_DIR / "2.b.pdf",
        [pu_deterministic_series] + sde_series,
        title="2.b Pu: SDE vs determinista",
        x_label="Tiempo (dias)",
        y_label="Pu",
    )

    rng_gillespie = random.Random(98765)
    gillespie_trajectories = [simulate_gillespie(random.Random(rng_gillespie.randrange(1, 10_000_000)), T_MAX) for _ in range(5)]
    gillespie_series = [
        {
            "points": downsample(create_point_series(traj, 3)),
            "color": (0.9, 0.6, 0.6),
            "width": 0.9,
            "label": f"Trayectoria Gillespie {idx + 1}",
        }
        for idx, traj in enumerate(gillespie_trajectories)
    ]
    write_line_plot_pdf(
        OUTPUT_DIR / "2.c.pdf",
        [pu_deterministic_series] + gillespie_series,
        title="2.c Pu: Gillespie vs determinista",
        x_label="Tiempo (dias)",
        y_label="Pu",
    )

    threshold = 80.0
    prob_results: List[Dict[str, float]] = []

    deterministic_cross = crosses_threshold(extract_plot_series(deterministic.time, deterministic.Pu), threshold)
    prob_results.append({
        "method": "Determinista",
        "trials": 1,
        "hits": 1 if deterministic_cross else 0,
        "probability": 1.0 if deterministic_cross else 0.0,
        "uncertainty": 0.0,
    })

    sde_seed_stream = random.Random(11111)
    prob_results.append(
        estimate_probability(
            "SDE RK2",
            generator=lambda: simulate_sde(dt_sde, T_MAX, random.Random(sde_seed_stream.randrange(1, 10_000_000))),
            trial_count=1000,
            threshold=threshold,
        )
    )

    gillespie_seed_stream = random.Random(22222)
    prob_results.append(
        estimate_probability(
            "Gillespie",
            generator=lambda: simulate_gillespie(random.Random(gillespie_seed_stream.randrange(1, 10_000_000)), T_MAX),
            trial_count=1000,
            threshold=threshold,
        )
    )

    lines: List[str] = []
    lines.append("Metodo,Probabilidad (%),Incertidumbre (%),Exitos,Intentos")
    for result in prob_results:
        probability_percent = result["probability"] * 100.0
        uncertainty_percent = result["uncertainty"] * 100.0
        lines.append(
            f"{result['method']},{probability_percent:.2f},{uncertainty_percent:.2f},{int(result['hits'])},{int(result['trials'])}"
        )

    deterministic_msg = "La solucion determinista no alcanza el umbral." if not deterministic_cross else "La solucion determinista supera el umbral sin estocasticidad."
    discussion = [
        deterministic_msg,
        "Las trayectorias estocasticas rara vez alcanzan 80 unidades en 30 dias con los parametros dados.",
    ]
    lines.append("")
    lines.extend(discussion)

    (OUTPUT_DIR / "2.d.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
