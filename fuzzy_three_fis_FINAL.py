# -*- coding: utf-8 -*-
"""
Fuzzy Logic Comparative Application
Mamdani vs. Zero-Order Takagi-Sugeno vs. Tsukamoto

Standalone, reproducible script compatible with older Python/NumPy environments.

Inputs
------
1) Battery temperature error, ΔT = T_battery - T_target (°C)
2) Normalized heat-load indicator, Q (%)

Output
------
Cooling demand, Y (%) with linguistic levels:
Low, Moderate, High

The script:
- evaluates the common antecedent membership functions,
- computes the nine common rule firing strengths,
- evaluates Mamdani, zero-order Sugeno, and Tsukamoto,
- prints all six deterministic scenarios in one table,
- saves scenario_comparison.csv,
- saves five publication-ready figures in ./figures/.

No use is made of np.trapezoid or np.trapz, so the script is robust to
older NumPy releases commonly found in legacy Python environments.
"""

from __future__ import print_function

import csv
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

RULES = [
    ("Safe", "Low", "Low"),
    ("Safe", "Medium", "Low"),
    ("Safe", "High", "Moderate"),
    ("Warm", "Low", "Moderate"),
    ("Warm", "Medium", "Moderate"),
    ("Warm", "High", "High"),
    ("Hot", "Low", "High"),
    ("Hot", "Medium", "High"),
    ("Hot", "High", "High"),
]

SCENARIOS = [
    ("S1: Stable", 1.0, 15.0),
    ("S2: Moderate thermal rise", 4.5, 45.0),
    ("S3: Fast charging", 8.0, 75.0),
    ("S4: Severe thermal stress", 14.0, 90.0),
    ("S5: Hot ambient + medium load", 10.5, 50.0),
    ("S6: Warm battery + low load", 6.0, 20.0),
]

BASE_DIR = Path(__file__).resolve().parent
FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)
CSV_PATH = BASE_DIR / "scenario_comparison.csv"
RULE_CSV_PATH = BASE_DIR / "representative_rule_activation.csv"


def trapmf_scalar(x, a, b, c, d):
    """Trapezoidal membership function for one scalar input."""
    if x < a or x > d:
        return 0.0
    if a == b and x <= b:
        left = 1.0
    elif a < x < b:
        left = (x - a) / float(b - a)
    else:
        left = 1.0 if x >= b else 0.0

    if c == d and x >= c:
        right = 1.0
    elif c < x < d:
        right = (d - x) / float(d - c)
    else:
        right = 1.0 if x <= c else 0.0

    return float(np.clip(min(left, right), 0.0, 1.0))


def trimf_scalar(x, a, b, c):
    """Triangular membership function for one scalar input."""
    if x < a or x > c:
        return 0.0
    if x == b:
        return 1.0
    if a < x < b:
        return float((x - a) / float(b - a))
    if b < x < c:
        return float((c - x) / float(c - b))
    return 0.0


def input_memberships(temp_error, heat_load):
    """Common antecedent membership grades used by all three FIS models."""
    temperature = {
        "Safe": trapmf_scalar(temp_error, -5.0, -5.0, 0.0, 2.0),
        "Warm": trimf_scalar(temp_error, 1.0, 5.0, 9.0),
        "Hot": trapmf_scalar(temp_error, 7.0, 10.0, 20.0, 20.0),
    }
    load = {
        "Low": trapmf_scalar(heat_load, 0.0, 0.0, 20.0, 40.0),
        "Medium": trimf_scalar(heat_load, 25.0, 50.0, 75.0),
        "High": trapmf_scalar(heat_load, 60.0, 80.0, 100.0, 100.0),
    }
    return temperature, load


def firing_strengths(temp_error, heat_load):
    """Return the common rule firing strengths using the minimum t-norm."""
    temp_mf, load_mf = input_memberships(temp_error, heat_load)
    fired = []
    for temp_term, load_term, consequent in RULES:
        alpha = min(temp_mf[temp_term], load_mf[load_term])
        fired.append((temp_term, load_term, consequent, float(alpha)))
    return fired


# ---------------------------------------------------------------------------
# Membership functions for array-valued output grids
# ---------------------------------------------------------------------------


def trapmf_array(x, a, b, c, d):
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)

    if b > a:
        mask = (x > a) & (x < b)
        y[mask] = (x[mask] - a) / float(b - a)
    else:
        y[x <= b] = 1.0

    y[(x >= b) & (x <= c)] = 1.0

    if d > c:
        mask = (x > c) & (x < d)
        y[mask] = (d - x[mask]) / float(d - c)
    else:
        y[x >= c] = 1.0

    return np.clip(y, 0.0, 1.0)


def trimf_array(x, a, b, c):
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)

    if b > a:
        mask = (x >= a) & (x <= b)
        y[mask] = (x[mask] - a) / float(b - a)
    if c > b:
        mask = (x >= b) & (x <= c)
        y[mask] = np.maximum(y[mask], (c - x[mask]) / float(c - b))

    y[x == b] = 1.0
    return np.clip(y, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Numerical integration compatible with old and new NumPy
# ---------------------------------------------------------------------------


def trapezoid_integral(values, x):
    """Numerical integral without np.trapezoid or np.trapz."""
    values = np.asarray(values, dtype=float)
    x = np.asarray(x, dtype=float)
    if values.size != x.size or values.size < 2:
        raise ValueError("values and x must have the same length and contain at least two points")
    return float(np.sum(0.5 * (values[:-1] + values[1:]) * np.diff(x)))


# ---------------------------------------------------------------------------
# Three inference architectures
# ---------------------------------------------------------------------------


def mamdani(temp_error, heat_load):
    """Mamdani inference with max aggregation and centroid defuzzification."""
    y = np.linspace(0.0, 100.0, 5001)
    output_mf = {
        "Low": trapmf_array(y, 0.0, 0.0, 20.0, 40.0),
        "Moderate": trimf_array(y, 25.0, 50.0, 75.0),
        "High": trapmf_array(y, 60.0, 80.0, 100.0, 100.0),
    }

    aggregated = np.zeros_like(y)
    for _, _, consequent, alpha in firing_strengths(temp_error, heat_load):
        aggregated = np.maximum(
            aggregated,
            np.minimum(alpha, output_mf[consequent])
        )

    area = trapezoid_integral(aggregated, y)
    if area <= 1e-12:
        return 0.0

    moment = trapezoid_integral(y * aggregated, y)
    return float(moment / area)


def sugeno(temp_error, heat_load):
    """Zero-order Takagi-Sugeno inference with constant consequents."""
    z = {
        "Low": 20.0,
        "Moderate": 50.0,
        "High": 80.0,
    }
    fired = firing_strengths(temp_error, heat_load)
    denominator = sum(alpha for _, _, _, alpha in fired)
    if denominator <= 1e-12:
        return 0.0

    numerator = sum(alpha * z[consequent] for _, _, consequent, alpha in fired)
    return float(numerator / denominator)


def tsukamoto_rule_output(consequent, alpha):
    """Invert each monotonic Tsukamoto consequent membership function."""
    # Low: monotonic non-increasing on the full universe; it reaches zero at 40%.
    if consequent == "Low":
        return 40.0 * (1.0 - alpha)

    # Moderate: monotonic non-decreasing; transition spans 20% to 70%.
    if consequent == "Moderate":
        return 20.0 + 50.0 * alpha

    # High: monotonic non-decreasing; transition spans 60% to 100%.
    return 60.0 + 40.0 * alpha


def tsukamoto(temp_error, heat_load):
    """Tsukamoto inference with monotonic consequents and weighted averaging."""
    fired = firing_strengths(temp_error, heat_load)
    denominator = sum(alpha for _, _, _, alpha in fired)
    if denominator <= 1e-12:
        return 0.0

    numerator = 0.0
    for _, _, consequent, alpha in fired:
        z_i = tsukamoto_rule_output(consequent, alpha)
        numerator += alpha * z_i

    return float(numerator / denominator)


def all_methods(temp_error, heat_load):
    return {
        "Mamdani": mamdani(temp_error, heat_load),
        "Sugeno": sugeno(temp_error, heat_load),
        "Tsukamoto": tsukamoto(temp_error, heat_load),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def figure_5_temperature_memberships():
    x = np.linspace(-5.0, 20.0, 1000)
    safe = trapmf_array(x, -5.0, -5.0, 0.0, 2.0)
    warm = trimf_array(x, 1.0, 5.0, 9.0)
    hot = trapmf_array(x, 7.0, 10.0, 20.0, 20.0)

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.plot(x, safe, linewidth=2.0, label="Safe")
    ax.plot(x, warm, linewidth=2.0, label="Warm")
    ax.plot(x, hot, linewidth=2.0, label="Hot")
    ax.set_xlabel("Battery temperature error, ΔT (°C)")
    ax.set_ylabel("Membership degree")
    ax.set_ylim(-0.03, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "Figure_5_Temperature_Error_MFs.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure_6_heat_load_memberships():
    x = np.linspace(0.0, 100.0, 1000)
    low = trapmf_array(x, 0.0, 0.0, 20.0, 40.0)
    medium = trimf_array(x, 25.0, 50.0, 75.0)
    high = trapmf_array(x, 60.0, 80.0, 100.0, 100.0)

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.plot(x, low, linewidth=2.0, label="Low")
    ax.plot(x, medium, linewidth=2.0, label="Medium")
    ax.plot(x, high, linewidth=2.0, label="High")
    ax.set_xlabel("Normalized heat load (%)")
    ax.set_ylabel("Membership degree")
    ax.set_ylim(-0.03, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "Figure_6_Heat_Load_MFs.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure_7_consequents():
    y = np.linspace(0.0, 100.0, 1000)
    low = trapmf_array(y, 0.0, 0.0, 20.0, 40.0)
    moderate = trimf_array(y, 25.0, 50.0, 75.0)
    high = trapmf_array(y, 60.0, 80.0, 100.0, 100.0)

    # Tsukamoto consequents must be monotonic over the full output universe.
    ts_low = np.where(y <= 40.0, 1.0 - y / 40.0, 0.0)
    ts_moderate = np.where(y <= 20.0, 0.0, np.minimum(1.0, (y - 20.0) / 50.0))
    ts_high = np.where(y <= 60.0, 0.0, np.minimum(1.0, (y - 60.0) / 40.0))

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.8))

    axes[0].plot(y, low, linewidth=2.0, label="Low")
    axes[0].plot(y, moderate, linewidth=2.0, label="Moderate")
    axes[0].plot(y, high, linewidth=2.0, label="High")
    axes[0].set_title("Mamdani")
    axes[0].set_xlabel("Cooling demand (%)")
    axes[0].set_ylabel("Membership degree")
    axes[0].grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].scatter([20.0, 50.0, 80.0], [0.5, 0.5, 0.5], s=80)
    axes[1].hlines(0.5, 20, 80, linewidth=1.5)
    axes[1].text(20, 0.58, "Low: z=20", ha="center", fontsize=9)
    axes[1].text(50, 0.42, "Moderate: z=50", ha="center", fontsize=9)
    axes[1].text(80, 0.58, "High: z=80", ha="center", fontsize=9)
    axes[1].set_xlim(0, 100)
    axes[1].set_ylim(0, 1)
    axes[1].set_yticks([])
    axes[1].set_xlabel("Cooling demand (%)")
    axes[1].set_title("Sugeno (zero-order)")
    axes[1].grid(alpha=0.2)

    axes[2].plot(y, ts_low, linewidth=2.0, linestyle="--", label="Low (decreasing)")
    axes[2].plot(y, ts_moderate, linewidth=2.0, linestyle="-.", label="Moderate (increasing)")
    axes[2].plot(y, ts_high, linewidth=2.0, linestyle=":", label="High (increasing)")
    axes[2].set_title("Tsukamoto")
    axes[2].set_xlabel("Cooling demand (%)")
    axes[2].set_ylabel("Membership degree")
    axes[2].set_ylim(-0.03, 1.05)
    axes[2].grid(alpha=0.2)
    axes[2].legend(frameon=False, fontsize=7)

    fig.suptitle("Consequent representation used by the three inference architectures", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "Figure_7_All_Three_FIS_Consequents.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def calculate_all_scenarios():
    rows = []
    for name, temp_error, heat_load in SCENARIOS:
        values = all_methods(temp_error, heat_load)
        range_value = max(values.values()) - min(values.values())
        rows.append({
            "Scenario": name,
            "TemperatureError_C": temp_error,
            "HeatLoad_pct": heat_load,
            "Mamdani": values["Mamdani"],
            "Sugeno": values["Sugeno"],
            "Tsukamoto": values["Tsukamoto"],
            "Range": range_value,
        })
    return rows


def figure_8_heatmap(rows):
    labels = [r["Scenario"].replace("S1: ", "S1\n")
              .replace("S2: ", "S2\n")
              .replace("S3: ", "S3\n")
              .replace("S4: ", "S4\n")
              .replace("S5: ", "S5\n")
              .replace("S6: ", "S6\n") for r in rows]

    matrix = np.array([
        [r["Mamdani"], r["Sugeno"], r["Tsukamoto"]]
        for r in rows
    ])

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    image = ax.imshow(matrix, aspect="auto", interpolation="nearest")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Mamdani", "Sugeno", "Tsukamoto"])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Inference architecture")
    ax.set_ylabel("Scenario")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=9)

    ax.set_title("Cooling-demand outputs across six deterministic scenarios")
    fig.colorbar(image, ax=ax, label="Cooling demand (%)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "Figure_8_Cooling_Demand_Heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure_9_response_curve():
    errors = np.linspace(0.0, 15.0, 151)
    heat_load = 50.0
    mam = []
    sug = []
    tsu = []

    for e in errors:
        vals = all_methods(float(e), heat_load)
        mam.append(vals["Mamdani"])
        sug.append(vals["Sugeno"])
        tsu.append(vals["Tsukamoto"])

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.plot(errors, mam, linewidth=2.0, label="Mamdani")
    ax.plot(errors, sug, linewidth=2.0, label="Sugeno")
    ax.plot(errors, tsu, linewidth=2.0, label="Tsukamoto")
    ax.set_xlabel("Battery temperature error, ΔT (°C)")
    ax.set_ylabel("Cooling demand (%)")
    ax.set_title("Response to temperature-error changes at fixed heat load = 50%")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "Figure_9_Response_Curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_results(rows):
    with open(str(CSV_PATH), "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "Scenario",
            "TemperatureError_C",
            "HeatLoad_pct",
            "Mamdani",
            "Sugeno",
            "Tsukamoto",
            "Range",
        ])
        for r in rows:
            writer.writerow([
                r["Scenario"],
                "{:.4f}".format(r["TemperatureError_C"]),
                "{:.4f}".format(r["HeatLoad_pct"]),
                "{:.4f}".format(r["Mamdani"]),
                "{:.4f}".format(r["Sugeno"]),
                "{:.4f}".format(r["Tsukamoto"]),
                "{:.4f}".format(r["Range"]),
            ])

    # Representative overlapping case: four rules fire simultaneously.
    rep_temp_error = 1.5
    rep_heat_load = 30.0
    fired = firing_strengths(rep_temp_error, rep_heat_load)
    with open(str(RULE_CSV_PATH), "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["TemperatureError_C", "HeatLoad_pct", "Rule", "Temperature", "Load", "Consequent", "FiringStrength"])
        rule_id = 1
        for temp_term, load_term, consequent, alpha in fired:
            if alpha > 0.0:
                writer.writerow([
                    "{:.4f}".format(rep_temp_error),
                    "{:.4f}".format(rep_heat_load),
                    "R{}".format(rule_id),
                    temp_term,
                    load_term,
                    consequent,
                    "{:.4f}".format(alpha),
                ])
            rule_id += 1


def print_results(rows):
    print("\nScenario                           TempErr(C)  HeatLoad(%)   Mamdani   Sugeno  Tsukamoto   Range")
    print("-" * 98)
    for r in rows:
        print(
            "{:<35s} {:>9.2f}   {:>10.2f}   {:>8.2f}  {:>7.2f}    {:>8.2f}  {:>7.2f}".format(
                r["Scenario"],
                r["TemperatureError_C"],
                r["HeatLoad_pct"],
                r["Mamdani"],
                r["Sugeno"],
                r["Tsukamoto"],
                r["Range"],
            )
        )
    print("\nResults saved to: {}".format(CSV_PATH))
    print("Figures saved to: {}".format(FIG_DIR))


def main():
    # Core calculations
    rows = calculate_all_scenarios()
    save_results(rows)

    # Reproducible figures
    figure_5_temperature_memberships()
    figure_6_heat_load_memberships()
    figure_7_consequents()
    figure_8_heatmap(rows)
    figure_9_response_curve()

    # One consolidated terminal output
    print_results(rows)


if __name__ == "__main__":
    main()
