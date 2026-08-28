# -*- coding: utf-8 -*-
"""
Fuzzy Logic Comparative Application
Mamdani vs. Zero-Order Takagi-Sugeno vs. Tsukamoto

Controlled, reproducible numerical demonstration for the manuscript:
"Fuzzy Logic Theory and a Reproducible Three-FIS Application".

The application uses two directly specified engineering-model inputs:
    1) Battery temperature error, ΔT (°C)
    2) Normalized load indicator, Q (%)

Output:
    Cooling-demand index, Y (%), with linguistic levels Low, Moderate, High.

All three inference architectures use the same:
    - antecedent membership functions,
    - nine-rule knowledge base,
    - minimum t-norm for AND,
    - deterministic input scenarios.

Only the consequent representation and final output formation differ.

Compatibility:
    Designed for Python 3.x and legacy NumPy/Matplotlib installations.
    It intentionally does not use np.trapezoid or np.trapz.

Outputs generated in one execution:
    - scenario_comparison.csv
    - representative_rule_activation.csv
    - figures/Figure_14_Temperature_Error_MFs.png
    - figures/Figure_15_Normalized_Load_MFs.png
    - figures/Figure_16_Three_FIS_Consequents.png
    - figures/Figure_17_Cooling_Demand_Heatmap.png
    - figures/Figure_18_Response_Curves.png
"""

from __future__ import print_function

import csv
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

# Deterministic operating points used only to compare the three FIS architectures.
SCENARIOS = [
    ("S1: Stable", 1.0, 15.0),
    ("S2: Moderate thermal rise", 4.5, 45.0),
    ("S3: High normalized load", 8.0, 75.0),
    ("S4: Severe thermal stress", 14.0, 90.0),
    ("S5: High temperature error + medium load", 10.5, 50.0),
    ("S6: Warm temperature error + low load", 6.0, 20.0),
]

TEMP_DOMAIN = (0.0, 15.0)
LOAD_DOMAIN = (0.0, 100.0)
OUTPUT_DOMAIN = (0.0, 100.0)

BASE_DIR = Path(__file__).resolve().parent
FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)
CSV_PATH = BASE_DIR / "scenario_comparison.csv"
RULE_CSV_PATH = BASE_DIR / "representative_rule_activation.csv"


# ---------------------------------------------------------------------------
# Scalar membership functions
# ---------------------------------------------------------------------------

def trapmf_scalar(x, a, b, c, d):
    """Evaluate a trapezoidal membership function at one scalar x."""
    x = float(x)
    if x < a or x > d:
        return 0.0

    if a == b:
        left = 1.0 if x >= b else 0.0
    elif x <= a:
        left = 0.0
    elif x < b:
        left = (x - a) / float(b - a)
    else:
        left = 1.0

    if c == d:
        right = 1.0 if x <= c else 0.0
    elif x >= d:
        right = 0.0
    elif x > c:
        right = (d - x) / float(d - c)
    else:
        right = 1.0

    return float(np.clip(min(left, right), 0.0, 1.0))


def trimf_scalar(x, a, b, c):
    """Evaluate a triangular membership function at one scalar x."""
    x = float(x)
    if x < a or x > c:
        return 0.0
    if x == b:
        return 1.0
    if a < x < b:
        return float((x - a) / float(b - a))
    if b < x < c:
        return float((c - x) / float(c - b))
    return 0.0


def input_memberships(temp_error, normalized_load):
    """Return the common antecedent membership grades."""
    temperature = {
        "Safe": trapmf_scalar(temp_error, -5.0, -5.0, 0.0, 2.0),
        "Warm": trimf_scalar(temp_error, 1.0, 5.0, 9.0),
        "Hot": trapmf_scalar(temp_error, 7.0, 10.0, 20.0, 20.0),
    }
    load = {
        "Low": trapmf_scalar(normalized_load, 0.0, 0.0, 20.0, 40.0),
        "Medium": trimf_scalar(normalized_load, 25.0, 50.0, 75.0),
        "High": trapmf_scalar(normalized_load, 60.0, 80.0, 100.0, 100.0),
    }
    return temperature, load


def firing_strengths(temp_error, normalized_load):
    """Compute the nine common firing strengths using the minimum t-norm."""
    temp_mf, load_mf = input_memberships(temp_error, normalized_load)
    fired = []
    for rule_index, (temp_term, load_term, consequent) in enumerate(RULES, start=1):
        alpha = min(temp_mf[temp_term], load_mf[load_term])
        fired.append((rule_index, temp_term, load_term, consequent, float(alpha)))
    return fired


# ---------------------------------------------------------------------------
# Array membership functions and numerical integration
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


def trapezoid_integral(values, x):
    """Compatibility-safe trapezoidal integration without NumPy integration aliases."""
    values = np.asarray(values, dtype=float)
    x = np.asarray(x, dtype=float)
    if values.size != x.size or values.size < 2:
        raise ValueError("values and x must have the same length and contain at least two points")
    return float(np.sum(0.5 * (values[:-1] + values[1:]) * np.diff(x)))


# ---------------------------------------------------------------------------
# Three inference architectures
# ---------------------------------------------------------------------------

def mamdani(temp_error, normalized_load):
    """Mamdani inference: min implication, max aggregation, centroid output."""
    y = np.linspace(OUTPUT_DOMAIN[0], OUTPUT_DOMAIN[1], 5001)
    output_mf = {
        "Low": trapmf_array(y, 0.0, 0.0, 20.0, 40.0),
        "Moderate": trimf_array(y, 25.0, 50.0, 75.0),
        "High": trapmf_array(y, 60.0, 80.0, 100.0, 100.0),
    }

    aggregated = np.zeros_like(y)
    for _, _, _, consequent, alpha in firing_strengths(temp_error, normalized_load):
        aggregated = np.maximum(
            aggregated,
            np.minimum(alpha, output_mf[consequent])
        )

    area = trapezoid_integral(aggregated, y)
    if area <= 1e-12:
        return 0.0

    moment = trapezoid_integral(y * aggregated, y)
    return float(moment / area)


def sugeno(temp_error, normalized_load):
    """Zero-order Takagi-Sugeno inference with constant consequents."""
    z = {
        "Low": 20.0,
        "Moderate": 50.0,
        "High": 80.0,
    }

    fired = firing_strengths(temp_error, normalized_load)
    denominator = sum(alpha for _, _, _, _, alpha in fired)
    if denominator <= 1e-12:
        return 0.0

    numerator = sum(
        alpha * z[consequent]
        for _, _, _, consequent, alpha in fired
    )
    return float(numerator / denominator)


def tsukamoto_rule_output(consequent, alpha):
    """Inverse mapping for the monotonic Tsukamoto consequents."""
    alpha = float(np.clip(alpha, 0.0, 1.0))

    # Low: monotonic non-increasing on [0, 40].
    if consequent == "Low":
        return 40.0 * (1.0 - alpha)

    # Moderate: monotonic non-decreasing on [20, 70].
    if consequent == "Moderate":
        return 20.0 + 50.0 * alpha

    # High: monotonic non-decreasing on [60, 100].
    if consequent == "High":
        return 60.0 + 40.0 * alpha

    raise ValueError("Unknown consequent: {}".format(consequent))


def tsukamoto(temp_error, normalized_load):
    """Tsukamoto inference: inverse monotonic consequent + weighted average."""
    fired = firing_strengths(temp_error, normalized_load)
    denominator = sum(alpha for _, _, _, _, alpha in fired)
    if denominator <= 1e-12:
        return 0.0

    numerator = 0.0
    for _, _, _, consequent, alpha in fired:
        rule_value = tsukamoto_rule_output(consequent, alpha)
        numerator += alpha * rule_value

    return float(numerator / denominator)


def all_methods(temp_error, normalized_load):
    return {
        "Mamdani": mamdani(temp_error, normalized_load),
        "Sugeno": sugeno(temp_error, normalized_load),
        "Tsukamoto": tsukamoto(temp_error, normalized_load),
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_model():
    """Run structural and numerical checks before generating publication outputs."""
    # Check rule completeness.
    if len(RULES) != 9:
        raise AssertionError("The common rule base must contain exactly nine rules.")

    # Check consequent labels.
    valid_consequents = {"Low", "Moderate", "High"}
    if any(rule[2] not in valid_consequents for rule in RULES):
        raise AssertionError("Invalid rule consequent detected.")

    # Check output range for a dense deterministic grid.
    test_errors = np.linspace(0.0, 15.0, 31)
    test_loads = np.linspace(0.0, 100.0, 21)
    for error in test_errors:
        for load in test_loads:
            values = all_methods(float(error), float(load))
            for name, value in values.items():
                if not (OUTPUT_DOMAIN[0] - 1e-8 <= value <= OUTPUT_DOMAIN[1] + 1e-8):
                    raise AssertionError("{} output out of declared domain: {}".format(name, value))

    # Check membership and firing grades remain in [0,1].
    for error in test_errors:
        for load in test_loads:
            temp_mf, load_mf = input_memberships(float(error), float(load))
            for grade in list(temp_mf.values()) + list(load_mf.values()):
                if not (-1e-12 <= grade <= 1.0 + 1e-12):
                    raise AssertionError("Membership grade outside [0,1].")
            for _, _, _, _, alpha in firing_strengths(float(error), float(load)):
                if not (-1e-12 <= alpha <= 1.0 + 1e-12):
                    raise AssertionError("Firing strength outside [0,1].")

    # Check Tsukamoto monotonicity on the complete output universe.
    y = np.linspace(OUTPUT_DOMAIN[0], OUTPUT_DOMAIN[1], 5001)
    low = np.where(y <= 40.0, 1.0 - y / 40.0, 0.0)
    moderate = np.where(y <= 20.0, 0.0, np.minimum(1.0, (y - 20.0) / 50.0))
    high = np.where(y <= 60.0, 0.0, np.minimum(1.0, (y - 60.0) / 40.0))

    if np.any(np.diff(low) > 1e-10):
        raise AssertionError("Tsukamoto Low consequent is not non-increasing.")
    if np.any(np.diff(moderate) < -1e-10):
        raise AssertionError("Tsukamoto Moderate consequent is not non-decreasing.")
    if np.any(np.diff(high) < -1e-10):
        raise AssertionError("Tsukamoto High consequent is not non-decreasing.")


# ---------------------------------------------------------------------------
# Publication figures
# ---------------------------------------------------------------------------

def figure_14_temperature_memberships():
    x = np.linspace(-5.0, 20.0, 1200)
    safe = trapmf_array(x, -5.0, -5.0, 0.0, 2.0)
    warm = trimf_array(x, 1.0, 5.0, 9.0)
    hot = trapmf_array(x, 7.0, 10.0, 20.0, 20.0)

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.plot(x, safe, linewidth=2.0, label="Safe")
    ax.plot(x, warm, linewidth=2.0, label="Warm")
    ax.plot(x, hot, linewidth=2.0, label="Hot")
    ax.set_xlabel("Battery temperature error, ΔT (°C)")
    ax.set_ylabel("Membership degree")
    ax.set_xlim(-5.0, 20.0)
    ax.set_ylim(-0.03, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "Figure_14_Temperature_Error_MFs.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure_15_load_memberships():
    x = np.linspace(0.0, 100.0, 1200)
    low = trapmf_array(x, 0.0, 0.0, 20.0, 40.0)
    medium = trimf_array(x, 25.0, 50.0, 75.0)
    high = trapmf_array(x, 60.0, 80.0, 100.0, 100.0)

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.plot(x, low, linewidth=2.0, label="Low")
    ax.plot(x, medium, linewidth=2.0, label="Medium")
    ax.plot(x, high, linewidth=2.0, label="High")
    ax.set_xlabel("Normalized load indicator, Q (%)")
    ax.set_ylabel("Membership degree")
    ax.set_xlim(0.0, 100.0)
    ax.set_ylim(-0.03, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "Figure_15_Normalized_Load_MFs.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure_16_consequents():
    y = np.linspace(0.0, 100.0, 1200)
    low = trapmf_array(y, 0.0, 0.0, 20.0, 40.0)
    moderate = trimf_array(y, 25.0, 50.0, 75.0)
    high = trapmf_array(y, 60.0, 80.0, 100.0, 100.0)

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
    axes[0].set_ylim(-0.03, 1.05)
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

    fig.suptitle("Consequent representations used by the three inference architectures", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "Figure_16_Three_FIS_Consequents.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure_17_heatmap(rows):
    labels = [r["Scenario"].replace(": ", "\n", 1) for r in rows]
    matrix = np.array([[r["Mamdani"], r["Sugeno"], r["Tsukamoto"]] for r in rows])

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    image = ax.imshow(matrix, aspect="auto", interpolation="nearest")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Mamdani", "Sugeno", "Tsukamoto"])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Inference architecture")
    ax.set_ylabel("Deterministic scenario")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, "{:.2f}".format(matrix[i, j]), ha="center", va="center", fontsize=9)

    ax.set_title("Cooling-demand outputs across six deterministic scenarios")
    fig.colorbar(image, ax=ax, label="Cooling-demand index (%)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "Figure_17_Cooling_Demand_Heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure_18_response_curve():
    errors = np.linspace(0.0, 15.0, 151)
    normalized_load = 50.0
    mam = []
    sug = []
    tsu = []

    for error in errors:
        values = all_methods(float(error), normalized_load)
        mam.append(values["Mamdani"])
        sug.append(values["Sugeno"])
        tsu.append(values["Tsukamoto"])

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.plot(errors, mam, linewidth=2.0, label="Mamdani")
    ax.plot(errors, sug, linewidth=2.0, label="Sugeno")
    ax.plot(errors, tsu, linewidth=2.0, label="Tsukamoto")
    ax.set_xlabel("Battery temperature error, ΔT (°C)")
    ax.set_ylabel("Cooling-demand index (%)")
    ax.set_title("Response to temperature-error changes at fixed Q = 50%")
    ax.set_xlim(0.0, 15.0)
    ax.set_ylim(0.0, 105.0)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "Figure_18_Response_Curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Results and reproducibility files
# ---------------------------------------------------------------------------

def calculate_all_scenarios():
    rows = []
    for name, temp_error, normalized_load in SCENARIOS:
        values = all_methods(temp_error, normalized_load)
        spread = max(values.values()) - min(values.values())
        rows.append({
            "Scenario": name,
            "TemperatureError_C": temp_error,
            "NormalizedLoad_pct": normalized_load,
            "Mamdani": values["Mamdani"],
            "Sugeno": values["Sugeno"],
            "Tsukamoto": values["Tsukamoto"],
            "OutputSpread": spread,
        })
    return rows


def save_results(rows):
    with open(str(CSV_PATH), "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "Scenario",
            "TemperatureError_C",
            "NormalizedLoad_pct",
            "Mamdani",
            "Sugeno",
            "Tsukamoto",
            "OutputSpread",
        ])
        for row in rows:
            writer.writerow([
                row["Scenario"],
                "{:.4f}".format(row["TemperatureError_C"]),
                "{:.4f}".format(row["NormalizedLoad_pct"]),
                "{:.4f}".format(row["Mamdani"]),
                "{:.4f}".format(row["Sugeno"]),
                "{:.4f}".format(row["Tsukamoto"]),
                "{:.4f}".format(row["OutputSpread"]),
            ])

    # Worked overlap example used in the manuscript.
    representative_error = 1.5
    representative_load = 30.0
    fired = firing_strengths(representative_error, representative_load)

    with open(str(RULE_CSV_PATH), "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "TemperatureError_C",
            "NormalizedLoad_pct",
            "Rule",
            "Temperature",
            "Load",
            "Consequent",
            "FiringStrength",
        ])
        for rule_id, temp_term, load_term, consequent, alpha in fired:
            if alpha > 0.0:
                writer.writerow([
                    "{:.4f}".format(representative_error),
                    "{:.4f}".format(representative_load),
                    "R{}".format(rule_id),
                    temp_term,
                    load_term,
                    consequent,
                    "{:.4f}".format(alpha),
                ])


def print_results(rows):
    print("\nScenario                                      ΔT(C)   Q(%)   Mamdani   Sugeno  Tsukamoto  Spread")
    print("-" * 105)
    for row in rows:
        print(
            "{:<46s} {:>6.2f}  {:>6.2f}  {:>8.2f}  {:>7.2f}  {:>9.2f}  {:>7.2f}".format(
                row["Scenario"],
                row["TemperatureError_C"],
                row["NormalizedLoad_pct"],
                row["Mamdani"],
                row["Sugeno"],
                row["Tsukamoto"],
                row["OutputSpread"],
            )
        )

    print("\nRepresentative overlapping case: ΔT = 1.50 °C, Q = 30.00%")
    print("Rule  Temperature  Load    Consequent  Firing strength")
    print("-" * 62)
    for rule_id, temp_term, load_term, consequent, alpha in firing_strengths(1.5, 30.0):
        if alpha > 0.0:
            print("R{:<4d}{:<13s}{:<8s}{:<12s}{:>8.4f}".format(
                rule_id, temp_term, load_term, consequent, alpha
            ))

    print("\nResults CSV: {}".format(CSV_PATH))
    print("Rule-activation CSV: {}".format(RULE_CSV_PATH))
    print("Figures directory: {}".format(FIG_DIR))


def main():
    validate_model()
    rows = calculate_all_scenarios()
    save_results(rows)

    figure_14_temperature_memberships()
    figure_15_load_memberships()
    figure_16_consequents()
    figure_17_heatmap(rows)
    figure_18_response_curve()

    print_results(rows)
    print("\nValidation: PASSED")


if __name__ == "__main__":
    main()
