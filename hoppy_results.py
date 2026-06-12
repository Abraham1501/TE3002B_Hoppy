
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ── Estilo global ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size":        11,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.35,
    "grid.linewidth":   0.5,
    "lines.linewidth":  1.8,
})

STANCE_COLOR = "#ede0ff"
FLIGHT_COLOR = "#e0f0ff"
C_HIP   = "#2176AE"
C_KNEE  = "#E05C2A"
C_FX    = "#2176AE"
C_FZ    = "#E05C2A"
C_WORLD = "#1A7A4A"
C_EST   = "#2176AE"
C_REAL  = "#aaaaaa"
TAU_MAX = 3.728


# ── Utilidades ─────────────────────────────────────────────────────────────────
def load(csv_path: str, t_max: float = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df["phase"] = df["phase"].str.upper().str.strip()
    if t_max is not None:
        df = df[df["t"] <= t_max]
    return df.reset_index(drop=True)


def shade_phases(ax, df):
    t = df["t"].values
    ph = df["phase"].values
    i = 0
    while i < len(t):
        j = i
        while j < len(t) and ph[j] == ph[i]:
            j += 1
        color = STANCE_COLOR if ph[i] == "STANCE" else FLIGHT_COLOR
        ax.axvspan(t[i], t[min(j, len(t)-1)], color=color, alpha=0.55, lw=0)
        i = j


def save(fig, out_dir: Path, name: str):
    p = out_dir / name
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {p}")


def xlabel(ax):
    ax.set_xlabel("Tiempo (s)", fontsize=10)


def phase_legend(fig):
    h = [mpatches.Patch(color=FLIGHT_COLOR, alpha=0.8, label="FLIGHT"),
         mpatches.Patch(color=STANCE_COLOR, alpha=0.8, label="STANCE")]
    fig.legend(handles=h, loc="lower center", ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, -0.01), framealpha=0.9)


# ── 1. Posiciones articulares ──────────────────────────────────────────────────
def plot_pos_articular(df, out_dir):
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    fig.suptitle("Posiciones articulares", fontweight="500")

    for ax, col, lbl, color, desc in [
        (axes[0], "hip_pos",  "Hip",  C_HIP,  "Posición hip  (rad)"),
        (axes[1], "knee_pos", "Knee", C_KNEE, "Posición knee  (rad)"),
    ]:
        shade_phases(ax, df)
        ax.plot(df["t"], df[col], color=color, label=lbl)
        ax.set_ylabel(desc, fontsize=10)
        ax.legend(fontsize=9, loc="upper right")

    xlabel(axes[-1])
    phase_legend(fig)
    fig.tight_layout()
    save(fig, out_dir, "1_posiciones_articulares.png")


# ── 2. Posiciones Cartesianas ──────────────────────────────────────────────────
def plot_pos_cartesian(df, out_dir):
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    fig.suptitle("Posiciones Cartesianas del pie", fontweight="500")

    panels = [
        ("foot_x",     C_FX,    "Posición X pie  (m)",           "Pie X (marco hip)"),
        ("foot_z",     C_FZ,    "Posición Z pie  (m)",           "Pie Z (marco hip)"),
        ("foot_z_world", C_WORLD, "Altura cuerpo sobre suelo  (m)", "Altura cuerpo (mundo)"),
    ]

    for ax, (col, color, ylabel, lbl) in zip(axes, panels):
        shade_phases(ax, df)
        y = df[col] - (1.0 if col == "foot_z_world" else 0.0)
        ax.plot(df["t"], y, color=color, label=lbl)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(fontsize=9, loc="upper right")

    axes[2].set_ylabel("Altura sobre suelo  (m)", fontsize=10)
    xlabel(axes[-1])
    phase_legend(fig)
    fig.tight_layout()
    save(fig, out_dir, "2_posiciones_cartesianas.png")


# ── 3. Velocidades articulares ─────────────────────────────────────────────────
def plot_vel_articular(df, out_dir):
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    fig.suptitle("Velocidades articulares — estimada vs. real", fontweight="500")

    pairs = [
        ("hip_vel_est",  "hip_vel_real",  "Hip",  axes[0]),
        ("knee_vel_est", "knee_vel_real", "Knee", axes[1]),
    ]

    for est_col, real_col, name, ax in pairs:
        shade_phases(ax, df)
        ax.plot(df["t"], df[real_col], color=C_REAL, lw=1.2, ls="--",
                label="Real (qvel)", alpha=0.7)
        ax.plot(df["t"], df[est_col], color=C_EST,
                label="Estimada (encoder)")
        ax.set_ylabel(f"ω {name}  (rad/s)", fontsize=10)
        ax.legend(fontsize=9, loc="upper right")

    xlabel(axes[-1])
    phase_legend(fig)
    fig.tight_layout()
    save(fig, out_dir, "3_velocidades_articulares.png")


# ── 4. Velocidades Cartesianas ─────────────────────────────────────────────────
def plot_vel_cartesian(df, out_dir):
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    fig.suptitle("Velocidades Cartesianas del pie", fontweight="500")

    for ax, col, lbl, color in [
        (axes[0], "foot_vx", "Velocidad X pie  (m/s)", C_FX),
        (axes[1], "foot_vz", "Velocidad Z pie  (m/s)", C_FZ),
    ]:
        shade_phases(ax, df)
        ax.plot(df["t"], df[col], color=color, label=lbl)
        ax.axhline(0, color="gray", lw=0.8, ls=":")
        ax.set_ylabel(lbl, fontsize=10)
        ax.legend(fontsize=9, loc="upper right")

    xlabel(axes[-1])
    phase_legend(fig)
    fig.tight_layout()
    save(fig, out_dir, "4_velocidades_cartesianas.png")


# ── 5. Fuerzas de contacto ─────────────────────────────────────────────────────
def plot_contact(df, out_dir):
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    fig.suptitle("Fuerzas de contacto", fontweight="500")

    shade_phases(axes[0], df)
    axes[0].plot(df["t"], df["Fn_sensor"], color="#1A7A4A",
                 label="Fn sensor  (N)")
    axes[0].set_ylabel("Fuerza normal  Fn  (N)", fontsize=10)
    axes[0].legend(fontsize=9, loc="upper right")

    shade_phases(axes[1], df)
    axes[1].plot(df["t"], df["foot_compression"] * 1e3,
                 color="#8E44AD", label="Compresión del pie  (mm)")
    axes[1].set_ylabel("Compresión  (mm)", fontsize=10)
    axes[1].legend(fontsize=9, loc="upper right")

    xlabel(axes[-1])
    phase_legend(fig)
    fig.tight_layout()
    save(fig, out_dir, "5_fuerzas_contacto.png")


# ── 6. Torques de control ──────────────────────────────────────────────────────
def plot_torques(df, out_dir):
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    fig.suptitle("Torques de control", fontweight="500")

    for ax, col, color, lbl in [
        (axes[0], "tau_hip",  C_HIP,  "Torque hip  (N·m)"),
        (axes[1], "tau_knee", C_KNEE, "Torque knee  (N·m)"),
    ]:
        shade_phases(ax, df)
        ax.plot(df["t"], df[col], color=color, label=lbl)
        ax.axhline( TAU_MAX, color="red", lw=1.0, ls="--", alpha=0.6,
                    label=f"+τ_max = {TAU_MAX} N·m")
        ax.axhline(-TAU_MAX, color="red", lw=1.0, ls="--", alpha=0.6,
                    label=f"−τ_max = {TAU_MAX} N·m")
        ax.set_ylabel(lbl, fontsize=10)
        ax.legend(fontsize=8, loc="upper right", ncol=2)

    xlabel(axes[-1])
    phase_legend(fig)
    fig.tight_layout()
    save(fig, out_dir, "6_torques_control.png")


# ── 7. Estados FSM ─────────────────────────────────────────────────────────────
def plot_fsm(df, out_dir):
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    fig.suptitle("Estados de la máquina híbrida (FSM)", fontweight="500")

    state_bin = (df["phase"] == "STANCE").astype(int)
    axes[0].fill_between(df["t"], state_bin, step="post",
                         color=STANCE_COLOR, alpha=0.9, label="STANCE")
    axes[0].fill_between(df["t"], 1 - state_bin, step="post",
                         color=FLIGHT_COLOR, alpha=0.9, label="FLIGHT")
    axes[0].set_yticks([0, 1])
    axes[0].set_yticklabels(["FLIGHT", "STANCE"], fontsize=10)
    axes[0].set_ylabel("Estado FSM", fontsize=10)
    axes[0].legend(fontsize=9, loc="upper right")
    axes[0].grid(False)

    shade_phases(axes[1], df)
    axes[1].plot(df["t"], df["Fn_sensor"], color="#1A7A4A", lw=1.5,
                 label="Fn sensor")
    axes[1].set_ylabel("Fuerza normal  (N)", fontsize=10)
    axes[1].legend(fontsize=9, loc="upper right")

    shade_phases(axes[2], df)
    axes[2].plot(df["t"], df["foot_z_world"] - 1.0,
                 color=C_WORLD, lw=1.5, label="Altura cuerpo")
    axes[2].set_ylabel("Altura sobre suelo  (m)", fontsize=10)
    axes[2].legend(fontsize=9, loc="upper right")

    xlabel(axes[-1])
    phase_legend(fig)
    fig.tight_layout()
    save(fig, out_dir, "7_fsm_estados.png")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="HOPPY v2 — Gráficas de resultados")
    ap.add_argument("--csv",   required=True, help="Archivo CSV de simulación")
    ap.add_argument("--out",   default="resultados",
                    help="Subcarpeta de salida (default: resultados/)")
    ap.add_argument("--t_max", type=float, default=None,
                    help="Recortar a t_max segundos")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nCSV: {args.csv}")
    print(f"Salida: {out_dir.resolve()}\n")

    df = load(args.csv, args.t_max)
    print(f"  {len(df)} muestras  |  "
          f"t=[{df['t'].min():.2f}, {df['t'].max():.2f}] s  |  "
          f"STANCE={( df['phase']=='STANCE').mean()*100:.1f}%\n")

    plot_pos_articular(df, out_dir)
    plot_pos_cartesian(df, out_dir)
    plot_vel_articular(df, out_dir)
    plot_vel_cartesian(df, out_dir)
    plot_contact(df, out_dir)
    plot_torques(df, out_dir)
    plot_fsm(df, out_dir)

    print(f"\nFiguras guardadas en '{out_dir}/'")


if __name__ == "__main__":
    main()