import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
from parse_sat3 import load_all_cases


def plot_scan(
    df,
    x_col,
    pure_letters,
    title,
    xlabel,
    ylabel,
    output_path,
    is_x_log=False,
    species_list=["H", "D", "T"]
):
    fig, ax = plt.subplots(figsize=(6, 4.5))

    species_colors = {
        "H": "#1f77b4",
        "D": "#2ca02c",
        "T": "#d62728"
    }

    df_pure = df[df["letter"].isin(pure_letters)]
    df_other = df[~df["case"].isin(df_pure["case"])]

    for spec in species_list:
        sub_other = df_other[df_other["species_type_2"] == spec]
        color = species_colors.get(spec, "#7f7f7f")
        ax.scatter(
            sub_other[x_col],
            sub_other["Q_i"],
            color=color,
            alpha=0.25,
            s=35,
            edgecolors="none"
        )

    for spec in species_list:
        sub_pure = df_pure[df_pure["species_type_2"] == spec].sort_values(x_col)
        color = species_colors.get(spec, "#7f7f7f")
        if not sub_pure.empty:
            ax.plot(
                sub_pure[x_col],
                sub_pure["Q_i"],
                color=color,
                marker="o",
                markersize=6,
                linewidth=1.8,
                label=spec
            )

    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel(xlabel, fontsize=10.5)
    ax.set_ylabel(ylabel, fontsize=10.5)
    ax.grid(True, linestyle="-", alpha=0.3, color="#d3d3d3")

    if is_x_log:
        ax.set_xscale("log")

    legend_elements = []
    for spec in species_list:
        legend_elements.append(
            Line2D(
                [0],
                [0],
                color=species_colors.get(spec, "#7f7f7f"),
                marker="o",
                markersize=6,
                linewidth=1.8,
                label=spec
            )
        )
    legend_elements.append(
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#b0b0b0",
            markersize=6,
            alpha=0.5,
            label="other inputs\nalso changed"
        )
    )

    legend_title = "Ion species\n(lines: only drive changed)" if len(species_list) > 1 else "Deuterium scan\n(lines: only drive changed)"
    ax.legend(
        handles=legend_elements,
        title=legend_title,
        loc="center left",
        bbox_to_anchor=(1.04, 0.5),
        frameon=True,
        fontsize=9,
        title_fontsize=9.5
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    dataset_path = Path("SAT3_dataset/data")
    df, _ = load_all_cases(dataset_path)
    df["letter"] = df["case"].apply(lambda x: x.split("_")[0])

    out_dir = Path("plots")
    out_dir.mkdir(exist_ok=True, parents=True)

    plot_scan(
        df=df,
        x_col="RLTS_2",
        pure_letters=["b", "c", "a", "d"],
        title="Ion heat flux vs. temperature gradient",
        xlabel="Ion temperature gradient $R/L_{T_i}$",
        ylabel="$Q_i$ (GB units)",
        output_path=out_dir / "qi_vs_temperature_gradient.png",
        is_x_log=False,
        species_list=["H", "D", "T"]
    )

    plot_scan(
        df=df,
        x_col="RLNS_2",
        pure_letters=["a", "e", "f"],
        title="Ion heat flux vs. density gradient",
        xlabel="Ion density gradient $R/L_{n_i}$",
        ylabel="$Q_i$ (GB units)",
        output_path=out_dir / "qi_vs_density_gradient.png",
        is_x_log=False,
        species_list=["H", "D", "T"]
    )

    plot_scan(
        df=df,
        x_col="XNUE",
        pure_letters=["j", "a"],
        title="Ion heat flux vs. collision frequency (Baseline drive)",
        xlabel="Electron collision frequency $\\nu_{ei}$",
        ylabel="$Q_i$ (GB units)",
        output_path=out_dir / "qi_vs_collision_frequency_baseline.png",
        is_x_log=True,
        species_list=["H", "D", "T"]
    )

    plot_scan(
        df=df,
        x_col="XNUE",
        pure_letters=["x", "y", "f", "z"],
        title="Ion heat flux vs. collision frequency ($R/L_n = 3$)",
        xlabel="Electron collision frequency $\\nu_{ei}$",
        ylabel="$Q_i$ (GB units)",
        output_path=out_dir / "qi_vs_collision_frequency_steep_density.png",
        is_x_log=True,
        species_list=["H", "D", "T"]
    )

    plot_scan(
        df=df,
        x_col="Q_PRIME_LOC",
        pure_letters=["g", "h", "a", "i"],
        title="Ion heat flux vs. magnetic shear (Deuterium)",
        xlabel="Magnetic shear derivative $dq/dr$",
        ylabel="$Q_i$ (GB units)",
        output_path=out_dir / "qi_vs_magnetic_shear_deuterium.png",
        is_x_log=False,
        species_list=["D"]
    )

    plot_scan(
        df=df,
        x_col="KAPPA_LOC",
        pure_letters=["a", "p", "q", "r"],
        title="Ion heat flux vs. plasma elongation (Deuterium)",
        xlabel="Plasma elongation $\\kappa$",
        ylabel="$Q_i$ (GB units)",
        output_path=out_dir / "qi_vs_elongation_deuterium.png",
        is_x_log=False,
        species_list=["D"]
    )

    plot_scan(
        df=df,
        x_col="RMIN_LOC",
        pure_letters=["w", "a", "v"],
        title="Ion heat flux vs. minor radius (Deuterium)",
        xlabel="Minor radius $r/a$",
        ylabel="$Q_i$ (GB units)",
        output_path=out_dir / "qi_vs_minor_radius_deuterium.png",
        is_x_log=False,
        species_list=["D"]
    )

    print("Generated 7 plots in plots/ directory.")


if __name__ == "__main__":
    main()
