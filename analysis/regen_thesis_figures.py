"""Redraw every figure used in the thesis from saved results, without rerunning the studies
(after a change of style, e.g. the colour rule of analysis/palette.py).

    python -m analysis.regen_thesis_figures
    python firmware/tools/memory_figure.py ...   (fig20, needs the firmware's linker map)
"""
import pandas as pd

from analysis import figures, figures4, figures_key, figures_motor_model, figures_evaluation, longitudinal


def main():
    figures.fig_motor_signature()            # fig1
    figures4.fig_swap_map()                  # fig7
    figures_key.fig_autotest_days()          # fig8
    figures_key.fig_temperature()            # fig9
    figures_key.fig_real_boots()             # fig13
    out = longitudinal.OUT                   # fig15, fig16
    longitudinal.figures(pd.read_csv(out / "genuine.csv"), pd.read_csv(out / "impostors.csv"),
                         pd.read_csv(out / "speed_window.csv"))
    figures_motor_model.main()               # fig17, fig18, fig19
    figures_evaluation.main()                # fig21
    print("done")


if __name__ == "__main__":
    main()
