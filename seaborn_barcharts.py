import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_palette("Set2")
sns.set(font_scale=2.2, rc={"text.usetex": False}, style="ticks", font="STIXGeneral")


def show_values_on_bars(axs, h_v="v", space_x=5.9, space_y=-0.25):
    def _show_on_single_plot(ax):
        if h_v == "v":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height()
                value = "{.2f}".format(p.get_height())
                ax.text(_x, _y, value, ha="center")
        elif h_v == "h":
            for p in ax.patches:
                _x = p.get_x() + float(space_x)
                _y = p.get_y() + p.get_height() + float(space_y)
                value = "{0:.2f}".format(float(p.get_width()))
                ax.text(_x, _y, value, ha="left", color="white")

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)


df_200 = pd.DataFrame(
    {
        "labels": ["saved human effort", "test accuracy", "combined score"],
        "percentage": [93.63, 69.68, 79.90],
    }
)

df_1500 = pd.DataFrame(
    {
        "labels": ["saved human effort", "test accuracy", "combined score"],
        "percentage": [55.30, 86.40, 67.43],
    }
)


df_comparison_weak = pd.DataFrame(
    {
        "labels": ["saved human effort", "test accuracy", "combined score"],
        "No Weak": [0, 86.71, 0],
        "WeakClust": [65.03, 86.32, 73.81],
        "WeakCert": [51.14, 86.26, 64.21],
        "Both": [55.3, 86.4, 67.43],
    }
)

df_comparison_all = pd.DataFrame(
    {
        "labels": [
            "saved human effort",
            "test accuracy",
            "combined score",
            "global_score",
        ],
        "DWTC": [52.8, 85.7, 65.35, 78.3],
        "IBN_SINA": [85.76, 91.32, 88.46, 92.35],
        "HIVA": [97.83, 96.66, 97.24, 92.29],
        "ORANGE": [99.31, 98.11, 98.7, 91.72],
        "SYLVA": [98.28, 98.81, 98.54, 96.38],
        "ZEBRA": [96.73, 95.90, 96.32, 92.41],
    }
)


def create_barplot(df, title):
    fig = plt.figure()
    b = sns.barplot(x="percentage", y="labels", data=df)
    b.set(xticks=[0, 50, 100])
    #  b.set()
    b.set(ylabel=None)

    b.spines["top"].set_visible(False)
    b.spines["right"].set_visible(False)
    #  b.spines["bottom"].set_visible(False)
    #  b.spines["left"].set_visible(False)

    #  b.set_yticklabels(b.get_yticklabels(), weight="bold")

    show_values_on_bars(b, "h")

    plt.tight_layout()
    fig.savefig(
        "/home/julius/win_transfer/ds-active_learning/fig/" + title + ".pdf",
        verbose=True,
    )
    plt.clf()


create_barplot(df_1500, "dwtc_1500")
create_barplot(df_200, "dwtc_200")
#  create_barplot(df_comparison_all, "compare_all")
#  create_barplot(df_comparison_weak, "compare_weak")
