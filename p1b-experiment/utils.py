from time import sleep
import matplotlib.pyplot as plt
import numpy as np


def go_to_sleep(seconds=20):
    print("The kernel is going to sleep...")
    t = 0
    try:
        while t < seconds:
            if t % 5 == 0:
                print("zzz.", end="")
            else:
                print(".", end="")
            sleep(1)
            t += 1
        print()
    except KeyboardInterrupt:
        print()
        print(f"You woke the kernel after {t} seconds.")

    finally:
        print("The kernel is awake.")


def _plot_residuals(ax, y):
    ax.errorbar(
        np.arange(y.size),
        y,
        yerr=1,
        fmt="o",
        color="green",
        capsize=2,
        elinewidth=1.5,
        ecolor="black",
        capthick=1.5,
        markeredgecolor="black",
        label="data",
        zorder=0,
    )
    ax.axhline(
        0, color="r", linestyle="--", linewidth="2.0", label="best-fit curve", zorder=1
    )


def residuals_examples(n_points=50):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    ax1.set_ylabel("Residuals")
    ax2.set_ylabel("Residuals")
    ax3.set_ylabel("Residuals")
    ax3.set_xlabel("x")

    y0 = np.random.normal(size=n_points) - 0.5

    ax1.set_title("linearly increasing residuals")
    _plot_residuals(ax1, y0 + np.linspace(-3, 3, n_points))

    ax2.set_title("oscillating residuals")
    _plot_residuals(ax2, y0 + 3 * np.sin(np.linspace(-10, 10, n_points)))

    ax3.set_title("no obvious structure")
    _plot_residuals(ax3, y0)

    fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    residuals_with_structure()
