import symmetric_boundary_dice as sbd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from medpy import metric
import matplotlib
import timeit

# rcParams and settings
f = 1
plt.rcParams["figure.figsize"] = [10 * f, 10 * f]
plt.rcParams["font.size"] = 15
matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"


def hd(preds, targets, hd95=True):
    # checking if voxels are empty.
    if not np.any(preds) or np.all(preds) or not np.any(targets) or np.all(targets):
        haussdorf_dist = 0
    else:
        haussdorf_dist = metric.hd95(preds, targets, voxelspacing=None, connectivity=1)
    return haussdorf_dist


def create_circular_mask(h, w, center=None, radius=None):
    if center is None:  # use the middle of the image
        center = (int(w / 2), int(h / 2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    mask = dist_from_center <= radius
    return mask


def create_cv2_circle(x_a, y_a, x_b, y_b, radius_a, radius_b, i, show_plot=True):
    """Create a circle with the given thickness, center, size, and color.
    Args:
        thickness (float): Thickness of the circle.
        center (tuple): Center of the circle.
        size (tuple): Size of the circle.
        color (tuple): Color of the circle.
    Returns:
        cv2.circle: A circle with the given thickness, center, size, and color.
    """
    background = 0 * np.ones((2 * 500, 2 * 500))
    center_a = (x_a, y_a)
    center_b = (x_b, y_b)
    dist_c = np.sqrt((x_a - x_b) ** 2 + (y_a - y_b) ** 2)

    h, w = background.shape[:2]
    mask_a = create_circular_mask(h, w, center=center_a, radius=radius_a)
    mask_b = create_circular_mask(h, w, center=center_b, radius=radius_b)
    circle_a = background.copy()
    circle_b = background.copy()
    circle_a[mask_a] = 1
    circle_b[mask_b] = 1

    if show_plot is not False:
        plt.figure(figsize=(10, 10))
        plot_combined = circle_a + circle_b
        plt.imshow(plot_combined, cmap="gist_gray", vmin=0, vmax=2)
        plt.title(f"Offset circle = {dist_c:.1f} px", weight="bold")
        plt.colorbar()
        savepath = r"sbd_results"
        filepath = Path(savepath, f"circle_{i}.png")
        plt.savefig(filepath)
        plt.show()
    return circle_a, circle_b


def test_circles(i, c1_a, c2_a, c1_b, c2_b, radius_a, radius_b):
    """
    Test if the circle is created correctly.
    """
    arr_a, arr_b = create_cv2_circle(
        c1_a, c2_a, c1_b, c2_b, radius_a, radius_b, i, show_plot=True
    )
    metric = sbd.SBD_metric()
    sbd_sym = metric.Symmetric_Boundary_Dice(arr_a, arr_b)

    dice_coef = np.mean(metric.single_dice_coefficient(arr_a, arr_b))
    HD95_coef = hd(arr_a, arr_b)
    return sbd_sym, dice_coef, HD95_coef


def plot_evolution(dx, sbd_arr, dice_arr, title):
    plt.figure(figsize=(10, 10))
    plt.title(f"Metric evolution with increasing distance", weight="bold")
    plt.plot(dx, sbd_arr, color="tab:grey", linewidth=2, label="SBD")
    plt.plot(dx, dice_arr, color="tab:green", linewidth=2, label="Dice")
    plt.xlim(0)
    plt.ylim(0)
    plt.xlabel("Distance between circles (px)")
    plt.ylabel("Metric (-)")
    plt.legend(loc="upper right", fontsize=14)
    plt.tight_layout()
    fig_path = Path("sbd_results", f"sbd_vs_dice_synthetic_evolution_{title}.png")
    plt.savefig(fig_path, dpi=150)


def plot_evolution_HD(dx, HD95_arr, title):
    plt.figure(figsize=(10, 10))
    plt.title(f"Metric evolution with increasing distance", weight="bold")
    plt.plot(dx, HD95_arr, color="tab:red", linewidth=2, label="HD95")
    plt.xlim(0)
    plt.ylim(0)
    plt.xlabel("Distance between circles (px)")
    plt.ylabel("Hausdorff Distance (px)")
    plt.legend(loc="upper right", fontsize=14)
    plt.tight_layout()
    fig_path = Path("sbd_results" + f"HD95_synthetic_evolution_{title}.png")
    plt.savefig(fig_path, dpi=150)


def circles_dxdy(dx, radius_a, c1_a, c2_a):
    sbd_arr = []
    dice_arr = []
    HD95_arr = []
    radius_b = radius_a
    for i, d in enumerate(dx):
        # iteratively create circles with increasing distance
        c1_b = 500 + int(d)
        c2_b = 500 + int(d)
        sbd_sym, dice_co, HD95_co = test_circles(
            i, c1_a, c2_a, c1_b, c2_b, radius_a, radius_b
        )

        sbd_arr.append(sbd_sym)
        dice_arr.append(dice_co)
        HD95_arr.append(HD95_co)
        print(f"2D Dice coefficient:\t\t{dice_co:.5f}")
    return (
        sbd_arr,
        dice_arr,
        HD95_arr,
    )


def circles_dr(dr, radius_a, c1_a, c2_a):
    sbd_arr = []
    dice_arr = []
    HD95_arr = []
    c1_b = c1_a
    c2_b = c2_a

    for i, d in enumerate(dr):
        # iteratively create circles with decreasing radius
        radius_b = radius_a - d
        sbd_sym, dice_co, HD95_co = test_circles(
            i, c1_a, c2_a, c1_b, c2_b, radius_a, radius_b
        )
        sbd_arr.append(sbd_sym)
        dice_arr.append(dice_co)
        HD95_arr.append(HD95_co)
    return (
        sbd_arr,
        dice_arr,
        HD95_arr,
    )


if __name__ == "__main__":
    # code for running evaluation of SBD, dice metric with increasing distance
    #  dx = np.linspace(0, 150, 10)
    #  radius_a = 100
    #  c1_a = 500
    #  c2_a = 500

    dx = np.linspace(0, 150, 10)
    radius_a = 100
    c1_a = 500
    c2_a = 500
    radius_b = radius_a
    c1_b = 500
    c2_b = 500
    i = 1
    arr_a, arr_b = create_cv2_circle(
        c1_a, c2_a, c1_b, c2_b, radius_a, radius_b, i, show_plot=False
    )

    def test_setup_sbd():
        metric = sbd.SBD_metric()
        metric.Symmetric_Boundary_Dice(arr_a, arr_b)

    times = timeit.repeat(test_setup_sbd, repeat=100, number=1)
    # Calculate the average execution time
    average_time = sum(times) / len(times)
    print(f"Average execution time SBD: {average_time:.5f} seconds")

    def test_setup_dice():
        metric = sbd.SBD_metric()
        metric.single_dice_coefficient(arr_a, arr_b)

    times = timeit.repeat(test_setup_dice, repeat=100, number=1)
    # Calculate the average execution time
    average_time = sum(times) / len(times)
    print(f"Average execution time DICE: {average_time:.5f} seconds")

    def test_setup_hd():
        hd(arr_a, arr_b)

    times = timeit.repeat(test_setup_hd, repeat=100, number=1)
    # Calculate the average execution time
    average_time = sum(times) / len(times)
    print(f"Average execution time HD95: {average_time:.5f} seconds")

    # sbd_arr, dice_arr, HD95_arr = circles_dxdy(dx, radius_a, c1_a, c2_a)
    # plot_evolution(dx, sbd_arr, dice_arr, title="circles_dxdy")
    # plot_evolution_HD(dx, HD95_arr, title="circles_dxdy")

    # # code for running evaluation of SBD, dice metric with decreasing radius
    # dr = np.linspace(0, 50, 10)
    # print(dr)
    # radius_a = 100
    # c1_a = 500
    # c2_a = 500
    # sbd_arr, dice_arr, HD95_arr = circles_dr(dr, radius_a, c1_a, c2_a)
    # plot_evolution(dr, sbd_arr, dice_arr, title="circles_dr")
    # plot_evolution_HD(dr, HD95_arr, title="circles_dr")
