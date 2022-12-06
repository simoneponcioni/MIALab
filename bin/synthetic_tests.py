import symmetric_boundary_dice as sbd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


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
    return sbd_sym


def plot_evolution(dx, sbd_arr):
    plt.figure(figsize=(10, 10))
    plt.plot(dx, sbd_arr, color="tab:grey", linewidth=2)
    plt.show()


def circles_dxdy(dx, radius_a, c1_a, c2_a):
    sbd_arr = []
    radius_b = radius_a
    for i, d in enumerate(dx):
        # iteratively create circles with increasing distance
        c1_b = 500 + int(d)
        c2_b = 500 + int(d)
        sbd_sym = test_circles(i, c1_a, c2_a, c1_b, c2_b, radius_a, radius_b)
        sbd_arr.append(sbd_sym)
    return sbd_arr


def circles_dr(dr, radius_a, c1_a, c2_a):
    sbd_arr = []
    c1_b = c1_a
    c2_b = c2_a

    for i, d in enumerate(dr):
        # iteratively create circles with decreasing radius
        radius_b = radius_a - d
        sbd_sym = test_circles(i, c1_a, c2_a, c1_b, c2_b, radius_a, radius_b)
        sbd_arr.append(sbd_sym)
    return sbd_arr


if __name__ == "__main__":
    # code for running evaluation of SBD metric with increasing distance
    dx = np.linspace(0, 150, 5)
    radius_a = 100
    c1_a = 500
    c2_a = 500
    sbd_arr = circles_dxdy(dx, radius_a, c1_a, c2_a)
    plot_evolution(dx, sbd_arr)

    # code for running evaluation of SBD metric with decreasing radius
    dr = np.linspace(0, 50, 5)
    print(dr)
    radius_a = 100
    c1_a = 500
    c2_a = 500
    sbd_arr = circles_dr(dr, radius_a, c1_a, c2_a)
    plot_evolution(dr, sbd_arr)
