import symmetric_boundary_dice as sbd
import cv2
from nose.tools import assert_equal, assert_less_equal
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk


def create_cv2_rectangle(
    background, center_x, center_y, width, heigth, thickness, color
):
    """Create a rectangle with the given thickness, center, size, and color.
    Args:
        thickness (float): Thickness of the rectangle.
        center (tuple): Center of the rectangle.
        size (tuple): Size of the rectangle.
        color (tuple): Color of the rectangle.
    Returns:
        cv2.rectangle: A rectangle with the given thickness, center, size, and color.
    """
    c1, c2 = center_x, center_y
    a_l = (int(c1 - width / 2), int(c2 - heigth / 2))
    a_b = (int(c1 + width / 2), int(c2 + heigth / 2))
    rect = cv2.rectangle(background, a_l, a_b, color, thickness)
    return rect


# a = create_cv2_rectangle(background, c1, c2, 100, 130, thickness, 1)
# b = create_cv2_rectangle(background, c1 + 200, c2 + 200, 100, 130, thickness, 1)


def create_numpy_rectangle(
    c1_a, c2_a, c1_b, c2_b, width_a, height_a, width_b, height_b, show_plot=True
):
    """
    Create two numpy arrays with rectangles.

    Args:
        c1_a (int): center x of rectangle a
        c2_a (int): center y of rectangle a
        c1_b (int): center x of rectangle b
        c2_b (int): center y of rectangle b
        width_a (int): width of rectangle a
        height_a (int): height of rectangle a
        width_b (int): width of rectangle b
        height_b (int): height of rectangle b
        show_plot (bool, optional): Open graphical representation in external window. Defaults to True.

    Returns:
        arr_a, arr_b: Two numpy arrays with rectangles.
        arr=1 inside the rectangle, arr=0 outside the rectangle.
    """
    background = 0 * np.ones((2 * 500, 2 * 500))
    arr_a = np.copy(background)
    arr_a[
        int(c1_a - width_a / 2) : int(c1_a + width_a / 2),
        int(c2_a - height_a / 2) : int(c2_a + height_a / 2),
    ] = 1

    arr_b = np.copy(background)
    arr_b[
        int(c1_b - width_b / 2) : int(c1_b + width_b / 2),
        int(c2_b - height_b / 2) : int(c2_b + height_b / 2),
    ] = 1

    if show_plot is not False:
        plot_combined = arr_a + arr_b
        plt.imshow(plot_combined, cmap="gist_gray")
        plt.show()
    return arr_a, arr_b


def test_symmetric_boundary_dice_equal():
    """
    Same shape, same position, SBD=1
    """
    c1_a, c2_a = 500, 500
    c1_b, c2_b = c1_a, c2_a
    width_a, height_a = 100, 130
    width_b, height_b = width_a, height_a
    arr_a, arr_b = create_numpy_rectangle(
        c1_a, c2_a, c1_b, c2_b, width_a, height_a, width_b, height_b, show_plot=False
    )
    metric = sbd.SBD_metric()
    sbd_sym = metric.Symmetric_Boundary_Dice(arr_a, arr_b)
    assert_equal(sbd_sym, 1.0)


def test_symmetric_boundary_dice_offset():
    """
    Offset shape, rectangle b is shifted and should not touch rectangle a
    SBD=0
    """
    c1_a, c2_a = 500, 500
    c1_b, c2_b = c1_a + 200, c2_a + 200
    width_a, height_a = 100, 130
    width_b, height_b = width_a, height_a
    arr_a, arr_b = create_numpy_rectangle(
        c1_a, c2_a, c1_b, c2_b, width_a, height_a, width_b, height_b, show_plot=False
    )
    metric = sbd.SBD_metric()
    sbd_sym = metric.Symmetric_Boundary_Dice(arr_a, arr_b)
    assert_equal(sbd_sym, 0.0)


def test_symmetric_boundary_dice_less_than_one():
    """
    Same center, but rectangle b is smaller than rectangle a.
    SBD < 1
    """
    c1_a, c2_a = 500, 500
    c1_b, c2_b = c1_a, c2_a
    width_a, height_a = 300, 200
    width_b, height_b = 200, 100
    arr_a, arr_b = create_numpy_rectangle(
        c1_a, c2_a, c1_b, c2_b, width_a, height_a, width_b, height_b, show_plot=False
    )
    metric = sbd.SBD_metric()
    sbd_sym = metric.Symmetric_Boundary_Dice(arr_a, arr_b)
    assert_less_equal(sbd_sym, 1.0)
