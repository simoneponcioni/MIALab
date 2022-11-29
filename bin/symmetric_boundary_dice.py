import scipy as sp
import numpy as np
import cv2


class SBD_metric:
    def __init__(self):
        metric = str("Symmetric_Boundary_Dice")

    def single_dice_coefficient(self, y_true, y_pred_bin):
        # shape of y_true and y_pred_bin: (height, width)
        intersection = np.sum(y_true * y_pred_bin)
        if (np.sum(y_true) == 0) and (np.sum(y_pred_bin) == 0):
            return 1
        return (2 * intersection) / (np.sum(y_true) + np.sum(y_pred_bin))

    def Directional_Boundary_Dice(self, y_true, y_pred):
        """
        Directional Boundary Dice
        Edges found in the ground truth (y_true), Find 3x3 cube around these voxels
        and compute Dice for comparison with the voxels in the SAME SPACE in the segmented result (y_pred)
        """

        # Find edges
        imax = sp.ndimage.maximum_filter(y_true, size=3) != y_true
        imin = sp.ndimage.minimum_filter(y_true, size=3) != y_true
        icomb = np.logical_or(imax, imin)

        y_true_edges = np.where(icomb, y_true, 0)
        y_true_edges_idx = np.argwhere(y_true_edges)
        n_edge_points = len(y_true_edges_idx)

        true_subset = []
        pred_subset = []
        dice_results = np.empty(len(y_true_edges_idx))
        for dA in range(len(y_true_edges_idx)):
            true_subset = y_true[
                y_true_edges_idx[dA][0] - 1 : y_true_edges_idx[dA][0] + 2,
                y_true_edges_idx[dA][1] - 1 : y_true_edges_idx[dA][1] + 2,
            ]
            pred_subset = y_pred[
                y_true_edges_idx[dA][0] - 1 : y_true_edges_idx[dA][0] + 2,
                y_true_edges_idx[dA][1] - 1 : y_true_edges_idx[dA][1] + 2,
            ]
            true_subset = np.asarray(true_subset)
            pred_subset = np.asarray(pred_subset)
            dice_results[dA] = self.single_dice_coefficient(true_subset, pred_subset)

        DBD_sum = np.sum(dice_results.flatten())
        DBD = DBD_sum / n_edge_points
        return DBD, DBD_sum, n_edge_points

    def Symmetric_Boundary_Dice(self, y_true, y_pred):
        """
        Symmetric Boundary Dice
        Two way Directional Boundary Dice
        """
        true_DBD, true_sum_DBD, true_n_edge_points = self.Directional_Boundary_Dice(
            y_true, y_pred
        )
        pred_DBD, pred_sum_DBD, pred_n_edge_points = self.Directional_Boundary_Dice(
            y_pred, y_true
        )
        print(f"Directional Boundary Dice:\t{true_DBD:.5f}")

        symmetric_boundary_dice = (true_sum_DBD + pred_sum_DBD) / (
            true_n_edge_points + pred_n_edge_points
        )
        print(f"Symmetric Boundary Dice:\t{symmetric_boundary_dice:.5f}")
        return symmetric_boundary_dice

    def binarize(self, image):
        thresh, im_bw = cv2.threshold(image, 0.1, 1, cv2.THRESH_BINARY)
        return thresh, im_bw

    def get_y(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        binary = cv2.bitwise_not(gray)
        contours_arr = []
        contours, _ = cv2.findContours(
            binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        out = np.zeros_like(binary)
        outer_contour = cv2.drawContours(out, contours, -1, 1, 1)
        contours_arr.append(outer_contour)
        return contours_arr
