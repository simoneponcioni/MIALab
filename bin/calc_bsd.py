import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import symmetric_boundary_dice as sbd


def import_gt(img_path, label_=int(2)):
    img_sitk = sitk.ReadImage(str(Path(img_path).resolve()))
    spacing = img_sitk.GetSpacing()
    direction = img_sitk.GetDirection()
    origin = img_sitk.GetOrigin()
    img_np = sitk.GetArrayFromImage(img_sitk)
    mask = np.zeros(img_np.shape)
    mask[img_np == label_] = 1
    return mask, spacing, direction, origin


def import_pred(img_path, gt_spacing, gt_direction, gt_origin):
    img_sitk = sitk.ReadImage(str(Path(img_path).resolve()))
    img_sitk.SetSpacing(gt_spacing)
    img_sitk.SetDirection(gt_direction)
    img_sitk.SetOrigin(gt_origin)
    return sitk.GetArrayFromImage(img_sitk)


def plot_gt_pred(gt, pred, basepath, slice_=150):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(gt[:, :, slice_], cmap="gray")
    axes[1].imshow(pred[:, :, slice_], cmap="gray")
    filepath = basepath / "bin/mia-result/2022-12-06-16-50-06/117122_gt_pred"
    plt.savefig(str(Path(filepath).resolve()) + ".png")


def calc_bsd(gt, pred, slice_=150):
    gt_mask = np.zeros(gt.shape)
    gt_mask[gt == 1] = 1
    pred_mask = np.zeros(pred.shape)
    pred_mask[pred == 1] = 1
    metric = sbd.SBD_metric()
    return metric.Symmetric_Boundary_Dice(gt[:, :, slice_], pred[:, :, slice_])


def main():
    slice_s = 150
    basepath = Path("/home/simoneponcioni/Documents/03_LECTURES/MIALab/")
    pred_path = (
        basepath
        / "bin/mia-result/2022-12-06-16-50-06/masks/117122_SEG-PP/117122_SEG-PP_mask_2.mha"
    )
    gt_path = "/home/simoneponcioni/Documents/03_LECTURES/MIALab/data/test/117122/labels_native.nii.gz"
    gt, gt_spacing, gt_direction, gt_origin = import_gt(gt_path, label_=2)
    pred = import_pred(pred_path, gt_spacing, gt_direction, gt_origin)
    plot_gt_pred(gt, pred, basepath, slice_=slice_s)
    bsd = calc_bsd(gt, pred, slice_=slice_s)
    print(bsd)


if __name__ == "__main__":
    main()
