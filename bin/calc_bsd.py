import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import symmetric_boundary_dice as sbd


def import_gt(img_path, label_=int(2)):
    img_sitk = sitk.ReadImage(str(Path(img_path).resolve()))
    img_np = sitk.GetArrayFromImage(img_sitk)
    mask = np.zeros(img_np.shape)
    mask[img_np == label_] = 1
    return mask


def import_pred(img_path):
    img_sitk = sitk.ReadImage(str(Path(img_path).resolve()))
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
    slice_s = 100
    label_s = int(5)
    basepath = Path("/home/simoneponcioni/Documents/03_LECTURES/MIALab/")
    pred_path = (
        basepath
        / f"bin/mia-result/2022-12-06-16-50-06/masks/118932_SEG-PP/118932_SEG-PP_mask_{label_s}.mha"
    )
    gt_path = "/home/simoneponcioni/Documents/03_LECTURES/MIALab/data/test/118932/labels_native.nii.gz"
    gt = import_gt(gt_path, label_=label_s)
    pred = import_pred(pred_path)
    plot_gt_pred(gt, pred, basepath, slice_=slice_s)
    bsd = calc_bsd(gt, pred, slice_=slice_s)
    print(bsd)


if __name__ == "__main__":
    main()
