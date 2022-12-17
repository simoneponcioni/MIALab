import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import symmetric_boundary_dice as sbd
from coregister_arrays import register
import timeit
import pandas as pd


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


def calc_sbd(gt, pred, slice_=150):
    gt_mask = np.zeros(gt.shape)
    gt_mask[gt == 1] = 1
    pred_mask = np.zeros(pred.shape)
    pred_mask[pred == 1] = 1
    metric = sbd.SBD_metric()
    return metric.Symmetric_Boundary_Dice(gt[:, :, slice_], pred[:, :, slice_])


def main():
    slices_array = np.arange(0, 181, 1)
    sbd_array = np.zeros(len(slices_array))
    res = []
    subject_list = [
        "117122",
        "118528",
        "118730",
        "118932",
        "120111",
        "122317",
        "122620",
        "123117",
        "123925",
        "124422",
    ]
    df_tot_res = pd.DataFrame()

    for subject in subject_list:
        for label_s in range(1, 6):
            # fmt: off
            basepath = Path("/home/simoneponcioni/Documents/03_LECTURES/MIALab/")
            pred_path = (basepath / f"bin/mia-result/2022-12-13-11-56-12/masks/{subject}_SEG-PP/{subject}_SEG-PP_mask_{label_s}.mha")
            gt_path = f"/home/simoneponcioni/Documents/03_LECTURES/MIALab/data/labels_transformed/{subject}.nii.gz"
            # fmt: on
            gt = import_gt(gt_path, label_=label_s)
            pred = import_pred(pred_path)
            pred_reg = register(gt, pred)

            for i, slice_s in enumerate(slices_array):
                sbd = calc_sbd(gt, pred_reg, slice_=slice_s)
                sbd_array[i] = sbd

            sbd_arr_no_nans = sbd_array[~np.isnan(sbd_array)]
            # mask values < 0.1
            # sbd_arr_abv_0d1 = sbd_arr_no_nans[sbd_arr_no_nans > 0.1]
            sbd_arr_mean = np.mean(sbd_arr_no_nans)
            sbd_arr_stdev = np.std(sbd_arr_no_nans)

            res.append([subject, label_s, sbd_arr_mean, sbd_arr_stdev])
            df = pd.DataFrame(res, columns=["SUBJECT", "LABEL", "SBD", "SBD_STDEV"])
            replacement_dict = {
                1: "Amygdala",
                2: "GreyMatter",
                3: "Hippocampus",
                4: "Thalamus",
                5: "WhiteMatter",
            }
            df["LABEL"] = df["LABEL"].replace(replacement_dict)
        print(df)

        # append df
        df_tot_res = pd.concat([df_tot_res, df], ignore_index=True)
        # save csv
        df_tot_res.to_csv(
            "/home/simoneponcioni/Documents/03_LECTURES/MIALab/bin/mia-result/results_sbd_all.csv",
            index=False,
        )


if __name__ == "__main__":
    st = timeit.Timer()
    timeit.Timer(main)
    execution_time = timeit.timeit(main, number=1)
    print(f"CPU Execution time:\t{execution_time} (s)")
