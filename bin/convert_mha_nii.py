# convert mha to nii using SimpleITK
import SimpleITK as sitk
from pathlib import Path
import numpy as np
import glob


def convert_mha_nii(res_dir):
    res_dir_nii = Path(res_dir) / "nii"
    res_dir_nii.mkdir(exist_ok=True)
    result = glob.glob(f"{res_dir}/*.mha")

    for i in result:
        print(i)
        i = Path(i)
        i_stem = Path(Path(i).stem)
        sitk.WriteImage(sitk.ReadImage(str(i)), str(res_dir_nii / f"{i_stem}.nii.gz"))


def create_mha_masks(res_dir):
    masks_dir = Path(res_dir) / "masks"
    masks_dir.mkdir(exist_ok=True)
    result = glob.glob(f"{res_dir}/*.mha")

    for img in result:
        print(img)
        img_sitk = sitk.ReadImage(img)
        img_np = sitk.GetArrayFromImage(img_sitk)
        img = Path(img)
        img_stem = img.stem
        Path(img.parent / "masks" / img_stem).mkdir(exist_ok=True)
        img_stem = Path(Path(img).stem)

        for m in range(1, 5):
            mask = np.zeros(img_np.shape)
            mask[img_np == m] = 1
            mask_name = f"{img_stem}_mask_{m}.mha"
            mask_path = img.parent / "masks" / img_stem / mask_name
            sitk.WriteImage(sitk.GetImageFromArray(mask), str(mask_path))


res_dir = "bin/mia-result/2022-12-06-16-50-06"


def main():
    # convert_mha_nii(res_dir)
    create_mha_masks(res_dir)


if __name__ == "__main__":
    print("Running main")
    main()
