# convert mha to nii using SimpleITK
import SimpleITK as sitk
from pathlib import Path
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


res_dir = "bin/mia-result/2022-12-06-16-50-06"
# res_dir = "bin/mia-result/2022-11-06-16-12-39"


def main():
    convert_mha_nii(res_dir)


if __name__ == "__main__":
    print("Running main")
    main()
