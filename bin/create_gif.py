import imageio
import SimpleITK as sitk
import numpy as np
import os
from pathlib import Path


def main():
    # Load the two SimpleITK images
    image1 = sitk.ReadImage(
        "/Users/simoneponcioni/Documents/ARTORG/03_Lectures/MIALabs/MIALab/data/labels_transformed/117122.nii.gz"
    )
    image2 = sitk.ReadImage(
        "/Users/simoneponcioni/Documents/ARTORG/03_Lectures/MIALabs/MIALab/bin/mia-result/2022-12-13-11-56-12/masks/117122_SEG-PP/117122_SEG-PP_mask_1.mha"
    )

    # create mask for image1 containing only label 1
    image1 = sitk.BinaryThreshold(image1, 1, 1, 1, 0)

    # Get the spacing of image2
    spacing = image2.GetSpacing()

    # Create the transform object
    transform = sitk.Transform()

    # Create the resampled image using the Resample function
    image1 = sitk.Resample(
        image1,
        image2.GetSize(),
        transform,
        sitk.sitkLinear,
        image2.GetOrigin(),
        spacing,
        image2.GetDirection(),
    )

    gif = "MIALab/sbd_results/animation.gif"
    # Get the number of slices in each image
    num_slices1 = image1.GetSize()[2]
    num_slices2 = image2.GetSize()[2]

    #  define correct format for RGB
    image1 = sitk.Cast(image1, sitk.sitkUInt8)
    image2 = sitk.Cast(image2, sitk.sitkUInt8)

    red = [255, 0, 0]
    blue = [0, 0, 255]

    # Create a temporary directory to store the images
    temp_dir = Path("MIALab/sbd_results/temp").mkdir(parents=True, exist_ok=True)
    temp_dir = str(Path("MIALab/sbd_results/temp").resolve())

    # Create an empty list to store the images for the GIF
    images = []
    # Loop through the slices of both images
    for i in range(max(num_slices1, num_slices2)):
        # red goes to the first label, green to second, blue to third
        # body_label=0, air_label=1, lung_label=2
        contour_image_1 = sitk.LabelToRGB(
            sitk.LabelContour(
                image1[:, :, i], fullyConnected=True, backgroundValue=255
            ),
            colormap=red,
            backgroundValue=255,
        )

        contour_image_2 = sitk.LabelToRGB(
            sitk.LabelContour(
                image2[:, :, i], fullyConnected=True, backgroundValue=255
            ),
            colormap=blue,
            backgroundValue=255,
        )
        slice1 = sitk.GetArrayViewFromImage(contour_image_1)
        slice2 = sitk.GetArrayViewFromImage(contour_image_2)

        # Stack the slices vertically to create a single image
        combined_slice = slice1 + slice2

        # Save the image to a temporary file
        file_name = f"slice_{i}.gif"
        file_path = os.path.join(temp_dir, file_name)
        imageio.imwrite(file_path, combined_slice.astype("uint8"), format="gif")

        # Read the saved image back into memory
        image = imageio.imread(file_path)

        # Add the image to the list
        images.append(image)

        # Save the GIF using imageio
        imageio.mimwrite("animation.gif", images, fps=10)


if __name__ == "__main__":
    main()
