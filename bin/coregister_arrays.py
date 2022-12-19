import SimpleITK as sitk
import numpy as np


def register(fixed_array, moving_array):
    fixed_image = sitk.GetImageFromArray(fixed_array)
    moving_image = sitk.GetImageFromArray(moving_array)
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_image,
        moving_image,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=100)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=1000,
        estimateLearningRate=registration_method.Once,
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    try:
        final_transform = registration_method.Execute(fixed_image, moving_image)
        transformed_image = sitk.Resample(
            moving_image,
            fixed_image,
            final_transform,
            sitk.sitkLinear,
            0.0,
            moving_image.GetPixelID(),
        )
    except RuntimeError:
        print("Error during registration")
        return moving_array
    return sitk.GetArrayFromImage(transformed_image)
