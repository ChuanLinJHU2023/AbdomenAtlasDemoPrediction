import numpy as np
import nibabel as nib
from scipy.ndimage import label, binary_fill_holes, binary_dilation, binary_erosion
from skimage.morphology import disk
import os
from scipy.ndimage import generate_binary_structure

# Mapping for vertebrae classes
class_map_part_vertebrae = {
    1: "vertebrae_L5",
    2: "vertebrae_L4",
    3: "vertebrae_L3",
    4: "vertebrae_L2",
    5: "vertebrae_L1",
    6: "vertebrae_T12",
    7: "vertebrae_T11",
    8: "vertebrae_T10",
    9: "vertebrae_T9",
    10: "vertebrae_T8",
    11: "vertebrae_T7",
    12: "vertebrae_T6",
    13: "vertebrae_T5",
    14: "vertebrae_T4",
    15: "vertebrae_T3",
    16: "vertebrae_T2",
    17: "vertebrae_T1",
    18: "vertebrae_C7",
    19: "vertebrae_C6",
    20: "vertebrae_C5",
    21: "vertebrae_C4",
    22: "vertebrae_C3",
    23: "vertebrae_C2",
    24: "vertebrae_C1"
}

def smooth_binary_image(binary_image, iterations=1):
    # Check the number of dimensions in the binary image
    num_dimensions = binary_image.ndim
    structure = generate_binary_structure(num_dimensions, 1)  # Structure for the number of dimensions

    # Dilation followed by erosion (opening)
    smoothed_image = binary_dilation(binary_image, structure=structure, iterations=iterations)
    smoothed_image = binary_erosion(smoothed_image, structure=structure, iterations=iterations)

    return smoothed_image

def process_an_image(binary_image):
    # Label connected components in the binary image
    labeled_array, num_features = label(binary_image)

    # Get sizes of all components
    sizes = np.array([np.sum(labeled_array == region) for region in range(1, num_features + 1)])

    # Check if there are any features found
    if sizes.size > 0:
        largest_component_label = np.argmax(sizes) + 1  # Find the label of the largest component
        largest_size_before = sizes[largest_component_label - 1]
    else:
        largest_component_label = None
        largest_size_before = 0

    largest_component = (labeled_array == largest_component_label).astype(np.uint8)

    # Fill holes in the largest component
    filled_component = binary_fill_holes(largest_component).astype(np.uint8)

    # Apply morphological smoothing
    smoothed_component = smooth_binary_image(filled_component)

    # Count the size of the smoothed component
    largest_size_after = np.sum(smoothed_component)
    return smoothed_component.astype(int), sizes, largest_size_before, largest_size_after


def process_images_in_directory(input_directory, output_directory):
    os.makedirs(output_directory, exist_ok=True)

    combined_image = None
    image_shape = None

    for filename in sorted(os.listdir(input_directory)):
        if filename.endswith('.nii.gz'):
            file_path = os.path.join(input_directory, filename)
            nii_img = nib.load(file_path)
            binary_image = nii_img.get_fdata()

            if image_shape is None:
                image_shape = binary_image.shape
                combined_image = np.zeros(image_shape, dtype=int)

            binary_image = (binary_image > 0).astype(int)

            processed_image, sizes, largest_size_before, filled_size = process_an_image(
                binary_image)

            processed_nifti_img = nib.Nifti1Image(processed_image.astype(np.uint8), nii_img.affine)
            processed_filename = filename.replace(".nii.gz", "") + "_refined" + ".nii.gz"
            nib.save(processed_nifti_img, os.path.join(output_directory, processed_filename))

            # Use the correct vertebra label from the filename
            vertebra_name = filename.replace(".nii.gz", "")
            vertebra_label = next((key for key, value in class_map_part_vertebrae.items() if value == vertebra_name), 0)

            combined_image[processed_image == 1] = vertebra_label

            print(f"Processing {filename}:")
            print(f"Sizes of different components: {sizes}")
            print(f"Size of the largest component before morphology: {largest_size_before}")
            print(f"Size of the largest component after morphology: {filled_size}\n")

    combined_nifti_img = nib.Nifti1Image(combined_image.astype(np.uint8), nii_img.affine)
    nib.save(combined_nifti_img, "combined_labels_refined.nii.gz")

def main():
    input_directory = './segmentations'  # Current directory or specify your input directory
    output_directory = './segmentations_refined'  # Specify your desired output directory

    process_images_in_directory(input_directory, output_directory)


main()