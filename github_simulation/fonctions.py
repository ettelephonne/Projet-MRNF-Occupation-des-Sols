import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import rotate, distance_transform_edt, gaussian_filter, uniform_filter
from osgeo import gdal
from skimage.morphology import erosion, square, skeletonize
from matplotlib.path import Path
from skimage.util import pad

def normalize_array(array, new_min, new_max):
    """
    Normalizes an array to a specified range [new_min, new_max].
    Parameters:
    - array: The array to be normalized.
    - new_min: The minimum value of the new range.
    - new_max: The maximum value of the new range.
    Returns:
    - The normalized array.
    """
    old_min = np.min(array)
    old_max = np.max(array)
    # Normalize array to [0, 1], then scale to [new_min, new_max].
    array = (array - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
    return array

def rotation(img):
    """
    Rotates an image by a random angle between 0 and 360 degrees.
    Parameters:
    - img: The image to be rotated.
    Returns:
    - The rotated image.
    """
    angle = np.random.randint(0, 360)  # Random angle for rotation.
    # Rotate the image without changing its shape.
    img = rotate(img, angle, reshape=False, order=0, mode='constant', cval=0.0)
    return img

def generate_shape(shape):
    """
    Generates a random geometric shape (circle, rectangle, triangle) on an image.
    Parameters:
    - shape: Tuple of the dimensions of the output image.
    Returns:
    - An image with a random shape drawn on it.
    """
    rand_shape = np.random.choice(['circle', 'rectangle', 'triangle'], p=[0.1, 0.85, 0.05])
    img = np.zeros((shape[0], shape[1], 4))  # Initialize a transparent image.
    
    if rand_shape == 'circle':
        # Generate a circle with random center and radius.
        center = (np.random.randint(shape[1] // 4, shape[1]), np.random.randint(shape[0] // 4, shape[0]))
        radius_x = np.random.randint(shape[1] // 5, shape[1] // 3 + 1)
        radius_y = np.random.randint(shape[0] // 5, shape[0] // 3 + 1)
        y, x = np.ogrid[-center[1]:shape[0] - center[1], -center[0]:shape[1] - center[0]]
        mask = (x * x) / (radius_x * radius_x) + (y * y) / (radius_y * radius_y) <= 1
        img[mask] = 1  # Fill the circle with white color.
    
    elif rand_shape == 'rectangle':
        # Generate a rectangle with random top left and bottom right corners.
        ratio = 0
        while ratio > 10 or ratio < 0.1:  # Ensure the rectangle is not too elongated.
            top_left = (np.random.randint(0, shape[1] // 2), np.random.randint(0, shape[0] // 2))
            bottom_right = (np.random.randint(top_left[0] + 1, shape[1]), np.random.randint(top_left[1] + 1, shape[0]))
            height = bottom_right[1] - top_left[1]
            width = bottom_right[0] - top_left[0]
            ratio = height / width
        img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = 1  # Fill the rectangle with white color.
    
    elif rand_shape == 'triangle':
        # Generate a triangle with random vertices.
        vertices = np.array([
            [np.random.randint(0, shape[1]), np.random.randint(0, shape[0])],
            [np.random.randint(0, shape[1]), np.random.randint(0, shape[0])],
            [np.random.randint(0, shape[1]), np.random.randint(0, shape[0])]
        ])
        grid = np.mgrid[0:shape[0], 0:shape[1]].reshape(2, -1).T
        mask = Path(vertices).contains_points(grid).reshape(shape)
        img[mask] = 1  # Fill the triangle with white color.
    
    return img

def index_from_dist(mask, dist):
    """
    Finds an index within a specified distance range from the non-masked areas.
    Parameters:
    - mask: The binary mask to analyze.
    - dist: A tuple (min_distance, max_distance) specifying the range of distances.
    Returns:
    - Coordinates (cx, cy) as indices within the specified distance range.
    """
    distance_from_road = distance_transform_edt(1 - mask)  # Compute distance transform.
    possible_indices = np.where(np.logical_and(distance_from_road >= dist[0], distance_from_road <= dist[1]))
    if possible_indices[0].size == 0:
        print('No available index!!!')
        # Fallback: choose a random index if no suitable one is found.
        cx, cy = np.random.randint(0, mask.shape[1]), np.random.randint(0, mask.shape[0])
    else:
        random_index = np.random.choice(possible_indices[0].shape[0])
        cx, cy = possible_indices[0][random_index], possible_indices[1][random_index]
    return cx, cy

def crop_image_to_data(img, value):
    """
    Crops the image to the smallest rectangle containing all data different from 'value'.
    Parameters:
    - img: The image to be cropped.
    - value: The value to be ignored (considered as background).
    Returns:
    - A tuple containing the coordinates of the cropped area.
    """
    # Find indices where the image has data.
    ys, xs = np.where(img[:, :] != value)
    # Calculate the bounding box of the data.
    xmin, xmax = np.min(xs), np.max(xs)
    ymin, ymax = np.min(ys), np.max(ys)
    # Return the coordinates of the bounding box.
    return [ymin, ymax+1, xmin, xmax+1]

def crop_bordures(image, bordure):
    """
    Crops the borders from an image based on specified border size.
    Parameters:
    - image: The image to be cropped.
    - bordure: A tuple specifying the border size to be removed.
    Returns:
    - The cropped image.
    """
    # Crop the image by removing specified border size from each side.
    if len(image.shape) == 3:
        image = image[bordure[0]:image.shape[0]-bordure[0], bordure[1]:image.shape[1]-bordure[1], :]
    else:
        image = image[bordure[0]:image.shape[0] - bordure[0], bordure[1]:image.shape[1] - bordure[1]]
    return image
def distortion(image):    
    # Generate borders around the image with specified dimensions, [300, 300] in this case.
    image = generate_bordures(image, [300, 300])
    
    # Obtain the shape of the first color channel of the image.
    rows, cols = image[:, :, 0].shape
    
    # Create a mask based on the first color channel, setting to 1 where the channel's value is greater than 0.
    mask = image[:, :, 0].copy() * 0
    mask = np.where(image[:, :, 0] > 0, 1, mask)
    
    # Determine the center of the area of interest within the mask using a custom function.
    cx, cy = index_from_dist(mask, [5, 100])
    
    # Randomly select an amplitude for the distortion effect.
    amplitude = np.random.randint(100, 400, 1)
    
    # Generate coordinate matrices for the entire image.
    i, j = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
    
    # Calculate the distance of each point from the center.
    distances = np.sqrt((i - cy) ** 2 + (j - cx) ** 2)
    
    # Calculate the offset for each point based on the distance and a random factor.
    rand_factor = np.random.randint(4, 16, 1)[0] / 10
    offsets = (amplitude * np.sin(distances / (rand_factor * cols))).astype(int)
    
    # Update the indices with the calculated offsets, ensuring they remain within the image bounds.
    new_i = np.clip(i + offsets, 0, rows - 1)
    new_j = np.clip(j + offsets, 0, cols - 1)
    
    # Apply the distortion to the image by mapping pixels to the new indices.
    img_output = image[new_i, new_j]
    
    # Create a mask for the distorted image, erode it, and apply a Gaussian filter for smoothing.
    mask_output = np.where(img_output[:, :, 0] > 0, 1, 0).astype(float)
    padded_mask = np.pad(mask_output, ((1, 1), (1, 1)), mode='constant', constant_values=0)
    eroded_padded_mask = erosion(padded_mask, square(3))
    mask_output = eroded_padded_mask[1:-1, 1:-1]
    mask_filter_output = gaussian_filter(mask_output, sigma=1)
    
    # Crop the output image and masks to the area containing data.
    ymin, ymax, xmin, xmax = crop_image_to_data(mask_output, 0)
    img_output = img_output[ymin:ymax, xmin:xmax]
    mask_output = mask_output[ymin:ymax, xmin:xmax]
    mask_filter_output = mask_filter_output[ymin:ymax, xmin:xmax]
    
    return img_output, mask_output, mask_filter_output

def generate_bordures(image, bordure, max_dim=1024):
    # Convert the border dimensions to an array for easier manipulation.
    bordure = np.asarray(bordure)
    
    # Calculate the potential new dimensions of the image with the border applied.
    new_height = image.shape[0] + 2 * bordure[0]
    new_width = image.shape[1] + 2 * bordure[1]
    
    # Adjust the border size if the new dimensions exceed the maximum allowed dimension.
    if new_height > max_dim:
        bordure[0] = (max_dim - image.shape[0]) // 2
    if new_width > max_dim:
        bordure[1] = (max_dim - image.shape[1]) // 2
    
    # Create the new image with adjusted dimensions and apply the original image within the new borders.
    if len(image.shape) == 3:  # For color images
        bigger = np.zeros((image.shape[0] + 2 * bordure[0], image.shape[1] + 2 * bordure[1], image.shape[2]))
        bigger[bordure[0]:bordure[0] + image.shape[0], bordure[1]:bordure[1] + image.shape[1], :] = image
    elif len(image.shape) == 2:  # For grayscale images
        bigger = np.zeros((image.shape[0] + 2 * bordure[0], image.shape[1] + 2 * bordure[1]))
        bigger[bordure[0]:bordure[0] + image.shape[0], bordure[1]:bordure[1] + image.shape[1]] = image
    
    return bigger
def indices_list(mask, border_size, target_class, place):
    # Retrieve the indices (i, j) of pixels belonging to the specified class.
    if place == 'onData':
        indices = np.argwhere(mask == target_class).astype(int)

        # Filter indices to exclude those within the border area.
        filtered_indices = indices[
            (indices[:, 0] < mask.shape[0] - border_size[0] / 2 - 1) &
            (indices[:, 0] > border_size[0] / 2 + 1) &
            (indices[:, 1] > border_size[1] / 2 + 1) &
            (indices[:, 1] < mask.shape[1] - border_size[1] / 2 - 1)
            ].astype(int)

        # If no indices are found, select indices not belonging to the target class.
        if len(filtered_indices) == 0:
            indices = np.argwhere(mask != target_class).astype(int)
            # Filter these indices to exclude those within the border.
            filtered_indices = indices[
                (indices[:, 0] < mask.shape[0] - border_size[0] / 2 - 1) &
                (indices[:, 0] > border_size[0] / 2 + 1) &
                (indices[:, 1] > border_size[1] / 2 + 1) &
                (indices[:, 1] < mask.shape[1] - border_size[1] / 2 - 1)
                ].astype(int)
    elif place == 'mask':
        # Similar process for selecting indices where the mask is 0 (background or non-target class).
        indices = np.argwhere(mask == 0).astype(int)

        # Filter to exclude border areas.
        filtered_indices = indices[
            (indices[:, 0] < mask.shape[0] - border_size[0] / 2 - 1) &
            (indices[:, 0] > border_size[0] / 2 + 1) &
            (indices[:, 1] > border_size[1] / 2 + 1) &
            (indices[:, 1] < mask.shape[1] - border_size[1] / 2 - 1)
            ].astype(int)

        # If no suitable indices are found, select from non-background.
        if len(filtered_indices) == 0:
            indices = np.argwhere(mask != 0).astype(int)
            # Again, filter these to exclude border areas.
            filtered_indices = indices[
                (indices[:, 0] < mask.shape[0] - border_size[0] / 2 - 1) &
                (indices[:, 0] > border_size[0] / 2 + 1) &
                (indices[:, 1] > border_size[1] / 2 + 1) &
                (indices[:, 1] < mask.shape[1] - border_size[1] / 2 - 1)
                ].astype(int)

    else:
        # For any other specified place, select indices not belonging to the target class.
        indices = np.argwhere(mask != target_class).astype(int)
        # Filter to exclude border areas.
        filtered_indices = indices[
            (indices[:, 0] < mask.shape[0] - border_size[0] / 2 - 1) &
            (indices[:, 0] > border_size[0] / 2 + 1) &
            (indices[:, 1] > border_size[1] / 2 + 1) &
            (indices[:, 1] < mask.shape[1] - border_size[1] / 2 - 1)
            ].astype(int)

    # Randomly select an index from the filtered list and adjust for the border size.
    index = np.random.choice(len(filtered_indices))
    rand_row, rand_col = filtered_indices[index]
    rand_row_plot, rand_col_plot = rand_row.copy(), rand_col.copy()
    rand_row, rand_col = (rand_row - border_size[0] / 2).astype(int), (rand_col - border_size[1] / 2).astype(int)

    return rand_row, rand_col, rand_row_plot, rand_col_plot

def indices_list_no_filter(mask, target_class, place):
    # Retrieve indices (i, j) of pixels belonging to the specified class without filtering for border.
    if place == 'onData' or place == 'mask':
        indices = np.argwhere(mask == target_class).astype(int)

        # If no indices found for the target class, select indices for the opposite class.
        if len(indices) == 0:
            indices = np.argwhere(mask != target_class).astype(int)
    else:
        # For other specified places, select indices not belonging to the target class.
        indices = np.argwhere(mask != target_class).astype(int)

    # Randomly select an index from the list.
    index = np.random.choice(len(indices))
    rand_row, rand_col = indices[index]

    return rand_row, rand_col

def flip(rand_flip, raster_img, mask_filter=None, mask_img=None):
    # Apply flipping transformations based on the random choice.
    if rand_flip == 1:
        # Flip left to right.
        raster_img = np.fliplr(raster_img)
        if mask_filter is not None:
            mask_filter = np.fliplr(mask_filter)
        if mask_img is not None:
            mask_img = np.fliplr(mask_img)
    elif rand_flip == 2:
        # Flip up to down.
        raster_img = np.flipud(raster_img)
        if mask_filter is not None:
            mask_filter = np.flipud(mask_filter)
        if mask_img is not None:
            mask_img = np.flipud(mask_img)
    elif rand_flip == 3:
        # Flip both left to right and up to down.
        raster_img = np.fliplr(np.flipud(raster_img))
        if mask_filter is not None:
            mask_filter = np.fliplr(np.flipud(mask_filter))
        if mask_img is not None:
            mask_img = np.fliplr(np.flipud(mask_img))

    return raster_img, mask_filter, mask_img

def random_crop(matrix, crop_size):
    # Check if the specified crop size is smaller than the matrix dimensions.
    if crop_size[0] > matrix.shape[0] or crop_size[1] > matrix.shape[1]:
        print("Crop size should be smaller than matrix dimensions.")
        y = 0
        x = 0
    else:
        # Randomly select the top-left corner of the crop area.
        y = np.random.randint(0, matrix.shape[0] - crop_size[0])
        x = np.random.randint(0, matrix.shape[1] - crop_size[1])

    return matrix[y:y + crop_size[0], x:x + crop_size[1]]

def incrustation(img_sim, mask_sim, mask_prop, image, value, place, NDVI=False):

    # Select a threshold randomly for deciding on mixing operations and whether to mix the images.
    rand_seuil = np.random.choice([4, 6])/10
    mix = np.random.choice([True, False], p=[0.8, 0.2])
    
    # Store the dimensions of the simulation image for later use.
    rows_img_sim = img_sim.shape[0]
    cols_img_sim = img_sim.shape[1]
    
    # Randomly choose a flip operation to apply to the image.
    rand_flip = np.random.choice([0, 1, 2, 3], 1, p=[0.25, 0.25, 0.25, 0.25])
    raster_img, _, _ = flip(rand_flip, image)
    
    # Post-flip image value check.
    if np.max(raster_img) < 1:
        plt.imshow(raster_img[:, :, :3] / np.max(raster_img[:, :, :3]))
        plt.show()
        print('ALARNE après flip')
    
    # Randomly decide whether to rotate the image.
    rand_angle = np.random.choice([0, 1], p=[0.4, 0.6])
    if rand_angle == 1 and image.shape[0] > 8:
        raster_img = rotation(raster_img)
    
    # Post-rotation image value check.
    if np.max(raster_img) < 1:
        plt.imshow(raster_img[:, :, :3] / np.max(raster_img[:, :, :3]))
        plt.show()
        print('ALARNE après rot')
    
    # NDVI-specific processing, creating a mask based on NDVI values if applicable.
    if NDVI:
        img_NDVI = (raster_img[:, :, 3] - raster_img[:, :, 0]) / (raster_img[:, :, 3] + raster_img[:, :, 0] + 1)
        mask_img = np.where(raster_img[:, :, 0] > 0, 1, 0).astype(float)
        mask_img = np.where(img_NDVI > 0.6, 0, mask_img)
        if np.max(mask_img) < 1:
            print('NDVI', np.unique(mask_img), 'ALARME masque nulle')
    else:
        mask_img = np.where(raster_img[:, :, 0] > 0, 1, 0).astype(float)
        if np.max(mask_img) < 1:
            plt.imshow(mask_img)
            plt.show()
            print('pas NDVI', np.unique(mask_img), 'ALARME masque nulle', rows_img_sim, cols_img_sim, raster_img.shape)
            plt.imshow(image[:, :, :3] / np.max(image[:, :, :3]))
            plt.show()
    
    # Decide on mixing the masks based on the 'mix' flag.
    if mix:
        padded_mask = np.pad(mask_img, ((1, 1), (1, 1)), mode='constant', constant_values=0)
        mask_filter = uniform_filter(padded_mask, size=3)
        mask_filter = np.clip(mask_filter[1:-1, 1:-1], 0, 1) * mask_img
        mask_filter = normalize_array(mask_filter, 0, max(1, np.max(mask_filter)))
    else:
        mask_filter = mask_img
    
    # Ensure mask filter values are valid.
    if np.min(mask_filter) < 0:
        print(np.unique(mask_filter))
        print('ALARME MASK FILTER < 0')
    
    # Prepare for the incrustation process by selecting the appropriate area based on the mask and place.
    expand_mask_filter = np.repeat(mask_filter[:, :, np.newaxis], 4, axis=2)
    if place == 'mask':
        mask_ind = mask_sim
        target = 100
    elif place == 'onData':
        mask_ind = mask_sim
        target = np.argmax(np.bincount(mask_sim.flatten()))
    
    # Select the insertion point within the simulation image without filtering for specific conditions.
    rand_row, rand_col = indices_list_no_filter(mask_ind, target, place)
    rand_row, rand_col = int(rand_row + mask_img.shape[0]/2), int(rand_col + mask_img.shape[1]/2)
    
    # Combine the texture and simulation images based on the mask filter.
    sub_image_texture = expand_mask_filter * raster_img
    img_sim = generate_bordures(img_sim, [int(mask_img.shape[0]), int(mask_img.shape[1])], max_dim=10000)
    sub_image_sim = (1 - expand_mask_filter) * img_sim[rand_row:rand_row + expand_mask_filter.shape[0],
                                                       rand_col:rand_col + expand_mask_filter.shape[1], :]
    
    # Final composition and cleanup of the simulation image and masks.
    sub_image = sub_image_texture + sub_image_sim
    img_sim[rand_row:rand_row + mask_img.shape[0], rand_col:rand_col + mask_img.shape[1], :] = np.where(
        expand_mask_filter > rand_seuil, sub_image,
        img_sim[rand_row:rand_row + mask_img.shape[0], rand_col:rand_col + mask_img.shape[1], :])
    mask_sim = generate_bordures(mask_sim, [int(mask_img.shape[0]), int(mask_img.shape[1])], max_dim=10000)
    mask_prop = generate_bordures(mask_prop, [int(mask_img.shape[0]), int(mask_img.shape[1])], max_dim=10000)
    mask_sim[rand_row:rand_row + mask_img.shape[0], rand_col:rand_col + mask_img.shape[1]] = np.where(
        expand_mask_filter[:, :, 0] > rand_seuil, value,
        mask_sim[rand_row:rand_row + mask_img.shape[0], rand_col:rand_col + mask_img.shape[1]])
    mask_prop[rand_row:rand_row + mask_img.shape[0], rand_col:rand_col + mask_img.shape[1]] = np.where(
        expand_mask_filter[:, :, 0] > rand_seuil, value,
        mask_prop[rand_row:rand_row + mask_img.shape[0], rand_col:rand_col + mask_img.shape[1]])
    
    # Crop the simulation image and masks back to their original dimensions.
    img_sim = img_sim[int(mask_img.shape[0]): int(mask_img.shape[0]) + rows_img_sim, int(mask_img.shape[1]): int(mask_img.shape[1]) + cols_img_sim, :]
    mask_sim = mask_sim[int(mask_img.shape[0]): int(mask_img.shape[0]) + rows_img_sim, int(mask_img.shape[1]): int(mask_img.shape[1]) + cols_img_sim]
    mask_prop = mask_prop[int(mask_img.shape[0]): int(mask_img.shape[0]) + rows_img_sim, int(mask_img.shape[1]): int(mask_img.shape[1]) + cols_img_sim]

    # Final checks and return the modified simulation image and masks.
    if np.min(img_sim) < 0:
        print('ALARME img_sim < 0 pas rapport')
    return img_sim, mask_sim, mask_prop
