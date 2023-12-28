import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import rotate
from scipy.ndimage import distance_transform_edt
from osgeo import gdal
from scipy.ndimage import gaussian_filter, uniform_filter
from skimage.morphology import erosion, square
import os
from skimage.morphology import skeletonize
from matplotlib.path import Path
from skimage.util import pad

def normalize_array(array, new_min, new_max):
    old_min = np.min(array)
    old_max = np.max(array)
    #print(old_min, old_max)
    array = (array - old_min) / (old_max - old_min)  # Normalise l'array entre 0 et 1
    array = array * (new_max - new_min)  # Étire ou réduit l'array à la nouvelle plage
    array = array + new_min  # Décale l'array à la nouvelle plage
    return array
def rotation(img):
    angle = np.random.randint(0, 360, 1)[0]
    #rand_order = np.random.randint(1, 6)
    img = rotate(img, angle, reshape=False, order=0, mode='constant', cval=0.0)
    return img
def generate_shape(shape):
    rand_shape = np.random.choice(['circle', 'rectangle', 'triangle'], p=[0.1, 0.85, 0.05])
    if rand_shape == 'circle':
        center = (np.random.randint(shape[1] // 4, shape[1]), np.random.randint(shape[0] // 4, shape[0]))
        print('shape', shape)
        radius_x = np.random.randint(shape[1] // 5, shape[1] // 3 + 1)
        radius_y = np.random.randint(shape[0] // 5, shape[0] // 3 + 1)
        y, x = np.ogrid[-center[1]:shape[0] - center[1], -center[0]:shape[1] - center[0]]
        mask = (x * x) / (radius_x * radius_x) + (y * y) / (radius_y * radius_y) <= 1  # Équation de l'ellipse
        img = np.zeros((shape[0], shape[1], 4))
        img[mask] = 1
    if rand_shape == 'rectangle':
        ratio = 0
        while ratio > 10 or ratio < 0.1:  # en ajoutant aussi la condition inverse pour éviter des formes très plates.
            top_left = (np.random.randint(0, shape[1] // 2), np.random.randint(0, shape[0] // 2))
            bottom_right = (np.random.randint(top_left[0] + 1, shape[1]), np.random.randint(top_left[1] + 1, shape[0]))
            height = bottom_right[1] - top_left[1]
            width = bottom_right[0] - top_left[0]
            if width == 0:
                continue
            ratio = height / width
        img = np.zeros((shape[0], shape[1], 4))
        img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = 1
    if rand_shape == 'triangle':
        vertices = np.array([
            [np.random.randint(0, shape[1]), np.random.randint(0, shape[0])],
            [np.random.randint(0, shape[1]), np.random.randint(0, shape[0])],
            [np.random.randint(0, shape[1]), np.random.randint(0, shape[0])]
        ])

        grid = np.mgrid[0:shape[0], 0:shape[1]].reshape(2, -1).T
        mask = Path(vertices).contains_points(grid).reshape(shape)
        img = np.zeros((shape[0], shape[1], 4))
        img[mask] = 1
    #rand_rotation = np.random.choice([0, 1], 1, p=[0.5, 0.5])
    #if rand_rotation == 1:
    #    img = rotation(img)
    return img
def index_from_dist(mask, dist):
    distance_from_road = distance_transform_edt(1 - mask)
    possible_indices = np.where(np.logical_and(distance_from_road >= dist[0], distance_from_road <= dist[1]))
    if possible_indices[0].size == 0:
        print('pas  d indice dispo !!!')
        cx, cy = np.random.randint(0, mask.shape[1]), np.random.randint(0, mask.shape[0])
        return cx, cy
    random_index = np.random.choice(possible_indices[0].shape[0])
    cx, cy = possible_indices[0][random_index], possible_indices[1][random_index]
    return cx, cy
def crop_image_to_data(img, value):
    # Trouver les indices où il y a des données
    #print(np.unique(img), [np.count_nonzero(img==x) for x in np.unique(img)])

    ys, xs = np.where(img[:, :] != value)

    # Trouver les limites de l'image
    xmin, xmax = np.min(xs), np.max(xs)
    ymin, ymax = np.min(ys), np.max(ys)

    # Cropper l'image
    #cropped_img = img[ymin:ymax+1, xmin:xmax+1]

    return [ymin, ymax+1, xmin, xmax+1]
def crop_bordures(image, bordure):
    if len(image.shape) == 3:
        image = image[bordure[0]:image.shape[0]-bordure[0], bordure[1]:image.shape[1]-bordure[1], :]
    else:
        image = image[bordure[0]:image.shape[0] - bordure[0], bordure[1]:image.shape[1] - bordure[1]]
    return image
def distortion(image):
    #rand_rotation = np.random.choice([0, 1], p=[0.5, 0.5])
    #if rand_rotation == 1:
    #    image = rotation(image)
    image = generate_bordures(image, [300, 300])
    rows, cols = image[:, :, 0].shape
    mask = image[:, :, 0].copy() * 0
    mask = np.where(image[:, :, 0] > 0, 1, mask)
    cx, cy = index_from_dist(mask, [5, 100])
    amplitude = np.random.randint(100, 400, 1)
    # Création de matrices de coordonnées pour toute l'image
    i, j = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')

    # Calcul de la distance pour chaque point par rapport au centre
    distances = np.sqrt((i - cy) ** 2 + (j - cx) ** 2)

    # Calcul de l'offset pour chaque point
    rand_factor = np.random.randint(4, 16, 1)[0] / 10
    offsets = (amplitude * np.sin(distances / (rand_factor * cols))).astype(int)

    # Mise à jour des indices en utilisant les offsets
    new_i = np.clip(i + offsets, 0, rows - 1)
    new_j = np.clip(j + offsets, 0, cols - 1)

    img_output = image[new_i, new_j]
    mask_output = np.where(img_output[:, :, 0] > 0, 1, 0).astype(float)
    padded_mask = np.pad(mask_output, ((1, 1), (1, 1)), mode='constant', constant_values=0)
    eroded_padded_mask = erosion(padded_mask, square(3))
    mask_output = eroded_padded_mask[1:-1, 1:-1]
    mask_filter_output = gaussian_filter(mask_output, sigma=1)
    ymin, ymax, xmin, xmax = crop_image_to_data(mask_output, 0)
    img_output = img_output[ymin:ymax, xmin:xmax]
    mask_output = mask_output[ymin:ymax, xmin:xmax]
    mask_filter_output = mask_filter_output[ymin:ymax, xmin:xmax]

    return img_output, mask_output, mask_filter_output
def generate_bordures(image, bordure, max_dim=1024):
    bordure = np.asarray(bordure)
    # Calcul des nouvelles dimensions potentielles avec la bordure
    new_height = image.shape[0] + 2 * bordure[0]
    new_width = image.shape[1] + 2 * bordure[1]
    # Si l'une des nouvelles dimensions dépasse max_dim, ajustez la bordure
    if new_height > max_dim:
        print(bordure[0], max_dim, image.shape[0])
        bordure[0] = (max_dim - image.shape[0]) // 2
    if new_width > max_dim:
        bordure[1] = (max_dim - image.shape[1]) // 2
    # Créez la nouvelle image en fonction des dimensions ajustées
    if len(image.shape) == 3:
        bigger = np.zeros((image.shape[0] + 2 * bordure[0], image.shape[1] + 2 * bordure[1], image.shape[2]))
        bigger[bordure[0]:bordure[0] + image.shape[0], bordure[1]:bordure[1] + image.shape[1], :] = image
    elif len(image.shape) == 2:
        bigger = np.zeros((image.shape[0] + 2 * bordure[0], image.shape[1] + 2 * bordure[1]))
        bigger[bordure[0]:bordure[0] + image.shape[0], bordure[1]:bordure[1] + image.shape[1]] = image
    return bigger
def indices_list(mask, border_size, target_class, place):
    # Récupérer les indices (i, j) des pixels avec la classe spécifique
    if place == 'onData':
        indices = np.argwhere(mask == target_class).astype(int)

        # Filtrer les indices pour exclure ceux dans la bordure
        filtered_indices = indices[
            (indices[:, 0] < mask.shape[0] - border_size[0] / 2 - 1) &
            (indices[:, 0] > border_size[0] / 2 + 1) &
            (indices[:, 1] > border_size[1] / 2 + 1) &
            (indices[:, 1] < mask.shape[1] - border_size[1] / 2 - 1)
            ].astype(int)

        if len(filtered_indices) == 0:
            indices = np.argwhere(mask != target_class).astype(int)
            # Filtrer les indices pour exclure ceux dans la bordure
            filtered_indices = indices[
                (indices[:, 0] < mask.shape[0] - border_size[0] / 2 - 1) &
                (indices[:, 0] > border_size[0] / 2 + 1) &
                (indices[:, 1] > border_size[1] / 2 + 1) &
                (indices[:, 1] < mask.shape[1] - border_size[1] / 2 - 1)
                ].astype(int)
    if place == 'mask':
        indices = np.argwhere(mask == 0).astype(int)

        # Filtrer les indices pour exclure ceux dans la bordure
        filtered_indices = indices[
            (indices[:, 0] < mask.shape[0] - border_size[0] / 2 - 1) &
            (indices[:, 0] > border_size[0] / 2 + 1) &
            (indices[:, 1] > border_size[1] / 2 + 1) &
            (indices[:, 1] < mask.shape[1] - border_size[1] / 2 - 1)
            ].astype(int)

        if len(filtered_indices) == 0:
            indices = np.argwhere(mask != 0).astype(int)
            # Filtrer les indices pour exclure ceux dans la bordure
            filtered_indices = indices[
                (indices[:, 0] < mask.shape[0] - border_size[0] / 2 - 1) &
                (indices[:, 0] > border_size[0] / 2 + 1) &
                (indices[:, 1] > border_size[1] / 2 + 1) &
                (indices[:, 1] < mask.shape[1] - border_size[1] / 2 - 1)
                ].astype(int)

    else:
        indices = np.argwhere(mask != target_class).astype(int)
        # Filtrer les indices pour exclure ceux dans la bordure
        filtered_indices = indices[
            (indices[:, 0] < mask.shape[0] - border_size[0] / 2 - 1) &
            (indices[:, 0] > border_size[0] / 2 + 1) &
            (indices[:, 1] > border_size[1] / 2 + 1) &
            (indices[:, 1] < mask.shape[1] - border_size[1] / 2 - 1)
            ].astype(int)


    index = np.random.choice(len(filtered_indices))
    rand_row, rand_col = filtered_indices[index]
    rand_row_plot, rand_col_plot = rand_row.copy(), rand_col.copy()
    rand_row, rand_col = (rand_row - border_size[0] / 2).astype(int), (rand_col - border_size[1] / 2).astype(int)

    return rand_row, rand_col, rand_row_plot, rand_col_plot
def indices_list_no_filter(mask, target_class, place):
    # Récupérer les indices (i, j) des pixels avec la classe spécifique
    if place == 'onData':
        indices = np.argwhere(mask == target_class).astype(int)

        if len(indices) == 0:
            indices = np.argwhere(mask != target_class).astype(int)
    elif place == 'mask':
        indices = np.argwhere(mask == target_class).astype(int)

        if len(indices) == 0:
            indices = np.argwhere(mask != target_class).astype(int)
    else:
        indices = np.argwhere(mask != target_class).astype(int)

    index = np.random.choice(len(indices))
    rand_row, rand_col = indices[index]
    rand_row, rand_col = (rand_row).astype(int), (rand_col).astype(int)

    return rand_row, rand_col
def flip(rand_flip, raster_img, mask_filter=None, mask_img=None):
    if rand_flip == 1:
        raster_img = np.fliplr(raster_img)
        if mask_filter is not None:
            mask_filter = np.fliplr(mask_filter)
        if mask_img is not None:
            mask_img = np.fliplr(mask_img)
    elif rand_flip == 2:
        raster_img = np.flipud(raster_img)
        if mask_filter is not None:
            mask_filter = np.flipud(mask_filter)
        if mask_img is not None:
            mask_img = np.flipud(mask_img )
    elif rand_flip == 3:
        raster_img = np.fliplr(raster_img)
        if mask_filter is not None:
            mask_filter = np.fliplr(mask_filter)
        if mask_img is not None:
            mask_img = np.fliplr(mask_img)
        raster_img = np.flipud(raster_img)
        if mask_filter is not None:
            mask_filter = np.flipud(mask_filter)
        if mask_img is not None:
            mask_img = np.flipud(mask_img)
    return raster_img, mask_filter, mask_img
def random_crop(matrix, crop_size):
    if crop_size[0] > matrix.shape[0] or crop_size[1] > matrix.shape[1]:
        print("Crop size should be smaller than matrix dimensions.")
        y = 0
        x = 0
    else:
        y = np.random.randint(0, matrix.shape[0] - crop_size[0])
        x = np.random.randint(0, matrix.shape[1] - crop_size[1])
    return matrix[y:y + crop_size[0], x:x + crop_size[1]]
def incrustation(img_sim, mask_sim, mask_prop, image, value, place, NDVI=False):
    if np.max(image) < 1:
        print('ALARNE dès le début')
    rand_seuil = np.random.choice([4, 6])/10
    mix = np.random.choice([True, False], p=[0.8, 0.2])
    rows_img_sim = img_sim.shape[0]
    cols_img_sim = img_sim.shape[1]
    rand_flip = np.random.choice([0, 1, 2, 3], 1, p=[0.25, 0.25, 0.25, 0.25])
    raster_img, _, _ = flip(rand_flip, image)
    if np.max(raster_img) < 1:
        plt.imshow(raster_img[:, :, :3] / np.max(raster_img[:, :, :3]))
        plt.show()
        print('ALARNE après flip')
    rand_angle = np.random.choice([0, 1], p=[0.4, 0.6])
    if rand_angle == 1 and image.shape[0] > 8:
        raster_img = rotation(raster_img)
    if np.max(raster_img) < 1:
        plt.imshow(raster_img[:, :, :3] / np.max(raster_img[:, :, :3]))
        plt.show()
        print('ALARNE après rot')
    if NDVI == True:
        img_NDVI = (raster_img[:, :, 3] - raster_img[:, :, 0]) / (raster_img[:, :, 3] + raster_img[:, :, 0] + 1)
        mask_img = np.where(raster_img[:, :, 0] > 0, 1, 0).astype(float)
        #plt.imshow(mask_img)
        #plt.show()
        mask_img = np.where(img_NDVI > 0.6, 0, mask_img)
        if np.max(mask_img) < 1:
            print('NDVI', np.unique(mask_img), 'ALARME masque nulle')
        #plt.imshow(mask_img)
        #plt.show()
    else:
        mask_img = np.where(raster_img[:, :, 0] > 0, 1, 0).astype(float)
        if np.max(mask_img) < 1:
            plt.imshow(mask_img)
            plt.show()
            print('pas NDVI', np.unique(mask_img), 'ALARME masque nulle', rows_img_sim, cols_img_sim, raster_img.shape)
            plt.imshow(image[:, :, :3] / np.max(image[:, :, :3]))
            plt.show()
    if mix == True:
        padded_mask = np.pad(mask_img, ((1, 1), (1, 1)), mode='constant', constant_values=0)
        #plt.imshow(padded_mask)
        #plt.show()
        mask_filter = uniform_filter(padded_mask, size=3)
        mask_filter = np.clip(mask_filter[1:-1, 1:-1], 0, 1)*mask_img
        mask_filter = normalize_array(mask_filter, 0, max(1, np.max(mask_filter)))
        #plt.imshow(mask_filter)
        #plt.show()
    else:
        mask_filter = mask_img
    if np.min(mask_filter) < 0:
        print(np.unique(mask_filter))
        print('ALARME MASK FILTER < 0')
    expand_mask_filter = np.repeat(mask_filter[:, :, np.newaxis], 4, axis=2)
    if place == 'mask':
        mask_ind = mask_sim
        target = 100
    if place == 'onData':
        mask_ind = mask_sim
        target = np.argmax(np.bincount(mask_sim.flatten()))
    rand_row, rand_col = indices_list_no_filter(mask_ind, target, place)
    rand_row, rand_col = int(rand_row+mask_img.shape[0]/2), int(rand_col+mask_img.shape[1]/2)
    #raster_img = (np.random.randint(9, 12) / 10) * raster_img
    #if np.min(raster_img) < 0:
    #    print('ALARME raster_img < 0')
    #for band in range(raster_img.shape[2]):
    #    non_nul = raster_img[:, :, band] > 0
    #    raster_mean = np.mean(raster_img[:, :, band][non_nul])
    #    min_val = np.min(raster_img[:, :, band][non_nul])
    #    low_bound = max(-0.05 * raster_mean, -min_val)
    #    high_bound = 0.05 * raster_mean
    #    lum = np.random.randint(low_bound, high_bound)
        #print(raster_mean, lum, np.min(raster_img[:, :, band][non_nul]))
    #    raster_img[:, :, band][non_nul] = raster_img[:, :, band][non_nul] + lum
    #    if np.min(raster_img) < 0:
    #        print('ALARME raster_img < 0 bande numero', band)
    #        print(raster_mean, lum, np.min(raster_img[:, :, band][non_nul]))
    sub_image_texture = expand_mask_filter * raster_img
    img_sim = generate_bordures(img_sim, [int(mask_img.shape[0]), int(mask_img.shape[1])], max_dim=10000)
    sub_image_sim = (1 - expand_mask_filter) * img_sim[rand_row:rand_row + expand_mask_filter.shape[0],
                                        rand_col:rand_col + expand_mask_filter.shape[1], :]
    #plt.imshow(np.clip(sub_image_sim[:, :, :3], 0, 2000)/2500)
    #plt.show()
    #plt.imshow(np.clip(sub_image_texture[:, :, :3], 0, 2000)/2500)
    #plt.show()
    if np.min(sub_image_sim) < 0:
        print('ALARME sub_image_sim < 0')
    sub_image = sub_image_texture + sub_image_sim
    if np.min(sub_image_sim) < 0:
        print('ALARME sub_image_sim < 0 dans l addition')
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
    #plt.imshow(mask_img)
    #plt.show()
    #plt.imshow(np.clip(expand_mask_filter[:, :, 0], 0, 1))
    #plt.show()
    #img_plot = np.clip(img_sim[rand_row:rand_row + mask_img.shape[0], rand_col:rand_col + mask_img.shape[1], :3], 0, 2000) / 2500
    #print(np.min(img_plot), np.max(img_plot))
    #plt.imshow(img_plot)
    #plt.show()
    img_sim = img_sim[int(mask_img.shape[0]): int(mask_img.shape[0])+rows_img_sim, int(mask_img.shape[1]): int(mask_img.shape[1])+cols_img_sim, :]
    mask_sim = mask_sim[int(mask_img.shape[0]): int(mask_img.shape[0]) + rows_img_sim,
              int(mask_img.shape[1]): int(mask_img.shape[1]) + cols_img_sim]
    mask_prop = mask_prop[int(mask_img.shape[0]): int(mask_img.shape[0]) + rows_img_sim,
              int(mask_img.shape[1]): int(mask_img.shape[1]) + cols_img_sim]

    if np.min(img_sim) < 0:
        print('ALARME img_sim < 0 pas rapport')
    return img_sim, mask_sim, mask_prop

def incrustation_complex(img_sim, mask_sim, mask_prop, image, mask_img, place='mask'):
    if np.max(mask_img) > 200:
        mask_img = mask_img / 255
    '''rand_angle = np.random.choice([0, 1], p=[0.6, 0.4])
    if rand_angle == 1:
        angle = np.random.randint(0, 360, 1)[0]
        rand_order = np.random.randint(1, 5, 1)[0]
        mask_img = rotate(mask_img, angle, reshape=True, order=rand_order, mode='mirror')
        #plt.imshow(mask_img)
        #plt.show()
        #plt.imshow(image[:, :, :3] / np.max(image[:, :, :3]))
        #plt.show()
        image = rotate(image, angle, reshape=True, order=rand_order, mode='mirror')
        #plt.imshow(image[:, :, :3] / np.max(image[:, :, :3]))
        #plt.show()'''
    if place == 'mask':
        mask_ind = mask_prop
        target = 0
    if place == 'onData':
        mask_ind = mask_sim
        target = np.argmax(np.bincount(mask_sim.flatten()))
    # rand_row, rand_col, rand_row_plot, rand_col_plot = indices_list(mask_ind, [mask_img.shape[0],
    #                                                                                mask_img.shape[1]], target, place)
    rand_row, rand_col = indices_list_no_filter(mask_ind, target, place)
    rand_row, rand_col = int(rand_row + mask_img.shape[0] / 2), int(rand_col + mask_img.shape[1] / 2)
    image = (np.random.randint(9, 12) / 10) * image
    for band in range(image.shape[2]):
        non_nul = image[:, :, band] != 0
        raster_mean = np.mean(image[:, :, band][non_nul])
        image[:, :, band] = image[:, :, band] + np.random.randint(-0.1 * raster_mean - 1, 0.1 * raster_mean + 1, 1)
    img_sim = generate_bordures(img_sim, [int(mask_img.shape[0]), int(mask_img.shape[1])], max_dim=10000)
    #plt.imshow(image[:, :, :3]/np.max(image[:, :, :3]))
    #plt.show()
    #print('unique', np.unique(image[:10, :10, :3]))
    img_sim[rand_row:rand_row + mask_img.shape[0], rand_col:rand_col + mask_img.shape[1], :] = np.where(image>0, image, img_sim[rand_row:rand_row + mask_img.shape[0], rand_col:rand_col + mask_img.shape[1], :])
    mask_sim = generate_bordures(mask_sim, [int(mask_img.shape[0]), int(mask_img.shape[1])], max_dim=10000)
    mask_sim[rand_row:rand_row + mask_img.shape[0], rand_col:rand_col + mask_img.shape[1]] = np.where(image[:, :, 0]>0, mask_img, mask_sim[rand_row:rand_row + mask_img.shape[0], rand_col:rand_col + mask_img.shape[1]])
    img_sim = crop_bordures(img_sim, [int(mask_img.shape[0]), int(mask_img.shape[1])])
    mask_sim = crop_bordures(mask_sim, [int(mask_img.shape[0]), int(mask_img.shape[1])])
    return img_sim, mask_sim
def add_object(img, mask_img, thing, thing_mask_filter, coords, value):
    bord = False
    size_thing = thing_mask_filter.shape
    diff = np.asarray(coords) - np.asarray(size_thing)
    sum_shapes = img[:, :, 0].shape - (np.asarray(coords) + np.asarray(size_thing))
    print('diff', diff, coords, sum_shapes)
    print(img.shape, coords, thing.shape)
    if np.min(diff) < 0 or np.min(sum_shapes) < 0:
        bord = True
        neg_diff = diff[diff < 0]
        neg_sum_shapes = sum_shapes[sum_shapes < 0]
        min_neg_diff = abs(np.min(neg_diff)) if neg_diff.size > 0 else 0
        min_neg_sum_shapes = abs(np.min(neg_sum_shapes)) if neg_sum_shapes.size > 0 else 0
        bord_size = np.max([min_neg_diff, min_neg_sum_shapes])
        img = generate_bordures(img, (bord_size, bord_size), max_dim=5000)
        #print(img.shape)
        mask_img = generate_bordures(mask_img, (bord_size, bord_size), max_dim=5000)
        coords = coords + bord_size
    depart_rows = coords[0]-int(size_thing[0]/2)
    stop_rows = depart_rows + size_thing[0]
    depart_cols = coords[1]-int(size_thing[1]/2)
    stop_cols = depart_cols + size_thing[1]
    #print('debur fin', stop_rows-depart_rows, stop_cols-depart_cols, stop_rows, stop_cols)
    #print('func', img.shape, thing.shape, thing_mask_filter.shape, img[depart_rows:stop_rows, depart_cols:stop_cols, :].shape)
    print(img.shape, depart_rows, stop_rows, depart_cols, stop_cols)
    img[depart_rows:stop_rows, depart_cols:stop_cols, :] = \
        thing * np.repeat(thing_mask_filter[:, :, np.newaxis], 4, axis=2) + (1-np.repeat(thing_mask_filter[:, :, np.newaxis], 4, axis=2)) * img[depart_rows:stop_rows, depart_cols:stop_cols, :]
    mask_img[depart_rows:stop_rows, depart_cols:stop_cols] = np.where(thing_mask_filter > 0.4, value, mask_img[depart_rows:stop_rows, depart_cols:stop_cols])
    if bord == True:
        img = crop_bordures(img, (bord_size, bord_size))
        mask_img = crop_bordures(mask_img, (bord_size, bord_size))
    #print('fin', img.shape, thing.shape)
    return img, mask_img
def rand_rotate_bands(img, img_mask_filter):
    #for band in range(4):
    #    img[:, :, band] = img[:, :, band] * np.random.randint(96, 105)/100
    #    mean_band = np.nanmean(img[:, :, band][img[:, :, band]!=0])
    #    print(mean_band)
    #    img[:, :, band] = img[:, :, band] + np.random.randint(int(-0.04*mean_band), int(0.04*mean_band))
    rand_shuffle = np.random.choice([0, 1], p=[0.98, 0.02])
    if rand_shuffle == 1:
        band_indices = list(range(img.shape[2]))
        band_indices.remove(3)
        np.random.shuffle(band_indices)
        band_indices.insert(3, 3)
        shuffled_img = img[:, :, band_indices]
    else:
        shuffled_img = img
    rand_flip = np.random.choice([0, 1, 2, 3], p=[0.4, 0.2, 0.2, 0.2])
    shuffled_img, shuffled_mask_filter_img, _ = flip(rand_flip, shuffled_img, img_mask_filter, mask_img=None)
    return shuffled_img, shuffled_mask_filter_img
def add_thing_function(img, mask_img, class_thing, dico_formes, dico_formes_masks_filter, value):
    # le choix de la distance doit être cohérent avec la taille de l'objet qu'on incruste
    rand_choice = np.random.randint(0, len(dico_formes[class_thing]))
    thing = dico_formes[class_thing][rand_choice]
    thing_mask_filter = dico_formes_masks_filter[class_thing][rand_choice]
    dist_min_max = [thing.shape[0]/2, 1.2 * np.max(thing.shape)]
    mask_dist = np.where(mask_img == value, 1, 0)
    cx, cy = index_from_dist(mask_dist, dist_min_max)
    thing, thing_mask_filter = rand_rotate_bands(thing, thing_mask_filter)
    img, mask_img = add_object(img, mask_img, thing, thing_mask_filter, [cx, cy],
                               list(dico_formes.keys()).index(class_thing))
    return img, mask_img
def add_geometric_thing_function(img, mask_img, dico_textures, class_thing, value, objets_size = None):
    # le choix de la distance doit être cohérent avec la taille de l'objet qu'on incruste
    #print(class_thing)
    rand_choice = np.random.randint(0, len(dico_textures[class_thing]))
    thing = dico_textures[class_thing][rand_choice]
    print('min max texture', np.min(thing), np.max(thing))
    if objets_size is None:
        objets_size = np.random.randint(10, 500)
    forme = np.clip(generate_shape((objets_size, objets_size)), 0, 1)
    print('min max forme shape', np.min(forme), np.max(forme))
    forme_mask = erosion(forme[:, :, 0].copy(), square(3))
    forme_mask_filter = gaussian_filter(forme_mask, sigma=1)
    forme = np.where(forme > 0, 1, forme)
    print('min max forme where', np.min(forme), np.max(forme))
    forme = forme * thing[0:forme.shape[0], 0:forme.shape[1]]
    print('min max forme *', np.min(forme), np.max(forme))
    dist_min_max = [forme.shape[0], 1.2 * np.max(forme.shape)]
    mask_dist = np.where(mask_img == value, 1, 0)
    #plt.imshow(mask_dist)
    #plt.show()
    cx, cy = index_from_dist(mask_dist, dist_min_max)
    print('min max forme', np.min(forme), np.max(forme))
    forme, forme_mask_filter = rand_rotate_bands(forme, forme_mask_filter)
    img, mask_img = add_object(img, mask_img, forme, forme_mask_filter, [cx, cy],
                               list(dico_textures.keys()).index(class_thing))
    #plt.imshow(2*img[:, :, :3]/np.max(img[:, :, :3]))
    #plt.show()
    return img, mask_img

def urban(dico_textures, dico_formes, dico_formes_masks_filter):
    # il faut une texture de fond
    rand_fond = np.random.choice(['herbes_seg', 'sol_nu_seg'], 1, p=[0.8, 0.2])[0]
    fond = dico_textures[rand_fond][np.random.randint(0, len(dico_textures[rand_fond]))]
    # on veut de base une route
    road = dico_formes['routes_seg'][np.random.randint(0, len(dico_formes['routes_seg']))]
    # on la déforme de temps à autre
    rand_deform = np.random.choice([0, 1], 1, p=[0, 1])
    if rand_deform == 1:
        road, road_mask, road_mask_filter = distortion(road)
    road = generate_bordures(road, [300, 300], 1024)
    road_mask = generate_bordures(road_mask, [300, 300], 1024)
    road_mask_filter = generate_bordures(road_mask_filter, [300, 300], 1024)
    #plt.imshow(road_mask)
    #plt.show()
    #plt.imshow(road_mask_filter)
    #plt.show()
    # on a la taille de l'image,on peut appliquer le fond
    img = road * np.repeat(road_mask_filter[:, :, np.newaxis], 4, axis=2) + fond[0:road_mask.shape[0], 0:road_mask.shape[1]] * (1-np.repeat(road_mask_filter[:, :, np.newaxis], 4, axis=2))
    mask_img = (img[:, :, 0].copy()*0 + 1)*list(dico_textures.keys()).index(str(rand_fond))
    mask_img = np.where(road_mask_filter > 0.4, list(dico_textures.keys()).index('routes_seg'), mask_img)
    #plt.imshow(img[:, :, :3]/np.max(img[:, :, :3]))
    #print(np.unique(mask_img))
    #plt.show()
    #plt.imshow(mask_img)
    #plt.show()
    # pour connaître le nombre de maisons à incruster il faut connaître la route
    skeleton = skeletonize(road_mask)
    route_length_estimate = np.sum(skeleton)
    # on a le masque de la route, on va choisir des indices le long
    repere = ['routes_seg', 'maisons_seg']
    class_thing = ['maisons_seg', 'piscines_seg']
    class_thing_textures = ['maisons_seg']

    for c in range(len(class_thing)):
        if class_thing[c] == 'maisons_seg' or class_thing[c] == 'piscines_seg':
            nbr_things = np.random.randint(int(route_length_estimate / 50), int(route_length_estimate / 20))
        for _ in range(nbr_things):
            print('vrai objet', class_thing[c])
            img, mask_img = add_thing_function(img, mask_img, class_thing[c], dico_formes, dico_formes_masks_filter, list(dico_textures.keys()).index(repere[c]))
    for c in range(len(class_thing_textures)):
        if class_thing[c] == 'maisons_seg':
            nbr_things = np.random.randint(int(route_length_estimate / 50), int(route_length_estimate / 20))
        for _ in range(nbr_things):
            print('fausse forme', class_thing_textures[c])
            img, mask_img = add_geometric_thing_function(img, mask_img, dico_textures, class_thing_textures[c], list(dico_textures.keys()).index(repere[c]))
    #while np.count_nonzero(mask_img==list(dico_textures.keys()).index('forests_seg')) / mask_img.size < 0.3:
    #    print('fausse forme', 'forests_seg')
    #    img, mask_img = add_geometric_thing_function(img, mask_img, dico_textures, 'forests_seg',
    #                                                 list(dico_textures.keys()).index('maisons_seg'))
    ymin, ymax, xmin, xmax = crop_image_to_data(mask_img, list(dico_textures.keys()).index(str(rand_fond)))
    mask_img = mask_img[ymin:ymax, xmin:xmax]
    img = img[ymin:ymax, xmin:xmax]
    return img, mask_img
    #plt.imshow(2*img[:, :, :3]/np.max(img[:, :, :3]))
    #plt.show()
    #plt.imshow(mask_img)
    #plt.show()

'''in_path_textures = r'F:\projet_MERN\data\pleiades\big_textures'
dico_textures = {}
for folder in os.listdir(in_path_textures):
    dico_textures[folder] = []
    count = 0
    for file in os.listdir(os.path.join(in_path_textures, folder)):
        count += 1
        if count >= 5:
            break
        raster = gdal.Open(os.path.join(in_path_textures, folder, file))
        texture = np.moveaxis(raster.ReadAsArray(), 0, 2)
        dico_textures[folder].append(texture)
in_path_shapes = r'F:\projet_MERN\data\pleiades\samples_4_sim_V2'
dico_formes = {}
dico_masks = {}
dico_masks_filter = {}
for folder in os.listdir(in_path_shapes):
    dico_formes[folder] = []
    dico_masks[folder] = []
    dico_masks_filter[folder] = []
    count = 0
    for file in os.listdir(os.path.join(in_path_shapes, folder, 'images')):
        count += 1
        if count >= 50:
            break
        raster = gdal.Open(os.path.join(in_path_shapes, folder, 'images', file))
        texture = np.moveaxis(raster.ReadAsArray(), 0, 2)
        raster_mask = gdal.Open(os.path.join(in_path_shapes, folder, 'masks', file))
        mask = raster_mask.ReadAsArray()
        raster_mask_filter = gdal.Open(os.path.join(in_path_shapes, folder, 'masks_filter', file))
        mask_filter = raster_mask_filter.ReadAsArray()
        dico_formes[folder].append(texture)
        dico_masks[folder].append(mask)
        dico_masks_filter[folder].append(mask_filter)'''
#urban(dico_textures, dico_formes, dico_masks_filter)
