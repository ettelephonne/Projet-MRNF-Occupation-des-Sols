import os
import numpy as np
from joblib import Parallel, delayed
from fonctions import *

def master_function(repet_all):
    print('Simulation number', repet_all)
    in_path_textures = r'Path to background textures'
    in_path_shapes = r'Path to samples'
    os.makedirs(r'Path to the simulated images', exist_ok=True)
    os.makedirs(r'Path to the simulated masks', exist_ok=True)
    out_path_img = r'Path to the simulated images'
    out_path_masks = r'Path to the simulated masks'
    classes_textures = ['water', 'big_urban', 'small_urban', 'grass_land', 'grass_intermed', 'sol_nu_roche', 'forests', 'sol_nu_terre']
    classes_formes = ['big_urban', 'clouds', 'forests', 'grass_intermed', 'grass_land', 'routes', 'small_urban', 'sol_nu_roche', 'sol_nu_terre', 'water']
    prob_cible = [5, 0, 5, 10, 10, 10, 40, 30, 15, 3]
    sum_prop_cible = sum(prob_cible)
    prob = [x / sum_prop_cible for x in prob_cible]
    sim_size = [1024, 1024]
    dico_textures = {}
    folders = os.listdir(in_path_textures)
    folders.sort()
    for folder in folders:
        dico_textures[folder] = []
        count = 0
        current_list = os.listdir(os.path.join(in_path_textures, folder))
        if len(current_list) > 200:
            current_list = np.random.choice(current_list, 200, replace=False)
        for file in current_list:
            count += 1
            if count >= 1000:
                break
            raster = gdal.Open(os.path.join(in_path_textures, folder, file))
            texture = np.moveaxis(raster.ReadAsArray(), 0, 2)
            dico_textures[folder].append(texture)
    dico_formes = {}
    folders = os.listdir(in_path_shapes)
    folders.sort()
    for folder in folders:
        dico_formes[folder] = []
        count = 0
        current_list = os.listdir(os.path.join(in_path_shapes, folder, 'images'))
        if len(current_list) > 200:
            current_list = np.random.choice(current_list, 200, replace=False)
        for file in current_list:
            count += 1
            raster = gdal.Open(os.path.join(in_path_shapes, folder, 'images', file))
            texture = np.moveaxis(raster.ReadAsArray(), 0, 2)
            texture = texture.astype(np.uint16)
            dico_formes[folder].append(texture)

    img_sim = np.zeros((sim_size[0], sim_size[1], 4)).astype(np.float32)
    mask_sim = np.zeros((sim_size[0], sim_size[1])).astype(int)
    for i in range(0, sim_size[0], 128):
        for j in range(0, sim_size[1], 128):
            rand_class = np.random.choice(classes_textures, p=prob)
            texture = random_crop(dico_textures[str(rand_class)][np.random.randint(0, len(dico_textures[rand_class]))],
                                  (128, 128))
            img_sim[i:i + 128, j:j + 128, :] = texture
            mask_sim[i:i + 128, j:j + 128] = list(dico_formes.keys()).index(str(rand_class))
            mask_prop = mask_sim.copy()*0 + 100

    print('incrustations random')
    only_rocks = np.random.choice([0, 1], p=[0.4, 0.6])
    if only_rocks == 0:
        prob_cible = [5, 0, 5, 10, 10, 10, 40, 30, 15, 3]
    else:
        prob_cible = [0, 0, 5, 10, 10, 2, 0, 50, 15, 3]
    sum_prop_cible = sum(prob_cible)
    prob = [x / sum_prop_cible for x in prob_cible]
    while np.count_nonzero(mask_prop==100)/mask_prop.size > 0.15:
        rand_class = np.random.choice(classes_formes, p=prob)
        rand_img = np.random.randint(0, len(dico_formes[rand_class]))
        image = dico_formes[rand_class][rand_img]
        if rand_class == 'routes':
            NDVI = True
        else:
            NDVI = False
        img_sim, mask_sim, mask_prop = incrustation(img_sim, mask_sim, mask_prop, image, list(dico_formes.keys()).index(rand_class), place='mask', NDVI=NDVI)
    print('incrustation routes')
    if only_rocks == 0:
        nbr_roads = np.random.randint(2, 30)
    else:
        nbr_roads = np.random.randint(100, 300)
    for _ in range(nbr_roads):
        rand_img = np.random.randint(0, len(dico_formes['routes']))
        image = dico_formes['routes'][rand_img]
        NDVI = True
        img_sim, mask_sim, mask_prop = incrustation(img_sim, mask_sim, mask_prop, image, list(dico_formes.keys()).index('routes'), place='mask', NDVI=NDVI)
    print('incrustation nuages')
    for _ in range(np.random.randint(0, 8)):
        rand_img = np.random.randint(0, len(dico_formes['clouds']))
        image = dico_formes['clouds'][rand_img]
        img_sim, mask_sim, mask_prop = incrustation(img_sim, mask_sim, mask_prop, image, list(dico_formes.keys()).index('clouds'), place='mask')

    driver = gdal.GetDriverByName('GTiff')
    new_img = driver.Create(out_path_img + '/sim_' + str(repet_all) + '.tif', img_sim.shape[1]
                            , img_sim.shape[0], 4, gdal.GDT_UInt16)

    img_sim = img_sim.astype(np.uint16)
    for z in range(4):
        band = new_img.GetRasterBand(z + 1)
        band.WriteArray(img_sim[:, :, z])

    new_mask = driver.Create(out_path_masks + '/sim_' + str(repet_all) + '.tif',
                             mask_sim.shape[1]
                             , mask_sim.shape[0], 1, gdal.GDT_Byte)
    band = new_mask.GetRasterBand(1)
    band.WriteArray(mask_sim)

    for i in range(len(classes_formes)):
        print(classes_formes[i], 100*np.count_nonzero(mask_sim==i) / mask_sim.size)

    new_img = None
    new_mask = None
repet_all = 0

tasks = []
for repet in range(200):
    repet_all += 1
    tasks.append(delayed(master_function)(repet_all))
Parallel(n_jobs=4)(tasks)
