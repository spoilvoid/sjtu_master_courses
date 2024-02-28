import os
import os.path as osp
import time
from tqdm import tqdm
import h5py
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing
from functools import partial


def extract_patches(image, patch_size, step):
    """
    Extracts patches from a single-channel image.

    Parameters:
    image (numpy.ndarray): The input image (2D array).
    patch_size (int): The size of each square patch.
    step (int): The step size for moving the window across the image.

    Returns:
    numpy.ndarray: An array of extracted patches. [num_atoms, patch_size, patch_size]
    """
    # Validate inputs
    if not (0 < step <= patch_size):
        raise ValueError("Step size must be positive and less than or equal to the patch size.")

    # Range used below

    # range_0 = chain(range(0, image.shape[0] - patch_size, step),
    # range(image.shape[0] - patch_size, image.shape[0] - patch_size + 1))
    # range_1 = chain(range(0, image.shape[1] - patch_size, step),
    # range(image.shape[1] - patch_size, image.shape[1] - patch_size + 1))

    range_0 = list(range(0, image.shape[0] - patch_size, step))
    range_0.append(image.shape[0] - patch_size)
    range_1 = list(range(0, image.shape[1] - patch_size, step))
    range_1.append(image.shape[1] - patch_size)
    # Calculate the number of patches to extract
    num_patches_0 = len(range_0)
    num_patches_1 = len(range_1)

    # Initialize an array to hold the patches
    patches = np.zeros((num_patches_0 * num_patches_1, patch_size, patch_size))

    # Extract patches
    patch_index = 0
    for i in range_0:
        # print(i)
        for j in range_1:
            # print(i, j)
            patches[patch_index] = image[i:i + patch_size, j:j + patch_size]
            patch_index += 1
    return patches


def reconstruct_image(dictionary, sparse_codes, patch_size, step, image_shape):
    """
    Reconstruct the single-channel image from the sparse coding of dictionary of patches
    i.e. DA

    Parameters:
    dictionary (numpy.ndarray): The learned dictionary (2D array)
    sparse_codes (numpy.ndarray): The sparse codes of the objective image under the dictionary (2D array)
    patch_size (int): The size of each square patch.
    step (int): The step size for moving the window across the image.
    image_shape(tuple): shape of image [0]--height [1]-- width

    Returns:
    image: numpy.ndarray: the objective single-channel image, image_shape
    """
    # Validate inputs
    if not (0 < step <= patch_size):
        raise ValueError("Step size must be positive and less than or equal to the patch size.")

    denoised_image = np.zeros(image_shape)
    # count how many overlapped pixels on each position
    pixel_counts = np.zeros(image_shape)

    # Range used below
    range_0 = list(range(0, image_shape[0] - patch_size, step))
    range_0.append(image_shape[0] - patch_size)
    range_1 = list(range(0, image_shape[1] - patch_size, step))
    range_1.append(image_shape[1] - patch_size)

    patch_index = 0
    for i in range_0:
        for j in range_1:
            patch = np.matmul(dictionary, sparse_codes[:, patch_index]).reshape((patch_size, patch_size))
            denoised_image[i:i+patch_size, j:j+patch_size] += patch
            pixel_counts[i:i+patch_size, j:j+patch_size] += 1
            patch_index += 1
    denoised_image /= pixel_counts

    return denoised_image


def sparse_encode(Y, D, S):
    M, N=D.shape[0],D.shape[1]
    K = Y.shape[1]
    result = np.zeros((N, K))
    for j in range(K):
        indices = []
        y = Y[:, j]
        R = y
        rr = 0
        for i in range(S):
            projection = np.matmul(D.T, R)
            pos = np.argmax(np.abs(projection))
            indices.append(pos)
            At = D[:,indices[:]]
            Ls = np.matmul(np.linalg.pinv(At), y)
            R = y - np.matmul(At, Ls)
            rr=R
            if (R**2).sum()<1e-6:
                break
        for t, s in zip(indices, Ls):
            result[t,j] += s
    return result


def psnr(A, B):
    if (A==B).all():
        return 0
    # mse=((A.astype(np.float)-B.astype(np.float))**2).mean()
    mse1 = ((255.0*A.astype(np.float64) - 255.0*B.astype(np.float64)) ** 2).mean()
    # mse = ((A.astype(np.float) - B.astype(np.float)) ** 2).mean()
    # print('mse:%.2f mse1: %.2f' %mse %mse1)
    return 10*np.log10((255.0**2)/mse1)


def image_denoise(dictionary, noisy_image, clean_image, patch_size=8, step=4, sparsity_level=5):
    """
    Denoise a single-channel image "noisy_image" with OMP

    Parameters:
    dictionary (numpy.ndarray): The learned dictionary (2D array)
    noisy_image (numpy.ndarray): The noisy image
    patch_size (int): The size of each square patch.
    step (int): The step size for moving the window across the image.
    sparsity_level (int): The number of nonzero element of the code for each of the patches.

    Returns:
    numpy.ndarray: the single-channel denoised image in the shape of noisy_image
    """
    best_psnr = 0
    best_image = None
    noisy_patches = extract_patches(noisy_image, patch_size, step).reshape(-1, patch_size**2)
    for test_level in range(1, sparsity_level+1):
        if test_level % 5 == 0:
            print(f'sparsity_level {test_level}')
        denoised_code = sparse_encode(noisy_patches.transpose(1, 0), dictionary, test_level)
        denoised_image = reconstruct_image(dictionary, denoised_code, patch_size, step, noisy_image.shape)
        current_psnr = psnr(denoised_image, clean_image)
        if best_psnr < current_psnr:
            best_psnr = current_psnr
            best_image = denoised_image
    return best_image


def image_denoise_rgb(dictionary, c_image, n_image, number, patch_size=8, step=4, sparsity_level=5):
    if not osp.exists('Task3'):
        os.makedirs('Task3')
    channels = ['R', 'G', 'B']
    dictionary_per_channel = dictionary.transpose(2, 0, 1)
    c_image_per_channel = c_image.transpose(2, 0, 1)
    n_image_per_channel = n_image.transpose(2, 0, 1)
    result_per_channel = []
    psnr_per_channel = []
    start_time = time.time()
    for color, dict_channel, c_channel, n_channel in zip(channels, dictionary_per_channel, c_image_per_channel,
                                                         n_image_per_channel):
        '''dict_channel shape:(patch_size**2, num_atoms)'''
        print(f"-----start No.{number} image OMP denoising in {color} channel-----")
        denoised_img = image_denoise(dict_channel, n_channel, c_channel, patch_size, step, sparsity_level)
        denoised_img_psnr = psnr(denoised_img, c_channel)
        print(f"No.{number} image psnr in {color} channel:{denoised_img_psnr}")
        psnr_per_channel.append(denoised_img_psnr)
        result_per_channel.append(denoised_img)

        print(f"-----finish No.{number} image OMP denoising update in {color} channel-----")

    '''dictionary shape:(patch_size**2, num_atoms, 3)'''
    denoised_img_rgb = np.clip(np.array(result_per_channel).transpose(1, 2, 0),0, 1)
    print(f"No.{number} image psnr:", psnr_per_channel)
    plt.imsave(f"Task3/McM{number}_denoise.png", denoised_img_rgb)

    end_time = time.time()
    process_time = end_time - start_time
    print(f"-----No.{number} image OMP denoising takes {process_time} s-----")
    return psnr_per_channel


if __name__ == "__main__":
    patch_size = 8
    step = 4
    sparsity_level = 20

    image_dir = "McM images"
    image_name = "McM"
    origin_suffix = ".tif"
    dict_dir = "Task2"
    dict_name = "McM_dict_"
    dict_suffix = ".npy"
    noise_dir = "McM images"
    noise_name = "McM"
    noise_suffix = "_noise.mat"
    numbers = [f"{i:02d}" for i in range(1, 19)]
    thread = 9

    c_image_list = [] #clean image list
    dict_list = []
    noi_image_list = []
    for number in numbers:
        print(f"-----loading original No.{number} image-----")
        origin_path = osp.join(image_dir, image_name + number + origin_suffix)
        c_image_list.append(plt.imread(origin_path).astype(np.float32)/255.0)

        print(f"-----loading learned dictionary No.{number} image-----")
        dict_path = osp.join(dict_dir, dict_name + number + dict_suffix)
        dict_clean = np.load(dict_path)
        dict_list.append(dict_clean.astype(np.float32))

        print(f"-----loading noisy No.{number} image-----")
        noise_path = osp.join(noise_dir, noise_name + number + noise_suffix)
        noi_image_list.append(h5py.File(noise_path)['u_n'][:].transpose(2, 1, 0).astype(np.float32)/255.0)
    c_image_list = np.array(c_image_list)
    dict_list = np.array(dict_list)
    noi_image_list = np.array(noi_image_list)

    process_task = partial(image_denoise_rgb, patch_size=patch_size, step=step, sparsity_level=sparsity_level)
    multiprocessing.set_start_method('spawn')
    with multiprocessing.Pool(thread) as pool:
        results = pool.starmap(process_task, zip(dict_list, c_image_list, noi_image_list, numbers))

    np.save('Task3/Task3_psnr.npy', results)
