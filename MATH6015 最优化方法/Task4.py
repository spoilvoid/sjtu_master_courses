import os
import os.path as osp
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing
from functools import partial


def kmeans(data, k=16, max_iters=100):
    # 随机初始化质心
    centroids = data[np.random.choice(range(len(data)), k, replace=False)]

    for _ in tqdm(range(max_iters)):
        # 将样本分配到最近的质心
        clusters = [[] for _ in range(k)]
        for sample in data:
            distances = [np.linalg.norm(sample - centroid) for centroid in centroids]
            closest = np.argmin(distances)
            clusters[closest].append(sample)
        # 更新质心
        new_centroids = np.array([np.mean(cluster, axis=0) if len(cluster)>0 else data[np.random.choice(range(len(data)), 1, replace=False)][0] for cluster in clusters])
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids

    return centroids, clusters


def kmeans_initialization(patches, num_samples, k=16, iters=100):
    num_patches = patches.shape[0]
    _, clusters = kmeans(patches, k, iters)
    selected_samples = []
    num_atoms = 0
    cluster_idx = 0
    nonempty_idxes = []
    nonempty_clusters = []
    num_nonempty_clusters = 0
    for cluster in clusters:
        cluster = np.array(cluster)
        if len(cluster) > 0:
            nonempty_idxes.append(cluster_idx)
            num_nonempty_clusters += 1
        cluster_idx += 1

    cluster_idx = 0
    nonempty_cluster_idx = 0
    for cluster in clusters:
        cluster = np.array(cluster)
        if (cluster_idx in nonempty_idxes) and nonempty_cluster_idx < num_nonempty_clusters-1:
            len_of_this_cluster = int(len(cluster)/num_patches*num_samples)
            num_atoms += len_of_this_cluster
            selected_samples.extend(cluster[np.random.choice(len(cluster), len_of_this_cluster, replace=True)])
            nonempty_cluster_idx += 1
        elif (cluster_idx in nonempty_idxes) and nonempty_cluster_idx >= num_nonempty_clusters-1:
            len_of_this_cluster = num_samples-num_atoms
            selected_samples.extend(cluster[np.random.choice(len(cluster), len_of_this_cluster, replace=True)])
            nonempty_cluster_idx += 1
        cluster_idx += 1
    return np.array(selected_samples).transpose(1,0)


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


def ksvd(dictionary, X, sparsity_level, max_iterations):
    """
    Dictionary learning with KSVD.

    Parameters:
    dictionary (numpy.ndarray): The initial dictionary (2D array).
    X (numpy.ndarray): The input image reshaped(patch_size*patch_size,-1) (2D array).
    sparsity_level (int): The number of nonzero element of the code for each of the patches.
    max_iterations (int): The times of dictionary updated.

    Returns:
    dictionary: numpy.ndarray: The learned dictionary of atoms. [patch_size*patch_size, num_atoms]
    sparse_codes: numpy.ndarray: The coefficients of the image. [num_atoms, X.shape[1]]
    """
    for iteration in range(max_iterations):
        print("Iteration:", iteration+1)
        # Sparse coding
        sparse_codes = sparse_encode(X, dictionary, sparsity_level)

        # Dictionary update
        for i in range(dictionary.shape[1]):
            # Find the patches that use the current atom
            indices = np.nonzero(sparse_codes[i])[0]
            if len(indices) == 0:
                continue
            # Update the current atom
            dictionary[:, i] = 0
            E = X[:, indices] - np.matmul(dictionary, sparse_codes[:, indices])
            U, S, Vt = np.linalg.svd(E, full_matrices=False)
            dictionary[:, i] = U[:, 0]
            sparse_codes[i, indices] = S[0] * Vt[0, :]

            # Normalize the updated atom
            dictionary[:, i] /= np.linalg.norm(dictionary[:, i])

    return dictionary, sparse_codes


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


def dictionary_learning(image, patch_size=8, step=4, num_atoms=256, dict_update_iters=2, sparsity_level=5, k=16, kmeans_iters=100):
    patches = extract_patches(image, patch_size, step).reshape(-1, patch_size**2)
    ini_dictionary = kmeans_initialization(patches, num_atoms, k, kmeans_iters)

    dictionary, sparse_codes = ksvd(ini_dictionary, patches.transpose(1, 0), sparsity_level, dict_update_iters)

    return dictionary, sparse_codes


def dictionary_learning_rgb(image, number, patch_size=8, step=4, num_atoms=256, dict_update_iters=2, sparsity_level=5, k=16, kmeans_iters=100):
    if not osp.exists('Task4'):
        os.makedirs('Task4')
    channels = ['R', 'G', 'B']
    image_per_channel = image.transpose(2, 0, 1)
    result_per_channel = []
    start_time = time.time()
    for color, channel in zip(channels, image_per_channel):
        '''dictionary shape:(patch_size**2, num_atoms)'''
        print(f"-----start No.{number} K-SVD dictionary update in {color} channel-----")
        dictionary, _ = dictionary_learning(channel, patch_size, step, num_atoms, dict_update_iters, sparsity_level,
                                         k, kmeans_iters)
        result_per_channel.append(dictionary)
        print(f"-----finish No.{number} K-SVD dictionary update in {color} channel-----")

    '''dictionary shape:(patch_size**2, num_atoms, 3)'''
    dictionary_rgb = np.array(result_per_channel).transpose(1, 2, 0)
    np.save(f"Task4/McM_dict_{number}.npy", dictionary_rgb)
    end_time = time.time()
    process_time = end_time - start_time
    print(f"-----K-SVD No.{number} dictionary update takes {process_time} s-----")


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


def psnr(A, B):
    if (A==B).all():
        return 0
    # mse=((A.astype(np.float)-B.astype(np.float))**2).mean()
    mse1 = ((255.0*A.astype(np.float64) - 255.0*B.astype(np.float64)) ** 2).mean()
    # mse = ((A.astype(np.float) - B.astype(np.float)) ** 2).mean()
    # print('mse:%.2f mse1: %.2f' %mse %mse1)
    # print('mse1: %.2f' % mse1)
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
    if not osp.exists('Task4'):
        os.makedirs('Task4')
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
    plt.imsave(f"Task4/McM{number}_denoise.png", denoised_img_rgb)

    end_time = time.time()
    process_time = end_time - start_time
    print(f"-----No.{number} image OMP denoising takes {process_time} s-----")
    return psnr_per_channel


if __name__ == "__main__":
    patch_size = 8
    step = 4
    num_atoms = 256
    k = 4
    kmeans_iters = 10
    dict_update_iters = 500
    sparsity_level_learn = 5
    sparsity_level_denoise = 20
    thread = 18

    denoised_image_dir = "BM3D images"
    denoised_image_name = "McM"
    denoised_suffix = "_denoise.png"
    numbers = [f"{i:02d}" for i in range(1, 19)]

    denoised_image_list = []
    for number in numbers:
        print(f"-----loading denoised No.{number} image-----")
        denoised_path = osp.join(denoised_image_dir, denoised_image_name + number + denoised_suffix)
        denoised_img = plt.imread(denoised_path)
        denoised_image_list.append(denoised_img.astype(np.float32))
    denoised_image_list = np.array(denoised_image_list)

    start_time = time.time()
    process_task = partial(dictionary_learning_rgb, patch_size=patch_size, step=step, num_atoms=num_atoms, dict_update_iters=dict_update_iters, sparsity_level=sparsity_level_learn, k=k, kmeans_iters=kmeans_iters)
    multiprocessing.set_start_method('spawn')
    with multiprocessing.Pool(thread) as pool:
        pool.starmap(process_task, zip(denoised_image_list, numbers))
    end_time = time.time()
    print("dict learning time:", end_time-start_time)

    clean_image_dir = "McM images"
    clean_image_name = "McM"
    clean_suffix = ".tif"
    dict_dir = "Task4"
    dict_name = "McM_dict_"
    dict_suffix = ".npy"

    clean_image_list, dict_list = [], []
    for number in numbers:
        print(f"-----loading clean No.{number} image-----")
        origin_path = osp.join(clean_image_dir, clean_image_name + number + clean_suffix)
        clean_image_list.append(plt.imread(origin_path).astype(np.float32)/255.0)

        print(f"-----loading learned dictionary No.{number} image-----")
        dict_path = osp.join(dict_dir, dict_name + number + dict_suffix)
        dict_clean = np.load(dict_path)
        dict_list.append(dict_clean.astype(np.float32))
    clean_image_list = np.array(clean_image_list)
    dict_list = np.array(dict_list)

    start_time = time.time()
    process_task = partial(image_denoise_rgb, patch_size=patch_size, step=step, sparsity_level=sparsity_level_denoise)
    with multiprocessing.Pool(thread) as pool:
        results = pool.starmap(process_task, zip(dict_list, clean_image_list, denoised_image_list, numbers))
    end_time = time.time()
    print("learning coefficient matrix time:", end_time-start_time)
    np.save('Task4/Task4_psnr.npy', results)