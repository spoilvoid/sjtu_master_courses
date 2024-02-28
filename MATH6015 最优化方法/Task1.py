import os
import os.path as osp
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

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


if __name__ == "__main__":
    patch_size = 8
    step = 2
    num_atoms = 256
    k = 16
    kmeans_iters = 10
    dict_update_iters = 500
    sparsity_level = 5

    image_dir = "Grayscale images"
    image_name = ["barbara", "cameraman", "lena"]
    gray_suffix = "_512.png"
    index = 0
    if not osp.exists('Task1'):
        os.makedirs('Task1')
    for name in image_name:
        clean_path = osp.join(image_dir, name + gray_suffix)
        # Load the clean image
        clean_image = plt.imread(clean_path)
        clean_image = np.array(clean_image).astype(np.float32)
        dictionary, X = dictionary_learning(clean_image, patch_size, step, num_atoms, dict_update_iters, sparsity_level, k, kmeans_iters)
        np.save(f"Task1/{name}_dict.npy", dictionary)  # the value within[0,1]

