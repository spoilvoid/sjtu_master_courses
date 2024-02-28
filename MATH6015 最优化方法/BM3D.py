import os
import os.path as osp
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.fftpack import dct, idct

import multiprocessing
from functools import partial


def PSNR(img1, img2):
    if img1.shape != img2.shape or len(img1.shape)!=2 or len(img1.shape)!=2:
        raise ValueError("incompatiable image shape")
    '''
    Calculate PSNR value between 2 images in a specific channel
    Args:
    img1, img2:[length, width]  pixel range[0, 255]  original image and denoised image(must be same shape)
    '''
    D = np.array(img1 - img2, dtype=np.int64)
    D[:, :] = D[:, :]**2
    RMSE = D.sum()/img1.size
    psnr = 10*math.log10(float(255.**2)/RMSE)
    return psnr


class BM3D_agent:
    def __init__(
            self,
            sigma = 15,
            hard_threshold_3D_ratio = 2.7,
            first_match_threshold = 2500,
            second_match_threshold = 400,
            beta_Kaiser = 2.0,
            max_match_count_1st = 16,
            block_size_1st = 8,
            block_step_1st = 3,
            search_step_1st = 3,
            search_window_1st = 39,
            max_match_count_2nd = 32,
            block_size_2nd = 8,
            block_step_2nd = 3,
            search_step_2nd = 3,
            search_window_2nd = 39,
    ):
        self.sigma = sigma
        self.hard_threshold_3D = hard_threshold_3D_ratio * sigma # hard threshold
        self.first_match_threshold = first_match_threshold # threshold of block similarity
        self.second_match_threshold = second_match_threshold
        self.beta_Kaiser = beta_Kaiser

        self.max_match_count_1st = max_match_count_1st # max number of group matching blocks 
        self.block_size_1st = block_size_1st # block_Size 2 dim
        self.block_step_1st = block_step_1st # horizontal and vertical block distance
        self.search_step_1st = search_step_1st # block search step
        self.search_window_1st = search_window_1st # local candidate blocks' search space

        self.max_match_count_2nd = max_match_count_2nd
        self.block_size_2nd = block_size_2nd
        self.block_step_2nd = block_step_2nd
        self.search_step_2nd = search_step_2nd
        self.search_window_2nd = search_window_2nd

    def denoise(self, noisy_image):
        basic_image = self._BM3D_1st_step(noisy_image)
        result_image = self._BM3D_2nd_step(basic_image, noisy_image)
        return result_image

    def _BM3D_1st_step(self, _noisyImg):
        """basic denoise"""
        # parameter init
        (width, height) = _noisyImg.shape
        block_Size = self.block_size_1st
        blk_step = self.block_step_1st
        Width_num = (width - block_Size)/blk_step
        Height_num = (height - block_Size)/blk_step
        Basic_img, m_Wight, m_Kaiser = self._setup(_noisyImg, block_Size, self.beta_Kaiser)

        # block iterative processing
        for i in range(int(Width_num+2)):
            for j in range(int(Height_num+2)):
                m_blockPoint = self._locate_blk(i, j, blk_step, block_Size, width, height)
                Similar_Blks, Positions, Count = self._step1_fast_match(_noisyImg, m_blockPoint)
                Similar_Blks, statis_nonzero = self._step1_3DFiltering(Similar_Blks)
                self._aggregation_HardThreshold(Similar_Blks, Positions, Basic_img, m_Wight, statis_nonzero, Count, m_Kaiser)
        Basic_img[:, :] /= m_Wight[:, :]
        basic = np.matrix(Basic_img, dtype=int)
        basic.astype(np.uint8)

        return basic
    
    def _BM3D_2nd_step(self, _basicImg, _noisyImg):
        '''improved grouping and collaborative Wiener filtering'''
        # parameter init
        (width, height) = _noisyImg.shape
        block_Size = self.block_size_2nd
        blk_step = self.block_step_2nd
        Width_num = (width - block_Size)/blk_step
        Height_num = (height - block_Size)/blk_step
        m_img, m_Wight, m_Kaiser = self._setup(_noisyImg, block_Size, self.beta_Kaiser)

        for i in range(int(Width_num+2)):
            for j in range(int(Height_num+2)):
                m_blockPoint = self._locate_blk(i, j, blk_step, block_Size, width, height)
                Similar_Blks, Similar_Imgs, Positions, Count = self._step2_fast_match(_basicImg, _noisyImg, m_blockPoint)
                Similar_Blks, Wiener_wight = self._step2_3DFiltering(Similar_Blks, Similar_Imgs)
                self._aggregation_Wiener(Similar_Blks, Wiener_wight, Positions, m_img, m_Wight, Count, m_Kaiser)
        m_img[:, :] /= m_Wight[:, :]
        Final = np.matrix(m_img, dtype=int)
        Final.astype(np.uint8)

        return Final

    def _setup(self, img, _blk_size, _Beta_Kaiser):
        """record and Kaiser windows init"""
        m_shape = img.shape
        m_img = np.matrix(np.zeros(m_shape, dtype=float))
        m_wight = np.matrix(np.zeros(m_shape, dtype=float))
        K = np.matrix(np.kaiser(_blk_size, _Beta_Kaiser))
        m_Kaiser = np.array(K.T * K)
        return m_img, m_wight, m_Kaiser

    def _locate_blk(self, i, j, blk_step, block_Size, width, height):
        '''block bound checking'''
        if i*blk_step+block_Size < width:
            point_x = i*blk_step
        else:
            point_x = width - block_Size

        if j*blk_step+block_Size < height:
            point_y = j*blk_step
        else:
            point_y = height - block_Size

        # left-up point of block
        m_blockPoint = np.array((point_x, point_y), dtype=int)

        return m_blockPoint

    def _define_SearchWindow(self, _noisyImg, _BlockPoint, _WindowSize, Blk_Size):
        """define Search Window and return vertex coordinates"""
        point_x = _BlockPoint[0]
        point_y = _BlockPoint[1]

        # point calculate
        LX = point_x+Blk_Size/2-_WindowSize/2
        LY = point_y+Blk_Size/2-_WindowSize/2
        RX = LX+_WindowSize                       
        RY = LY+_WindowSize                       

        # bound check
        if LX < 0:   LX = 0
        elif RX > _noisyImg.shape[0]:   LX = _noisyImg.shape[0]-_WindowSize
        if LY < 0:   LY = 0
        elif RY > _noisyImg.shape[0]:   LY = _noisyImg.shape[0]-_WindowSize

        return np.array((LX, LY), dtype=int)

    def _step1_fast_match(self, _noisyImg, _BlockPoint):
        '''
        return several neighbour blocks with highest similarity
        Args:
        _noisyImg: image with noise
        _BlockPoint: block size and coordinate
        '''
        (present_x, present_y) = _BlockPoint
        Blk_Size = self.block_size_1st
        Search_Step = self.search_step_1st
        Threshold = self.first_match_threshold
        max_matched = self.max_match_count_1st
        Window_size = self.search_window_1st

        blk_positions = np.zeros((max_matched, 2), dtype=int)  # 用于记录相似blk的位置
        Final_similar_blocks = np.zeros((max_matched, Blk_Size, Blk_Size), dtype=float)

        img = _noisyImg[present_x: present_x+Blk_Size, present_y: present_y+Blk_Size]

        # block 2-dim cosine transform
        dct_img = dct(dct(img.astype(np.float64).T, norm='ortho').T, norm='ortho')

        Final_similar_blocks[0, :, :] = dct_img
        blk_positions[0, :] = _BlockPoint

        Window_location = self._define_SearchWindow(_noisyImg, _BlockPoint, Window_size, Blk_Size)
        # calculate block number
        blk_num = (Window_size-Blk_Size)/Search_Step  
        blk_num = int(blk_num)
        (present_x, present_y) = Window_location

        similar_blocks = np.zeros((blk_num**2, Blk_Size, Blk_Size), dtype=float)
        m_Blkpositions = np.zeros((blk_num**2, 2), dtype=int)
        Distances = np.zeros(blk_num**2, dtype=float)

        # slide window iterative search
        matched_cnt = 0
        for i in range(blk_num):
            for j in range(blk_num):
                tem_img = _noisyImg[present_x: present_x+Blk_Size, present_y: present_y+Blk_Size]
                dct_Tem_img = dct(dct(tem_img.astype(np.float64).T, norm='ortho').T, norm='ortho')
                m_Distance = np.linalg.norm((dct_img-dct_Tem_img))**2 / (Blk_Size**2)

                if m_Distance < Threshold and m_Distance > 0:
                    similar_blocks[matched_cnt, :, :] = dct_Tem_img
                    m_Blkpositions[matched_cnt, :] = (present_x, present_y)
                    Distances[matched_cnt] = m_Distance
                    matched_cnt += 1
                present_y += Search_Step
            present_x += Search_Step
            present_y = Window_location[1]
        Distances = Distances[:matched_cnt]
        Sort = Distances.argsort()

        # matched block number
        if matched_cnt < max_matched:
            Count = matched_cnt + 1
        else:
            Count = max_matched

        if Count > 0:
            for i in range(1, Count):
                Final_similar_blocks[i, :, :] = similar_blocks[Sort[i-1], :, :]
                blk_positions[i, :] = m_Blkpositions[Sort[i-1], :]
        return Final_similar_blocks, blk_positions, Count

    def _step1_3DFiltering(self, _similar_blocks):
        '''
        3D transformation, threshold filtering in frequency domain, inverse transformation
        Args:
        _similar_blocks:similar blocks in frequency domain
        '''
        statis_nonzero = 0  # non-zero elements number
        m_Shape = _similar_blocks.shape

        for i in range(m_Shape[1]):
            for j in range(m_Shape[2]):
                tem_Vct_Trans = dct(_similar_blocks[:, i, j].T, norm='ortho').reshape(-1,1)
                tem_Vct_Trans[np.abs(tem_Vct_Trans[:]) < self.hard_threshold_3D] = 0.
                statis_nonzero += tem_Vct_Trans.nonzero()[0].size
                
                _similar_blocks[:, i, j] = idct(tem_Vct_Trans.reshape(-1).T, norm='ortho').reshape(-1,1)[0]
        return _similar_blocks, statis_nonzero

    def _aggregation_HardThreshold(self, _similar_blocks, blk_positions, m_basic_img, m_wight_img, _nonzero_num, Count, Kaiser):
        '''
        stacks' weight sum multiple Kaiser rate as basic image
        Args:
        _similar_blocks:similar blocks in frequency domain
        '''
        _shape = _similar_blocks.shape
        if _nonzero_num < 1:
            _nonzero_num = 1
        block_wight = (1./_nonzero_num) * Kaiser
        for i in range(Count):
            point = blk_positions[i, :]
            tem_img = (1./_nonzero_num)*idct(idct(_similar_blocks[i, :, :].T, norm='ortho').T, norm='ortho') * Kaiser
            m_basic_img[point[0]:point[0]+_shape[1], point[1]:point[1]+_shape[2]] += tem_img
            m_wight_img[point[0]:point[0]+_shape[1], point[1]:point[1]+_shape[2]] += block_wight

    def _step2_fast_match(self, _Basic_img, _noisyImg, _BlockPoint):
        '''
        fast matching, find neighbour blocks with highest similarity
        Args:
        _Basic_img: image after basic denoising
        _noisyImg: image with noise
        _BlockPoint: block coordinates and size
        '''
        (present_x, present_y) = _BlockPoint
        Blk_Size = self.block_size_2nd
        Threshold = self.second_match_threshold
        Search_Step = self.search_step_2nd
        max_matched = self.max_match_count_2nd
        Window_size = self.search_window_2nd

        blk_positions = np.zeros((max_matched, 2), dtype=int)
        Final_similar_blocks = np.zeros((max_matched, Blk_Size, Blk_Size), dtype=float)
        Final_noisy_blocks = np.zeros((max_matched, Blk_Size, Blk_Size), dtype=float)

        img = _Basic_img[present_x: present_x+Blk_Size, present_y: present_y+Blk_Size]
        dct_img = dct(dct(img.astype(np.float32).T, norm='ortho').T, norm='ortho')
        Final_similar_blocks[0, :, :] = dct_img

        n_img = _noisyImg[present_x: present_x+Blk_Size, present_y: present_y+Blk_Size]
        dct_n_img = dct(dct(n_img.astype(np.float32).T, norm='ortho').T, norm='ortho')
        Final_noisy_blocks[0, :, :] = dct_n_img

        blk_positions[0, :] = _BlockPoint

        Window_location = self._define_SearchWindow(_noisyImg, _BlockPoint, Window_size, Blk_Size)
        blk_num = (Window_size-Blk_Size)/Search_Step
        blk_num = int(blk_num)
        (present_x, present_y) = Window_location

        similar_blocks = np.zeros((blk_num**2, Blk_Size, Blk_Size), dtype=float)
        m_Blkpositions = np.zeros((blk_num**2, 2), dtype=int)
        Distances = np.zeros(blk_num**2, dtype=float)

        matched_cnt = 0
        for i in range(blk_num):
            for j in range(blk_num):
                tem_img = _Basic_img[present_x: present_x+Blk_Size, present_y: present_y+Blk_Size]
                dct_Tem_img = dct(dct(tem_img.astype(np.float32).T, norm='ortho').T, norm='ortho')
                m_Distance = np.linalg.norm((dct_img-dct_Tem_img))**2 / (Blk_Size**2)

                if m_Distance < Threshold and m_Distance > 0:
                    similar_blocks[matched_cnt, :, :] = dct_Tem_img
                    m_Blkpositions[matched_cnt, :] = (present_x, present_y)
                    Distances[matched_cnt] = m_Distance
                    matched_cnt += 1
                present_y += Search_Step
            present_x += Search_Step
            present_y = Window_location[1]
        Distances = Distances[:matched_cnt]
        Sort = Distances.argsort()

        if matched_cnt < max_matched:
            Count = matched_cnt + 1
        else:
            Count = max_matched

        if Count > 0:
            for i in range(1, Count):
                Final_similar_blocks[i, :, :] = similar_blocks[Sort[i-1], :, :]
                blk_positions[i, :] = m_Blkpositions[Sort[i-1], :]

                (present_x, present_y) = m_Blkpositions[Sort[i-1], :]
                n_img = _noisyImg[present_x: present_x+Blk_Size, present_y: present_y+Blk_Size]
                Final_noisy_blocks[i, :, :] = dct(dct(n_img.astype(np.float64).T, norm='ortho').T, norm='ortho')

        return Final_similar_blocks, Final_noisy_blocks, blk_positions, Count

    def _step2_3DFiltering(self, _Similar_Bscs, _Similar_Imgs):
        '''
        Collaborative filtering for 3D Wiener transform
        Args:
        _similar_blocks: similar blocks in frequency domain
        '''
        m_Shape = _Similar_Bscs.shape
        Wiener_wight = np.zeros((m_Shape[1], m_Shape[2]), dtype=float)

        for i in range(m_Shape[1]):
            for j in range(m_Shape[2]):
                tem_vector = _Similar_Bscs[:, i, j]
                tem_Vct_Trans = np.matrix(dct(tem_vector.T, norm='ortho').reshape(-1,1))
                Norm_2 = np.float64((tem_Vct_Trans.T * tem_Vct_Trans).item())
                m_weight = Norm_2/(Norm_2 + self.sigma**2)
                if m_weight != 0:
                    Wiener_wight[i, j] = 1./(m_weight**2 * self.sigma**2)
                # else:
                #     Wiener_wight[i, j] = 10000
                tem_vector = _Similar_Imgs[:, i, j]
                tem_Vct_Trans = m_weight * dct(tem_vector.T, norm='ortho').reshape(-1,1)

                _Similar_Bscs[:, i, j] = idct(tem_Vct_Trans.reshape(-1).T, norm='ortho').reshape(-1,1)[0]

        return _Similar_Bscs, Wiener_wight

    def _aggregation_Wiener(self, _Similar_Blks, _Wiener_wight, blk_positions, m_basic_img, m_wight_img, Count, Kaiser):
        '''
        Perform weighted sum on stack after 3D transformation and filtering
        Args:
        _similar_blocks:similar blocks in frequency domain
        '''
        _shape = _Similar_Blks.shape
        block_wight = _Wiener_wight # * Kaiser

        for i in range(Count):
            point = blk_positions[i, :]
            
            tem_img = _Wiener_wight * idct(idct(_Similar_Blks[i, :, :].T, norm='ortho').T, norm='ortho') # * Kaiser
            m_basic_img[point[0]:point[0]+_shape[1], point[1]:point[1]+_shape[2]] += tem_img
            m_wight_img[point[0]:point[0]+_shape[1], point[1]:point[1]+_shape[2]] += block_wight


def BM3D(origin_path, noise_path, number, sigma = 15, hard_threshold_3D_ratio = 2.7, first_match_threshold = 2500, second_match_threshold = 400, beta_Kaiser = 2.0, max_match_count_1st = 16, block_size_1st = 8, block_step_1st = 3, search_step_1st = 3, search_window_1st = 39, max_match_count_2nd = 32, block_size_2nd = 8, block_step_2nd = 3, search_step_2nd = 3, search_window_2nd = 39):
    if not osp.exists('BM3D images'):
        os.makedirs('BM3D images')
    channels = ['red', 'green', 'blue']
    print(f"-----loading original No.{number} image-----")
    image_origin = plt.imread(origin_path)
    image_origin_per_channel = image_origin.transpose(2,0,1)
    print(f"-----loading noisy No.{number} image-----")
    image_noise = h5py.File(noise_path)['u_n'][:].transpose(2,1,0).astype(np.float32)
    image_noise_per_channel = image_noise.transpose(2,0,1)

    result_per_channel = []
    denoise_agent = BM3D_agent(sigma = sigma, hard_threshold_3D_ratio = hard_threshold_3D_ratio, first_match_threshold = first_match_threshold, second_match_threshold = second_match_threshold, beta_Kaiser = beta_Kaiser, max_match_count_1st = max_match_count_1st, block_size_1st = block_size_1st, block_step_1st = block_step_1st, search_step_1st = search_step_1st, search_window_1st = search_window_1st, max_match_count_2nd = max_match_count_2nd, block_size_2nd = block_size_2nd, block_step_2nd = block_step_2nd, search_step_2nd = search_step_2nd, search_window_2nd = search_window_2nd)
    start_time = time.time()
    for color, channel_noise in zip(channels, image_noise_per_channel):
        print(f"-----start No.{number} image's BM3D in {color} channel-----")
        Final_img = denoise_agent.denoise(channel_noise)
        result_per_channel.append(Final_img)
        print(f"-----finish No.{number} image's BM3D in {color} channel-----")

    end_time = time.time()
    process_time = end_time - start_time
    print(f"-----No.{number} image's BM3D takes {process_time} s-----")

    psnr_list = []
    for color, image_origin, image_result in zip(channels, image_origin_per_channel, result_per_channel):
        psnr = PSNR(image_origin, image_result)
        psnr_list.append(psnr)
    print(f"-----No.{number} image's PSNR:{psnr_list}-----\n")

    result_image = np.clip(np.array(result_per_channel).transpose(1,2,0),0, 255).astype(np.uint8)
    plt.imsave(f"BM3D images/McM{number}_denoise.png", result_image)

    return psnr_list

if __name__ == '__main__':
    image_dir = "McM images"
    image_name = "McM"
    origin_suffix = ".tif"
    noise_suffix = "_noise.mat"
    numbers = [f"{i:02d}" for i in range(1, 19)]


    sigma = 15
    hard_threshold_3D_ratio = 2.7
    first_match_threshold = 2500
    second_match_threshold = 400
    beta_Kaiser = 2.0
    max_match_count_1st = 16
    block_size_1st = 8
    block_step_1st = 3
    search_step_1st = 3
    search_window_1st = 39
    max_match_count_2nd = 32
    block_size_2nd = 8
    block_step_2nd = 3
    search_step_2nd = 3
    search_window_2nd = 39
    
    thread = 18
    
    origin_path_list, noise_path_list = [], []
    for number in numbers:
        origin_path_list.append(osp.join(image_dir, image_name+number+origin_suffix))
        noise_path_list.append(osp.join(image_dir, image_name+number+noise_suffix))

    process_task = partial(BM3D, sigma = sigma, hard_threshold_3D_ratio = hard_threshold_3D_ratio, first_match_threshold = first_match_threshold, second_match_threshold = second_match_threshold, beta_Kaiser = beta_Kaiser, max_match_count_1st = max_match_count_1st, block_size_1st = block_size_1st, block_step_1st = block_step_1st, search_step_1st = search_step_1st, search_window_1st = search_window_1st, max_match_count_2nd = max_match_count_2nd, block_size_2nd = block_size_2nd, block_step_2nd = block_step_2nd, search_step_2nd = search_step_2nd, search_window_2nd = search_window_2nd)
    multiprocessing.set_start_method('spawn')
    with multiprocessing.Pool(thread) as pool:
        results = pool.starmap(process_task, zip(origin_path_list, noise_path_list, numbers))
    
    np.save('BM3D images/BM3D_psnr.npy', results)
