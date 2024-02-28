# Image Denoising with Sparse Coding
`组长：叶增渝`  `组员：周凡淇、罗颖特、潘沐晨`  

环境安装：需要安装anaconda3作为基础依赖，按照如下命令进行环境配置安装
```
conda create -n cvxopt python==3.11
conda activate cvxopt
pip install tqdm
pip install numpy
pip install matplotlib
pip install scipy
pip install h5py
```

根目录下文件相对关系
```
cvxopt
 |─ Grayscale images
 │   |─ barbara_512.png
 │   |─ cameraman_512.png
 │   └─ lena_512.png
 |─ McM images
 │   |─ McM01.tif
 │   |─ McM01_noise.mat
 :   :
 │   |─ McM18.tif
 │   └─ McM18_noise.mat
 |─ Task1.py
 |─ Task2.py
 |─ Task3.py
 |─ Task4.py
 |─ BM3D.py
 └─ README.md
```
`以下任务默认配置在Intel(R) Xeon(R) Gold 5416S下运行，计算默认运行时间`  

(1) Task1  单通道黑白图像字典学习
```
python Task1.py
```
默认参数预计学习时间8h，字典在./Task1文件夹中以npy形式存储
可以进入对应源文件中修改部分参数，建议修改参数:  
patch_size  字典块长与宽  
step  分割字典块时的步长  
num_atoms  学习的字典元素个数  
k kmeans  初始化时的中心数量  
kmeans_iters  kmeans的最大循环次数  
dict_update_iters  字典学习中字典更新最大循环次数  
sparsity_level  ksvd计算系数矩阵的最大循环次数  


(2) Task2  RGB三通道彩色图像字典学习
```
python Task2.py
```
默认参数预计学习时间36h，字典在./Task2文件夹中以npy形式存储。
可以进入对应源文件中修改部分参数，建议修改参数:  
patch_size  字典块长与宽  
step  分割字典块时的步长  
num_atoms  学习的字典元素个数  
k kmeans  初始化时的中心数量  
kmeans_iters  kmeans的最大循环次数  
dict_update_iters  字典学习中字典更新最大循环次数  
sparsity_level  ksvd计算系数矩阵的最大循环次数  
thread 使用MapReduce加速并行，同时进行的分任务数量  


(3) Task3  已知干净字典的RGB三通道彩色图像降噪
```
python Task3.py
```
默认参数预计降噪时间8h，降噪后的图像在./Task3文件夹中以png形式存储
可以进入对应源文件中修改部分参数，建议修改参数:  
patch_size  字典块长与宽  
step  分割字典块时的步长  
sparsity_level  sparse_encode重构图像的最大循环次数  
thread  使用MapReduce加速并行，同时进行的分任务数量  


(4) Task4  无ground truth的RGB三通道彩色图像降噪
```
python BM3D.py
python Task3.py
```
BM3D默认参数预计降噪时间4h，初步降噪后的图像在./BM3D images文件夹中以png形式存储；然后使用字典学习进行降噪默认参数预计降噪时间44h，降噪后的图像在./Task4文件夹中以png形式存储，同时以npy形式保存对应的字典学习结果。
可以进入对应源文件中修改部分参数，建议修改参数:  
BM3D：  
sigma  
hard_threshold_3D_ratio  
first_match_threshold  
second_match_threshold  
beta_Kaiser  
max_match_count_1st  
block_size_1st  
block_step_1st  
search_step_1st  
search_window_1st  
max_match_count_2nd  
block_size_2nd  
block_step_2nd  
search_step_2nd  
search_window_2nd  
Task4：  
patch_size  字典块长与宽  
step  分割字典块时的步长  
num_atoms  学习的字典元素个数  
k kmeans  初始化时的中心数量  
kmeans_iters  kmeans的最大循环次数  
dict_update_iters  字典学习中字典更新最大循环次数  
sparsity_level_learn  ksvd计算系数矩阵的最大循环次数  
sparsity_level_denoise  sparse_encode重构图像的最大循环次数  
thread 使用MapReduce加速，同时进行的分任务数量  

`tips:对应的psnr值在./Task3和./Task4中均有对应的f$task_name$_psnr.npy文件存储`

`已生成结果可以在Github项目https://github.com/Zengyu-Ye/image_denoise中查询`  
`由于我们的降噪结果以.png形式输出，虽然像素数值在0-1范围内，但是降噪后的图像像素值可能无法正好对应0-255点整数值，所以会出现PSNR值偏差`