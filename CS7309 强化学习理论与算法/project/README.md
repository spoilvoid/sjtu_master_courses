需要保证utils文件夹下文件不缺失并且文件相对位置正确，想要运行2个agent进行学习还需要在Linux系统下进行如下操作安装环境:
```
conda create -n RL python==3.11
conda activate RL
pip install gymnasium
pip gymnasium[accept-rom-license]
pip gymnasium[atari]
pip gymnasium[mujoco]
pip install matplotlib
pip3 install torch torchvision torchaudio
pip install tensorboard
pip install pillow
```

在根目录下直接运行如下命令进行RainbowDQN强化学习任务，其中可选环境名为'VideoPinball-ramNoFrameskip-v4', 'BreakoutNoFrameskip-v4', 'PongNoFrameskip-v4', 'BoxingNoFrameskip-v4'这4个选项:
```
python RainbowDQN.py --env_name VideoPinball-ramNoFrameskip-v4
```

然后在根目录下直接运行如下命令进行DDPG强化学习任务，其中可选环境名为'Hopper-v4', 'Humanoid-v4', 'HalfCheetah-v4', 'Ant-v4'这4个选项:
```
python DDPG.py --env_name Hopper-v4
```

可以使用如下命令进行可设定参数查看：
```
python RainbowDQN.py --help
python DDPG.py --help
```
最终模型会保存在~/models文件夹下，最终“奖励-episode曲线”保存在根目录下，图像名为'$model_name$_$task_name$.png'
