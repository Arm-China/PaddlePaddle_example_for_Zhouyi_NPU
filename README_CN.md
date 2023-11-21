# PaddlePaddle example for Zhouyi NPU
## 模型测试说明

本次测试模型包含8个分类模型、2个OCR模型和1个分割模型，在开发板端模型推理使用int8量化的模型。所有模型测试步骤包括：
- paddle浮点模型推理，获取推理结果作为对比
- TVM模型编译、TVM RPC方式开发板端推理（int8量化，Zhouyi NPU推理），获取量化模型推理结果
- TVM C++代码开发板端推理，获取profile数据
- 计算NPU利用率、精度、latency数据


## 芯片和开发板
本次使用芯驰SemiDrive的开发板，基于X9SP芯片，内置Zhouyi NPU X1


## 软件包和docker镜像获取
请联系ARM China[技术支持](Support-AI@armchina.com])获取

软件包文件包括：
```
workspace_path
|-- Paddle-example-for-Zhouyi-NPU    # 模型/demo验证测试包
|-- AI610-SDK-r1p2-00eac0            # Zhouyi Compasss SDK
|-- Paddle-example-for-Zhouyi_docker_images.tar  # docker镜像
|-- dataset                          # 原始数据集目录，该目录为空，如果用户需要，按照readme下载
```
目前在Compass SDK r1p2版本测试验证过

## 数据集说明
dataset路径下的数据集，可以按照`Classification-example,OCR-example,Segmentation-example`文件下的autorun.sh里面的提示下载，也可以直接使用已经挑选好的100或500张图片，已经分别保存在
```
Classification-example/dataset_prepare/dataset  # 500 images
OCR-example/ocr_det/dataset_prepare/dataset     # 100 images
OCR-example/ocr_rec/dataset_prepare/dataset     # 100 images
Segmentation-example/dataset_prepare/dataset    # 100 images
````

原始数据集dataset目录:
```
dataset/
├── ILSVRC2012_val           # ImageNet数据集
├── ILSVRC2012_val_list.txt
├── mini_supervisely         # Segmentation数据集
├── train_full_images_0      # OCR Detection数据集
├── train_full_labels.json
├── train_images             # OCR Recognition数据集
└── train.list
```

## Paddle-example-for-Zhouyi-NPU文件夹结构
`Paddle-example-for-Zhouyi-NPU`文件夹的拓扑结构如下，`device_package.tar.gz`解压放在开发板上（开发板路径`/data/project/`下已经部署好该解压包）,如果解压在其他路径，autorun.sh脚本里面的scp 拷贝命令对应目录需要修改
```
tar -zxvf device_package.tar.gz -C /data/project
```
```
Paddle-example-for-Zhouyi-NPU
├── Classification-example # 分类模型测试目录
├── OCR-example            # OCR模型测试目录
├── Segmentation-example   # Segmentatio模型测试目录
├── Demo                   # demo用到的开发板侧代码
├── device_package.tar.gz  # 模型测试时，开发板侧部署的代码，包括TVM，runtimelib等
├── env_setup.sh           # 环境配置
├── remote_IP_PORT.sh      # 开发板IP port配置
└── README.md

```


## 测试步骤
### 1 硬件环境配置
准备一台X86 Host主机，部署Linux系统如Ubuntu，芯驰开发板上电启动并配置网络，获取IP地址，确保X86 Host主机和开发板能够以网络通信，X86 Host主机可以SSH登录开发板。开发环境搭建、Paddle模型推理、模型编译在Host端完成，开发板端主要完成NPU推理。

### 2 软件环境配置
#### Host主机配置
- 启动容器

进入以下workspace_path路径所在文件夹：
```
# workspace_path
# |-- Paddle-example-for-Zhouyi-NPU
# |-- AI610-SDK-r1p2-00eac0
# |-- dataset
# |-- Paddle-example-for-Zhouyi_docker_images.tar

cd <path_to_workspace_path>
```

```
# 加载容器
sudo docker load -i Paddle-example-for-Zhouyi_docker_images.tar

# 将上面的workspace_path目录文件夹和docker内部/mnt共享
sudo docker run -it  -v $PWD:/mnt paddle-example-for-zhouyi_env:v1 /bin/bash
```

- 容器内配置环境
```
source /root/.bashrc
conda activate env_zhouyi

# 进入测试包目录
cd /mnt/Paddle-example-for-Zhouyi-NPU
source env_setup.sh
```

- 配置TVM RPC远程连接开发板的IP和端口(端口默认不用修改)
```
# 获取远程开发板的IP，然后配置在这里
vim remote_IP_PORT.sh
DEVICE_IP="10.188.72.21"
PORT=9090
```

- 配置容器内ssh免密登录开发板（开发板已经配置为免密ssh登录，如果没有请在docker容器内如下配置）
```
ssh-keygen -t rsa 
cat ~/.ssh/id_rsa.pub >> authorized_keys # 将authorized_keys拷贝到开发板~/.ssh/路径下
```

#### 开发板配置
```
# Host(非容器内)通过SSH登录开发板:
ssh root@10.188.72.1 # 修改成实际开发板IP
# user:root, passwd:root123

# 进入目录（device_package.tar.gz解压的目录，这里开发板已经部署好）
cd /data/project/device_package

# 通过以下脚本启动TVM RPC server
bash start_tvm_rpc_server.sh
```
### 3 模型推理验证
在容器内进入`Paddle-example-for-Zhouyi-NPU`目录，开始以下模型测试。
#### 3.1 分类模型
`Classification-example`目录下提供8个分类模型用于测试,下面以ResNet50为例说明流程，其他7个分类模型测试流程一致。
```
# 在Host端，容器内`Classification-example` 路径下，直接一键运行ResNet50  
bash autorun.sh ResNet50
```
运行结束屏幕打印top1,top5精度值

关于推理的具体步骤，可以看autorun.sh里面的详细注释。请确保模型名字正确，支持验证的8个模型`DLA34，DenseNet121，MobileNetV1，MobileNetV2，PeleeNet，ResNeXt50_32x4d，ResNet50，HarDNet68`


#### 3.2 OCR模型
在`OCR-example`路径下，分别进入ocr_det、ocr_rec模型下运行，关于推理的具体步骤，可以看autorun.sh里面的详细注释。
```
# 在Host端，`OCR-example` 路径下
cd ocr_det
bash autorun.sh
```
运行结束屏幕打印OCR detection模型precision，recall,hmean值

```
# 在Host端，`OCR-example` 路径下, rec模型编译导出比较慢，耐心等待
cd ocr_rec
bash autorun.sh
```
运行结束屏幕打印OCR recognition模型精度值

#### 3.3 Segmentation模型

进入`Segmentation-example`模型下运行，主要是模型的编译和导出、Zhouyi NPU推理输出和paddle模型推理输出对比等。

```
# 在Host端，`Segmentation-example` 路径下
cd Segmentation-example
bash autorun.sh
```
最后屏幕会打印dice_coef和iou值，另外可以看Zhouyi NPU int8模型推理和paddle浮点模型推理结果直观对比，如`tvm_rpc_infer/zhouyi_infer_result/zhouyi_infer_x.jpg`和 `paddle_infer/paddle_infer_result/paddle_infer_x.jpg` 

  
  

#### 3.4 开发板demo运行
`Demo/demo_segmentation_device.tar.gz`解压在开发板如`/data/project`路径下，在开发板端本地编译代码

```
# 路径：/data/project/demo_segmentation_device
# 编译
export PKG_CONFIG_PATH=`pwd`/pkgconfig
make

# 执行
export LD_LIBRARY_PATH=/data/opencv-4.5.5/build/lib:`pwd`/runtime_lib:$LD_LIBRARY_PATH
./csi-test -d video-evs0
```

或者直接运行`bash autorun.sh`，启动后，从摄像头获取图像，经过预处理、NPU模型推理和后处理，分割提取的人像显示在与开发板连接的显示屏上。




## 模型推理步骤解释
模型推理步骤与autorun.sh里描述的一致，这里以分类模型为例，解释模型的推理步骤:

- 数据集准备

已经预先挑选、处理好数据在本地`dataset_prepare/dataset`目录下，下面的命令可以不执行。如有需要，按以下要求执行
```
cd dataset_prepare 
# 从ImageNet(https://www.image-net.org/) 下载ILSVRC2012_val数据集
# ILSVRC2012_val.zip解压放在/mnt/dataset目录下，以及标定文件ILSVRC2012_val_list.txt

# 随机选取500图片放在dataset目录下
python3 random_select_image.py

# 对选取的500张ILSVRC2012_val图片（dataset目录下）进行预处理，并保存在preprocessed_data文件夹下
python3 preprocess_data.py

# 生成Zhouyi NPU int8量化模型需要的标定数据集
generate_calibration_dataset.py

```
- 模型下载

`model`路径下已经下载了模型，如有需要按以下操作下载模型
```
wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/模型名称_infer.tar
# 解压到 `model` 目录下，model路径下文件命令格式： `模型名字/模型inference文件`
```

- paddle 模型推理
```
cd ../paddle_infer

# 运行paddle模型推理
python3 paddle_infer.py ResNet50
```


- TVM 模型编译导出，端侧推理
```
cd ../tvm_rpc_infer

# 删除之前的编译缓存数据
if [ -e compass_output_paddle_model_* ]; then
  echo "cache file exists,remove"
  rm -rf compass_output_paddle_model_*
fi

# TVM编译paddle模型，并导出可在开发板端运行的aarch64版本模型文件
python3 export_model.py ResNet50


# TVM通过RPC远程连接开发板进行推理
python3 tvm_rpc_run_device.py ${DEVICE_IP} ${PORT}
```

- TVM C++代码交叉编译，端侧推理，获取profile数据
```
cd ../tvm_c++_device_infer
cd build
rm -r *
cmake ..
make

# 生成input.bin,给C++推理使用
cd ..
python3 generate_input.py
```
```
# 将推理执行app、输入文件、tvm模型文件和脚本传输给开发板
cd ..
scp tvm_c++_device_infer/build/profile_test \
    tvm_c++_device_infer/input.bin \
    tvm_rpc_infer/tvm_model_aarch64.so \
    tvm_c++_device_infer/device_run.sh \
    root@${DEVICE_IP}:/data/project/device_package/tvm_c++_device_infer

# ssh远程执行推理代码
ssh root@${DEVICE_IP} "cd /data/project/device_package/tvm_c++_device_infer; bash device_run.sh"

# 从开发板将latency.txt文件传输回本地
scp root@${DEVICE_IP}:/data/project/device_package/tvm_c++_device_infer/latency.txt ./profile

# 从开发板将profile_data.bin文件传输回本地
scp root@${DEVICE_IP}:/data/project/device_package/tvm_c++_device_infer/compass_output/tvmgen_default_aipu_compass_main_0/runtime/profile_data.bin ./profile/

# 拷贝graph.json并解析profile_data.bin数据，获取profile html报告
find tvm_rpc_infer/compass_output_paddle_*/tvmgen_default_aipu_compass_main_0/gbuilder/ -name 'graph.json' | xargs -I {} cp {} ./profile/
aipu_profiler profile/graph.json profile/profile_data.bin -o profile/npu_profile.html -f 750

```

- 计算精度和利用率
```
cd profile
# 基于paddle模型推理结果和Zhouyi NPU推理结果进行比对
python3 accuracy.py

# 基于推理时间、模型参数量、NPU MAC和频率计算NPU利用率
python3 utilization.py
```

## 分类模型文件夹结构
`Classification-example/OCR-example/Segmentation-example`下每个文件夹基本一致，以ResNet50为例：
```
Classification-example
|-- model                   # 模型
|-- model_cfg               # 模型配置文件，TVM编译需要
|-- autorun.sh              # 一键运行脚本
|-- paddle_infer            # paddle模型推理目录
|   |-- paddle_infer.py     # 模型对500张图片推理，获取精度数据
|   |-- paddle_infer_result.npy #paddle模型推理结果保存的文件
|-- tvm_rpc_infer                       # TVM 通过RPC方式在开发板推理的代码，获取Zhouyi NPU推理结果
|   |-- compass_output_paddle_model_*   # 导出模型自动生成的文件夹，存有模型编译相关信息
|   |-- export_model.py                 # 导出TVM编译后的模型文件
|   |-- tvm_model_aarch64.so            # 导出的TVM模型文件
|   |-- tvm_rpc_run_device.py           # TVM通过RPC在开发板推理的脚本
|   |-- zhouyi_infer_result.npy         # TVM在开发板Zhouyi NPU推理的结果
|-- tvm_c++_device_infer                # TVM在开发板C++推理代码，获取profile数据
|   |-- CMakeLists.txt                  # CMake文件
|   |-- build                           # 编译目录
|   |-- device_run.sh                   # 在开发板的运行脚本
|   |-- generate_input.py               # 生成模型推理需要的输入文件
|   |-- input.bin                       # 模型的输入文件
|   |-- profile_test.cc                 # 源代码
|-- profile                             
|   |-- accuracy.py                     # 精度计算代码
|   |-- graph.json                      # 从compass_output_paddle_model_*拷贝的模型json文件
|   |-- latency.txt                     # 从tvm_c++_device_infer推理结束，开发板拷贝来的保存latency文件
|   |-- npu_profile.html                # Zhouyi NPU工具生成的profile文件
|   |-- profile_data.bin                # 从tvm_c++_device_infer推理结束，开发板拷贝来的profile文件
|   |-- utilization.py                  # 计算NPU利用率 
|-- dataset_prepare
|   |-- random_select_image.py              # 从ImageNet随机选取500张图片代码
|   |-- preprocess_op.py                    # 预处理的算子代码    
|   |-- preprocess_data.py                  # 对dataset里的数据进行预处理，保存在preprocessed_data    
|   |-- data_list.txt                       # 选取的500张ImageNet图片名字和label  
|   |-- dataset                             # 选取的500张图片
|   |-- preprocessed_data                   # 预处理过的数据, npy格式
|   |-- generate_calibration_dataset.py     # 生成标定数据集
|   |-- calibration_dataset.npy             # 模型编译器需要的标定数据集
```

## Benchmark
### 分类模型

| 模型名称 | 输入尺寸 | 浮点精度 | 量化精度 |
| :-----  | :-----   | :-----  |:----- |
| ResNet50 | [1,3,224,224] | Top1: 0.768<br> Top5: 0.922 | Top1: 0.764<br> Top5: 0.914 |
| ResNeXt50_32x4d | [1,3,224,224] | Top1: 0.774<br> Top5: 0.93 | Top1: 0.772<br> Top5: 0.92 |
| DenseNet121 | [1,3,224,224] | Top1: 0.754<br> Top5: 0.926 | Top1: 0.736<br> Top5: 0.928 |
| HarDNet68 | [1,3,224,224] | Top1: 0.764<br> Top5: 0.918 | Top1: 0.76<br> Top5: 0.91 |
| DLA34 | [1,3,224,224] | Top1: 0.774<br> Top5: 0.924 | Top1: 0.774<br> Top5: 0.928 |
| PeleeNet | [1,3,224,224] | Top1: 0.712<br> Top5: 0.896 | Top1: 0.704<br> Top5: 0.902 |
| MobileNetV1 | [1,3,224,224] | Top1: 0.706<br> Top5: 0.886 | Top1: 0.702<br> Top5: 0.87 |
| MobileNetV2 | [1,3,224,224] | Top1: 0.688<br> Top5: 0.9 | Top1: 0.674<br> Top5: 0.892 |


### OCR模型
| 模型名称 | 输入尺寸 | 浮点精度 | 量化精度 |
| :-----  | :-----   | :-----  |:----- |
| ch_ppocr_server_v2.0_det_infer | [1,3,960,608] | precision: 0.817<br> recall: 0.648<br> hmean: 0.698 | precision: 0.807<br> recall: 0.645<br> hmean: 0.691
 |
| ch_ppocr_server_v2.0_rec_infer | [1,3,32,320] | acc: 0.98 | acc: 0.97 |

### Segmentation模型
| 模型名称 | 输入尺寸 | 浮点精度 | 量化精度 |
| :-----  | :-----   | :-----  |:----- |
| PP-HumanSegV2-Mobile | [1,3,192,192] | dice_coef: 0.895<br> iou: 0.831 | dice_coef: 0.899<br> iou: 0.836 |