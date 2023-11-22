# PaddlePaddle example for Zhouyi NPU

This repo contains 8 classification models, 2 OCR models and 1 segmentation model, all the models use int8 quantization and infer on development board.
The steps for testing each model:
- Floating Paddle model inference, get result as reference
- TVM model compile, TVM model infer on remote hardware board(with Zhouyi NPU inside) through RPC, get quantized model inference result
- TVM C++ app run on remote hardware board, get profile data
- Calculate NPU utilizatioin, accuracy & latency data


## Chip & Development Board
Theses examples run on SemiDrive development board which is based on SemiDrive SOC chip X9SP(with Zhouyi NPU X1 inside)


## Software package & Docker image
Plese contact ARM China [Technical support](Support-AI@armchina.com]) to get the required software & docker image.



Software package includes :
```
workspace_path
|-- Paddle-example-for-Zhouyi-NPU   # model/demo package
|-- AI610-SDK-r1p2-00eac0           # Zhouyi Compasss SDK
|-- Paddle-example-for-Zhouyi_docker_images.tar  # docker image
|-- dataset                         # original dataset direcotory, is's empty by default. If needed, please follow readme to download dataset 
```
Currently these modes are tested on Zhouyi Compass SDK r1p2 version.

## Dataset
The dataset under directory `dataset` can be downloaded per the instruction in `autorun.sh` in each folder of `Classification-example,OCR-example,Segmentation-example`, or directly use the selected 100/500 images in the following directory:
```
Classification-example/dataset_prepare/dataset  # 500 images
OCR-example/ocr_det/dataset_prepare/dataset     # 100 images
OCR-example/ocr_rec/dataset_prepare/dataset     # 100 images
Segmentation-example/dataset_prepare/dataset    # 100 images
````

The original `dataset` directory looks like:
```
dataset/
├── ILSVRC2012_val           # ImageNet dataset
├── ILSVRC2012_val_list.txt
├── mini_supervisely         # Segmentation dataset
├── train_full_images_0      # OCR Detection dataset
├── train_full_labels.json
├── train_images             # OCR Recognition dataset
└── train.list
```

## Folder topology of Paddle-example-for-Zhouyi-NPU
The folder topology of `Paddle-example-for-Zhouyi-NPU` is shown as below，`device_package.tar.gz` is decompressed in path `/data/project/` on the hardware board, if decompressed in other path, please modify the `scp ` commmand in `autorun.sh` accordingly. By default, `/data/project/` already has the decompressed package.

```
tar -zxvf device_package.tar.gz -C /data/project
```
```
Paddle-example-for-Zhouyi-NPU
├── Classification-example # Directory for testing classification model 
├── OCR-example            # Directory for testing OCR model 
├── Segmentation-example   # Directory for testing Segmentatio model 
├── Demo                   # Demo app code used on remote development board
├── device_package.tar.gz  # The code on development board when testing model, including TVM, runtime library, etc.
├── env_setup.sh           # Environment configuration
├── remote_IP_PORT.sh      # IP port config for development board 
└── README.md

```


## Steps to run model
### 1 Hardware configuration

Prepare one X86 host machine, deploy Linux OS like Ubuntu, power on SemiDrive development board and configure network to get IP address. Please make sure X86 host machine can communicate with development board through network, X86 host machine can SSH login development board. The host mainly focus on environment setup, floating Paddle model inference and model compiling, the development board mainly focus on NPU inference.

### 2 Software configuration
#### Host configuration
- Start docker container

Go to the path of `workspace_path`
```
# workspace_path
# |-- Paddle-example-for-Zhouyi-NPU
# |-- AI610-SDK-r1p2-00eac0
# |-- dataset
# |-- Paddle-example-for-Zhouyi_docker_images.tar

cd <path_to_workspace_path>
```

```
# load Container
sudo docker load -i Paddle-example-for-Zhouyi_docker_images.tar

# Share the workspace_path and `/mnt` in docker container
sudo docker run -it  -v $PWD:/mnt paddle-example-for-zhouyi_env:v1 /bin/bash
```

- Env setup in container 
```
source /root/.bashrc
conda activate env_zhouyi

# Enter the directory of Paddle-example
cd /mnt/Paddle-example-for-Zhouyi-NPU
source env_setup.sh
```

- Config IP and Port of remote development board, Host will connect remote board through TVM RPC
```
# Get remote board IP and config here
vim remote_IP_PORT.sh
DEVICE_IP="10.188.72.21"
PORT=9090
```

- Container configurate SSH login development board without password(By default, the development board has been configured to login without password, if not, please follow instruction below) 
```
# In container
ssh-keygen -t rsa 
cat ~/.ssh/id_rsa.pub >> authorized_keys # copy authorized_keys to development board in path `~/.ssh/`
```

#### Configuration on development board
```
# Host(not in container)  SSH login development board like:
ssh root@10.188.72.1 # (modify to real board IP)
# user:root, passwd:root123

# Enter directory of `/data/project/device_package` (device_package.tar.gz decompressed in /data/project/)
cd /data/project/device_package

# Start TVM RPC server
bash start_tvm_rpc_server.sh
```
### 3 Model inference test
Enter directory of `Paddle-example-for-Zhouyi-NPU` in container, start to test model inference as below:
#### 3.1 Classification model

There are 8 models in `Classification-example` directory, all the models have same process to test, take ResNet50 for example, please execute below command:
```
# On host, in the path of `Classification-example`, auto run ResNet50 
bash autorun.sh ResNet50
```
When finished, terminal will print top1, top5 accuracy.

Regarding the detailed steps of inference, please refer to autorun.sh. The 8 supported models are: `DLA34，DenseNet121，MobileNetV1，MobileNetV2，PeleeNet，ResNeXt50_32x4d，ResNet50，HarDNet68 `

#### 3.2 OCR model
In the path of `OCR-example`, enter ocr_det、ocr_rec ditectory respectively, and run below command.
```
# On host, in the path of `OCR-example`
cd ocr_det
bash autorun.sh
```
When finished, terminal will print precision，recall, hmean for OCR detection model

```
# on host端，in the path of `OCR-example` 
cd ocr_rec
bash autorun.sh
```
When finished, terminal will print accuracy for OCR recognition model  

#### 3.3 Segmentation model
Enter the path of `Segmentation-example`

```
# On host，in the path of `Segmentation-example`
cd Segmentation-example
bash autorun.sh
```
When finished, terminal will print dice_coef and iou value, the visible result comparison can be seen in path `tvm_rpc_infer/zhouyi_infer_result/zhouyi_infer_x.jpg` and `paddle_infer/paddle_infer_result/paddle_infer_x.jpg`.
  
  

#### 3.4 Demo run on development board
`Demo/demo_segmentation_device.tar.gz` is decompressed in path `/data/project`，compile code on development board

```
# Path：/data/project/demo_segmentation_device
# Compile
export PKG_CONFIG_PATH=`pwd`/pkgconfig
make

# Execute
export LD_LIBRARY_PATH=/data/opencv-4.5.5/build/lib:`pwd`/runtime_lib:$LD_LIBRARY_PATH
./csi-test -d video-evs0
```
Or directly run `bash autorun.sh`,  the app will get image from camera, then NPU do human segmentation model inference, after post-process the result will show on screen.




## Explanation for model inference process
All these explanation are same with the description in autorun.sh, take classification model as example:

- Prepare dataset

The directory `dataset_prepare/dataset` already has selected 500 images. If needed, please follow below instruction to download original dataset

```
cd dataset_prepare 
# Downloand ILSVRC2012_val dataset from ImageNet(https://www.image-net.org/) 
# ILSVRC2012_val.zip is decompressed in path `/mnt/dataset`，also label ILSVRC2012_val_list.txt is placed in path `/mnt/dataset`

# Random select 500 images and place in path `dataset`
python3 random_select_image.py

# Preprocess 500 images in path `dataset/` and save in path `preprocessed_data`
python3 preprocess_data.py

# Generate calibration dataset needed by Zhouyi NPU int8 quantization
generate_calibration_dataset.py

```
- Download model

`model` directory already has model, if needed, please follow below instruction to download
```
wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/model-name_infer.tar
# Decompress to `model` path, the name format under `model` directory: model-name/model inference file
```

- Paddle model inference
```
cd ../paddle_infer

# Run Paddle model inference
python3 paddle_infer.py ResNet50
```

- TVM model compile & export, NPU inference 
```
cd ../tvm_rpc_infer

# Delete cache data generated by last time compiling
if [ -e compass_output_paddle_model_* ]; then
  echo "cache file exists,remove"
  rm -rf compass_output_paddle_model_*
fi

# TVM compile Paddle model, and export aarch64 version model that can run on development board
python3 export_model.py ResNet50

# Model infernce on remote development board through TVM RPC
python3 tvm_rpc_run_device.py ${DEVICE_IP} ${PORT}
```

- Cross compile TVM C++ code, NPU inference, get profile data
```
cd ../tvm_c++_device_infer
cd build
rm -r *
cmake ..
make

# Generate input.bin, used by C++ inference app
cd ..
python3 generate_input.py
```
```
# Transmit inference app, input file, tvm model & script to remote board
cd ..
scp tvm_c++_device_infer/build/profile_test \
    tvm_c++_device_infer/input.bin \
    tvm_rpc_infer/tvm_model_aarch64.so \
    tvm_c++_device_infer/device_run.sh \
    root@${DEVICE_IP}:/data/project/device_package/tvm_c++_device_infer

# Run inference app through SSH
ssh root@${DEVICE_IP} "cd /data/project/device_package/tvm_c++_device_infer; bash device_run.sh"

# Copy latency.txt file from remote board to local `profile` directory
scp root@${DEVICE_IP}:/data/project/device_package/tvm_c++_device_infer/latency.txt ./profile

# Copy profile_data.bin from remote board to local `profile` directory 
scp root@${DEVICE_IP}:/data/project/device_package/tvm_c++_device_infer/compass_output/tvmgen_default_aipu_compass_main_0/runtime/profile_data.bin ./profile/

# Copy graph.json & parse profile_data.bin, get profile html format report
find tvm_rpc_infer/compass_output_paddle_*/tvmgen_default_aipu_compass_main_0/gbuilder/ -name 'graph.json' | xargs -I {} cp {} ./profile/
aipu_profiler profile/graph.json profile/profile_data.bin -o profile/npu_profile.html -f 750

```

- Calculate accuracy and utilization
```
cd profile
# Compare Paddle model inference result and Zhouyi's
python3 accuracy.py

# Calculate utilization based on inference time, model parameter number, NPU MAC number and NPU frequency 
python3 utilization.py
```

## Folder topology of classification model
There are almost the same topology for `Classification-example, OCR-example and Segmentation-example`, take ResNet50 as example：
```
Classification-example
|-- model                   # model
|-- model_cfg               # model config file, needed by TVM
|-- autorun.sh              # auto run script
|-- paddle_infer            # Paddle inference directory
|   |-- paddle_infer.py     # model infer on images
|   |-- paddle_infer_result.npy # inference result
|-- tvm_rpc_infer                       # TVM infer on remote board through RPC, get Zhouyi inference result
|   |-- compass_output_paddle_model_*   # Generated files when exporting model, which has infos about compiling
|   |-- export_model.py                 # compile model and export 
|   |-- tvm_model_aarch64.so            # exported model which can run on ARM board with Zhouyi NPU
|   |-- tvm_rpc_run_device.py           # TVM run inference on remote board through RPC
|   |-- zhouyi_infer_result.npy         # NPU inference result 
|-- tvm_c++_device_infer                # TVM C++ inference code on development board, get profile data
|   |-- CMakeLists.txt                  # CMake file
|   |-- build                           # directory for compiling
|   |-- device_run.sh                   # script to run on development board
|   |-- generate_input.py               # generate input file
|   |-- input.bin                       # input file
|   |-- profile_test.cc                 # source code
|-- profile                             
|   |-- accuracy.py                     # code to calculate accuracy
|   |-- graph.json                      # json file copyed from compass_output_paddle_model_*
|   |-- latency.txt                     # latency file copyed from development board after tvm_c++_device_infer execution
|   |-- npu_profile.html                # profile report using Zhouyi NPU tool
|   |-- profile_data.bin                # profile data copyed from development board after tvm_c++_device_infer execution
|   |-- utilization.py                  # calculate NPU utilization
|-- dataset_prepare
|   |-- random_select_image.py              # random select images from ImageNet dataset
|   |-- preprocess_op.py                    # preprocess op code  
|   |-- preprocess_data.py                  # preprocess images in `dataset`
|   |-- data_list.txt                       # selected images name and label list
|   |-- dataset                             # selected images
|   |-- preprocessed_data                   # preprocessed data, .npy format
|   |-- generate_calibration_dataset.py     # generate calibration dataset
|   |-- calibration_dataset.npy             # calibration dataset
```

## Benchmark
### Classification model

| Model name | Input size | Accuracy<br>(Floating model) |  Accuracy<br>(Quantization model) |
| :-----  | :-----   | :-----  |:----- |
| ResNet50 | [1,3,224,224] | Top1: 0.768<br> Top5: 0.922 | Top1: 0.764<br> Top5: 0.914 |
| ResNeXt50_32x4d | [1,3,224,224] | Top1: 0.774<br> Top5: 0.93 | Top1: 0.772<br> Top5: 0.92 |
| DenseNet121 | [1,3,224,224] | Top1: 0.754<br> Top5: 0.926 | Top1: 0.736<br> Top5: 0.928 |
| HarDNet68 | [1,3,224,224] | Top1: 0.764<br> Top5: 0.918 | Top1: 0.76<br> Top5: 0.91 |
| DLA34 | [1,3,224,224] | Top1: 0.774<br> Top5: 0.924 | Top1: 0.774<br> Top5: 0.928 |
| PeleeNet | [1,3,224,224] | Top1: 0.712<br> Top5: 0.896 | Top1: 0.704<br> Top5: 0.902 |
| MobileNetV1 | [1,3,224,224] | Top1: 0.706<br> Top5: 0.886 | Top1: 0.702<br> Top5: 0.87 |
| MobileNetV2 | [1,3,224,224] | Top1: 0.688<br> Top5: 0.9 | Top1: 0.674<br> Top5: 0.892 |

### OCR model
| Model name | Input size | Accuracy<br>(Floating model) | Accuracy<br>(Quantization model) |
| :-----  | :-----   | :-----  |:----- |
| ch_ppocr_server_v2.0_det_infer | [1,3,960,608] | precision: 0.817<br> recall: 0.648<br> hmean: 0.698 | precision: 0.807<br> recall: 0.645<br> hmean: 0.691 |
| ch_ppocr_server_v2.0_rec_infer | [1,3,32,320] | acc: 0.98 | acc: 0.97 |

### Segmentation model
| Model name | Input size | Accuracy<br>(Floating model)  | Accuracy<br>(Quantization model) |
| :-----  | :-----   | :-----  |:----- |
| PP-HumanSegV2-Mobile | [1,3,192,192] | dice_coef: 0.895<br> iou: 0.831 | dice_coef: 0.899<br> iou: 0.836 |
