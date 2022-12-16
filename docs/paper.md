## Goal
  - Find a 21 joints in 2D RGB-image
  - Not only 2D joints coordinates, but 3D joints
  - Not wrist-attached camera view, but general camera view
  - Make the virtual hand image in general camera view(like almost open hand data)

## Dataset
### Train
- about 800K 2D virtual RGB images (can increase images as many as we want)
  - If it can train with only virtual image, we use both virtual and each open hand dataset
- not only 2D joints coordinates, but also 3D joints
- Futermore, wrist-attached camera view but general camera view
### Test
- FreiHAND (2019 ICCV) 
	- first-large scale hand dataset / 130K images / Adobe Research 
- InterHAND (2020 ECCV) 
	- first-interaction hand dataset / 2.6M images / Facebook Research 
- HIU-DMTL (2021 ICCV) 
	- annoated all images manually / 40K images 
- OneHand10K 
- Coco-WholeBody (2020 ECCV) 
- Coco 
	- low-resolution hand images 
- CMU Panoptic 
	- Carnegie Mellon Univ. 
	- but, low-resolution hand images 

## Previous Model
### lifting the 2D joint into 3D space
  -  Nearast neighbor matihing of a given 2D prediction
-  A probabilistic 3D pose model based upon PCA bases
  -  Etc.
### CNN-based
  - Learning to Estimate 3D Hand Pose From Single RGB Images (ICCV 2017)
    - First 3D Hand Pose Model
  	- HandSegNet -> [PoseNet(=CPMs)](#convolutional-pose-machines) -> PosePrior
				
### Transformer-based
  - Mesh Graphormer

## 2D Hand Pose Model
  - DeepPose
  - HourGlass
  - SimpleBaseline
  - HRNet
  - MeshGraphormer
  - MediaPipe (Pre-Train)

## 3D Hand Pose Model
	- HRNet
	- MediaPipe (Pre-Train)
	- MeshGraphormer

## Etc
### Convolutional Pose Machines
- 순차적 CNN 구조이며 CVPR 2016 발표
- Stage 마다 loss를 계산 
  - Vanishing Gradient 문제해결
- 매 stage마다 이전 stage의 belief map(=heatmap)을 입력으로 함께 넣어줌
  - 찾기 쉬운거 관절 먼저 찾아주고 찾은 관절을 참고로 다른 관절들을 찾아나감
- Stage에 conv & pooling을 반복하여 점차 receptive field를 넓힘
  - 자연스레 다른 관절과의 상관관계도 고려
### Hands deep in deep learning for hand pose estimation (Deep-Prior)
- ICCV 2017 Hands Workshop
- Bottleneck 구조를 활용
	- 마지막 FC에서 바로 21개의 joint를 directly regression 하는 것이 아닌 해당 개수나 차원보다 낮게 얻은 뒤 full pose representation으로 reconstruction함
	- Deep-Prior++에선 ResNet을 거친 뒤 나오는데 1024개의 벡터를 2번의 Drop-out을 거치게 한 뒤 joint coordinate를 구함