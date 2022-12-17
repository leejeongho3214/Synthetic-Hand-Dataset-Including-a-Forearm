# Goal
  - Find a 21 joints in 2D RGB-image
  - Not only 2D joints coordinates, but 3D joints
  - Not wrist-attached camera view, but general camera view
  - Make the virtual hand image in general camera view(like almost open hand data)

# Dataset
## Train
- about 800K 2D virtual RGB images (can increase images as many as we want)
  - If it can train with only virtual image, we use both virtual and each open hand dataset
- not only 2D joints coordinates, but also 3D joints
- Futermore, wrist-attached camera view but general camera view
## Test
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

# Model
## 2D Hand Pose
### Direct regression
- DeepPose (CVPR 2014)
- 각 stage의 regressor로부터 얻은 좌표를 토대로 다음 stage에 해당 좌표를 기준으로 한 bbox를 입력으로 넣어주어 iterative하게 관절의 위치를 추정
### HeatMap
- HourGlass (ECCV 2016)
	- 하나의 hourglass는 대칭적 구조를 가지며, down & upsampling을 거쳐 local과 global feature를 얻을 수 있음
	- 8개의 hourglass를 이어 붙이며, 그 사이에는 intermediate supervision을 해줌으로써 점차 refinement되는 효과
- SimpleBaseline (ECCV 2018)
	- ResNet 네트워트 output에 decov만 해주는 아주 간단한 네트워크
	- 좋은 high resolution feature를 얻는게 좋지만, 그 방법들이 성능에 주는 차이는 미미
- HRNet (CVPR 2019)
	- 지금까지 네트워크들과 다르게 직렬이 아닌 병렬 추론방식
	- 즉, down & upsampling을 전혀 거치지 않아 온전하게 high resolution을 유지하였고 병렬로 처리한 low resolution feature 또한 fusion 해주어 성능이 우수

## 3D Hand Pose
### Direct 3D pose estimation
- Learning to Estimate 3D Hand Pose From Single RGB Images (ICCV 2017)
  - First 3D Hand Pose Model
  - HandSegNet -> [PoseNet(=CPMs)](#convolutional-pose-machines) -> PosePrior
### Lifting 2D pose to 3D pose
-  Nearast neighbor matihing of a given 2D prediction
-  A probabilistic 3D pose model based upon PCA bases
-  Etc.				
### MANO model based estimation
- Mesh Graphormer
	- ViT와 다르게 입력을 패치가 아닌 BackBone인 HRNet의 feature map을 넣어줌
	- MANO 모델을 통해 인접행렬과 3D joint & mesh template의 초기화 값을 얻어줌 (beta와 pose값을 0을 넣어주어 얻은 값)
	- 3개의 인코더 블럭 중 마지막 인코더에 graph conv를 사용
		- graph conv는 해당 모델에서 3d mesh를 추론하므로 778개의 관절에 대해 학습이 되겠지만 2D & 3D joint 추론하는 모델에서 쓸 수 없을 것으로 보임



# Etc
## Convolutional Pose Machines
- 순차적 CNN 구조이며 CVPR 2016 발표
- Stage 마다 loss를 계산 
  - Vanishing Gradient 문제해결
- 매 stage마다 이전 stage의 belief map(=heatmap)을 입력으로 함께 넣어줌
  - 찾기 쉬운거 관절 먼저 찾아주고 찾은 관절을 참고로 다른 관절들을 찾아나감
- Stage에 conv & pooling을 반복하여 점차 receptive field를 넓힘
	- 뒤로 갈수록 나오는 belief map의 크기가 작아지며 그에 따른 관절의 위치에 해당하는 receptive field가 커짐
  - 자연스레 다른 관절과의 상관관계도 고려
## Hands deep in deep learning for hand pose estimation (Deep-Prior)
- ICCV 2017 Hands Workshop
- Bottleneck 구조를 활용
	- 마지막 FC에서 바로 21개의 joint를 directly regression 하는 것이 아닌 해당 개수나 차원보다 낮게 얻은 뒤 full pose representation으로 reconstruction함
	- Deep-Prior++에선 ResNet을 거친 뒤 나오는데 1024개의 벡터를 2번의 Drop-out을 거치게 한 뒤 joint coordinate를 구함
## Likelihoood
- 딥러닝 관점으로 살펴보면, 만약 classification task에서 숫자를 판별하는 모델을 제작할 때, 정답이 0부터 9까지 있다고 하자. 3개의 모델을 구현했을 때, 해당 모델들의 마지막 softmax layer에서 확률 값을 뽑아보면 0에서 9까지의 수를 출력할 확률이 나오게 되고 우리는 이 모델 중에서 가장 데이터를 잘 설명하는 distribution을 찾으며 찾은 모델이 가장 높은 likelihood을 갖는다. 또한, 학습을 통해 모델의 likelihood를 최대화하는 것이 목적이며 그러기위해 최적의 모델 파라미터 θ를 찾아줌