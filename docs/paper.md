# Goal
- 한 장의 2D RGB image에서 3D 관절의 좌표를 딥러닝 모델을 통해 추론
- 일반적인 카메라로 촬영된 공개된 손 데이터셋에서 우리가 제작한 데이터로 학습한 모델이 가장 우수한 성능이 보이게 함
</br></br></br>

# Contibution
- 우리의 가상 이미지를 사용함으로써, 기존의 다른 데이터 셋에서 3D joint를 얻는 방식들보다 정확한 3D joint 추론이 가능하게 함
- 또한, 이미지를 원하는만큼 빠르게 제작이 가능하며 피실험자가 하기 힘든 손 동작들도 쉽게 취득 가능함
</br></br></br>

# Data
## Train
- 약 88만 장의 가상 이미지를 제작
- 2D와 3D 관절의 좌표 annotation을 가짐

## Test
- FreiHAND (2019 ICCV) 
	- first-large scale hand dataset / 130K images / Adobe Research 
- InterHAND (2020 ECCV) 
	- first-interaction hand dataset / 2.6M images / Facebook Research 
- HIU-DMTL (2021 ICCV) 
	- annoated all images manually / 40K images 
- Coco-WholeBody (2020 ECCV) 
	- low-resolution hand images 
- CMU Panoptic 
	- Carnegie Mellon Univ. 
	- but, low-resolution hand images 
- OneHand10K
</br></br></br> 

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

## 3D Hand Pose (RGB)
### Model-Free
- Learning to Estimate 3D Hand Pose From Single RGB Images (ICCV 2017)
  	- 처음으로 딥러닝 방법으로 3D hand pose를 regression
  	- HandSegNet -> [PoseNet(=CPMs)](#convolutional-pose-machines) -> PosePrior
		- 3D coordinate의 여러 추론 문제점들을 해결
			- A scale ambiguity 
				- 짝을 이루는 관절들의 뼈 길이(l2 norm)를 구한 뒤 coordinate에 나눠줌
			- Translation invariant representation
				- root 관절을 다른 관절에 빼줌으로써 위치에 무관한 coordinate를 얻어줌
- 3D Hand Shape and Pose from Images in the Wild (CVPR 2019)
	- 인코더를 통해 MANO 모델의 Pose, Beta 값과 Camera View 파라미터를 얻음
	- MANO 모델을 통해 3D joint와 동시에 Camera View 파라미터로 2D projection 시켜 2D & 3D joint를 얻음
		- Camera View 파라미터: Rotation, Translation, Scale
- Weakly Supervised 3D Hand Pose Estimation via Biomechanical Constraints (2020)
	- The biomechanical constraint를 사용하며 weakly supervised learning
	- 즉, 뼈의 길이, root bone 사이의 각도, 손가락 bone의 각도 등을 제약을 둬서 학습
		
### Model-based
- Mesh Graphormer
	- ViT와 다르게 입력을 패치가 아닌 BackBone인 HRNet의 feature map을 넣어줌
	- MANO 모델을 통해 인접행렬과 3D joint & mesh template의 초기화 값을 얻어줌 (beta와 pose값을 0을 넣어주어 얻은 값)
	- 3개의 인코더 블럭 중 마지막 인코더에 graph conv를 사용
		- graph conv는 해당 모델에서 3d mesh를 추론하므로 778개의 관절에 대해 학습이 되겠지만 2D & 3D joint 추론하는 모델에서 쓸 수 없을 것으로 보임
</br></br></br>

# Reference
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
## Hand pose estimation via latent 2.5 d heatmap regression
- ECCV 2018 발표, Nvidia Research
- 3D pose에서 scale과 depth 모호성의 문제를 해결하기 위해 2.5D pose representation을 제안
	- 2.5D에서 3D를 추론하는 방법이 위 문제로 언급된 모호성과 translation invariant를 해결
	- 뼈의 길이는 3D space에서 항상 같은 값을 가지므로 normalize한 값들을 absoulute 3D space로 reconstruct 가능케함
- 2.5D heatmap을 제안
	- 2D pose estimation에서 기본이 되는 heatmap 추론방식을 3D에서 사용하면 voxel heatmap의 연산량이 너무 커짐
	- 2D heatmap + depth heatmap
	- softargmax 방법을 사용하여 2D heatmap에서 coordinate를 얻어냄
## Crossing Nets: Combining GANs and VAEs with a Shared Latent Space for Hand Pose Estimation
- 3D hand pose for depth image // 2017년 CVPR 발표
- Semi-supervised Learning
- Deep Generative Model (GAN & VAE)를 사용
## Cross-modal Deep Variational Hand Pose Estimation
- 3D hand pose for semi-supervised learning // 2018년 CVPR 발표
- GAN & VAE를 사용
</br></br></br>

# 논문 읽으면서 배운 것들
## Likelihoood
- 딥러닝 관점으로 살펴보면, 만약 classification task에서 숫자를 판별하는 모델을 제작할 때, 정답이 0부터 9까지 있다고 하자. </br> 그리고 3개의 모델을 구현했을 때, 해당 모델들의 마지막 softmax layer에서 확률 값을 뽑아보면 0에서 9까지의 수로 판별할 확률이 나오게 되고 우리는 이 모델 중에서 가장 데이터를 잘 설명하는 distribution을 찾으며 찾은 모델이 가장 높은 likelihood을 갖는다.</br>또한, 학습을 통해 모델의 likelihood를 최대화하는 것이 목적이며 그러기위해 최적의 모델 파라미터 θ를 찾아줌
- Gaussian distribution(정규분포)를 확률 모형으로 대부분 표현함
	- 관찰된 전체 데이터 집합이 평균을 중심으로 하여 뭉쳐져 있는 형태를 표현하는데 적합
	- 평균과 분산만 알면 표현할 수 있음
- 주로 log likelihood를 최대화하는 방법을 사용
	- log를 붙임으로써 매우 작은 숫자도 표현하며, 곱셈을 덧셈으로 변환 가능하며, 가우시안 분포 식에서 지수를 날려버릴 수 있음
- Discrete vs Continuous
	- 이산확률변수에서 특정 사건이 일어날 확률은 likelihood가 되며, 연속확률변수에서는 PDF(Probability Density Function)으로 특정 구간에 속할 확률만 구해줄 수 있어 PDF의 y값을 likelihood라고 정의함
	- 만약, 정규분포를 따르는 PDF에서 0부터 100까지의 수 가운데 50을 뽑을 확률은 1/∞에 해당하는 "0"이지만 likelihood로는 해당 PDF의 y값으로 표현가능하여 여러 사건이 일어날 경우, 해당 likelihood를 곱해주면 해당 여러사건들이 일어날 가능도가 계산
## Posterior
- Maxlimum Likelihood Estimation은 철저히 데이터만을 가지고 구하는 반면에, 데이터와 더불어 우리가 갖고 있는 사전 지식까지 </br> 반영하고 싶다면 Maximum A Posterior를 이용
- 우리가 사전 지식을 갖고 있다면, 모델의 파라미터 w를 구하는데 도움이 됌
	- 예들 들어, 키 x에 따른 몸무게 y를 구할 때 몸무게 y>0라는 제약을 걸어 줄 수 있음
## Manifold learning
- 고차원 데이터가 있을 때, 차원축소를 위해 사용되며 이를 통해 고차원 데이터를 저차원에서도 잘 표현하는 공간인 manifold를 찾아 차원을 축소시킴
	- 고차원 상에서는 데이터 포인트가 의미적으로 가까워 보일 수 있으나 실제로는 거리가 먼 경우가 있음
## Generative Model
- 학습 데이터가 주어졌을 때, 학습 데이터가 가지는 실제 분포와 같은 분포에서 샘플링된 값으로 새로운 데이터를 생성하는 모델
### VAE (The Variational Auto-Encoders)
- AE에서 latent space 값이 하나의 값인 반면에, VAE는 평균과 분산으로 표현되는 어떠한 가우시안 분포
- 오토인코더(AE)와 목적이 전혀 다름 
	- AE: 어떤 데이터를 잘 압축, 특징을 잘 뽑아냄, 차원을 잘 줄임..
	- VAE: Generative model로써 어떤 새로운 데이터를 만들어내는 것
### GAN (The Generative Adversarial Networks)
- Generator와 Discriminator 사이의 경쟁을 통해 학습
- VAE는 차원축소용으로 많이 쓰이는 한편, GAN은 새로운 이미지를 만들어낼 때 주로 사용
- 또한, VAE는 노이즈를 주입하고 복원이 완벽하지 않기 때문에 결과는 GAN이 더 좋음
## Multimodal Learning
- 인간의 5가지 감각기관으로부터 수집되는 다양한 형태의 데이터를 사용하여 모델을 학습
	- Vision / Text / Speech / Touch / Smell (보통 이미지랑 text를 같이 쓰겠지)
	- 각각의 데이터의 특성을 잘 통합
		- 데이터 차원의 통합
			- 다른 특성의 데이터를 임베딩하여 특성이 같은 데이터로 추출
		- 모델 차원의 통합
			- 각기 다른 모델의 예측값을 통합 (각 모델은 다른 가중치를 가지고 통합)
## Evidence of Lower BOund (ELBO)
- variational lower bound라고도 불림
- 우리가 관찰한 P(z|x)가 다루기 힘든 분포를 이루고 있을 때, 비교적 쉬운 분포인 Q(x)로 대신 표현하려 하는 과정에서 </br> 두 분포 P(z|x)와 Q(x)의 차이(KL Divergence)를 최소화하기 위해 사용
- 결국, log p(x)는 E_q[log p(x,z)] - E_q[log q(z)]보다 크거나 같다고 함
	- Lower bound가 되는 이유 => ELBO = E_q[log p(x,z)] - E_q[log q(z)]
