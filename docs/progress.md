# Research
## 22.12.17 - now
- 조합들 중에 가장 성능이 우수했던 가상 이미지 13만장만을 가지고 학습
	- Hourglass, HRNet, Simplebaseline
	- ratio of aug를 0.1, 0.2 모델을 재학습 (37만장)
	- ratio of aug 1.0 모델을 학습해서 과연 기존 0.6과의 차이가 있을지 학습

- 3D Hand Pose의 성능을 비교하기 위한 joint error를 확인
	- 2D와 다르게 상대적으로 복잡하여 FreiHAND의 Eval방식을 따르기로 함 (MeshGraphormer 또한 위 방법 사용)
	- our virtual image 800K 이미지로 학습해서 성능보기


# 3D Hand Pose Estimation
##  FreiHAND
> Evaluation Metric=>[more details](#error-calculation)
>> pck의 threshold를 0 ~ 5cm 사이에 동일한 간격으로 서로 다른 100개의 값 사용

>> auc는 regressor를 분류기의 정확도처럼 표현하기 위해 pck 그래프의 아래 면적 값을 의미
>>> 최댓값은 1이며, 서로 다른 threshold에 의해 나온 pck 평균 값으로 봐도 무방

- Ours
	- use FreiHAND 130K image

				mean3d_error: 1.10 cm
				auc3d: 0.781

	- use our virtual image 830K

- MeshGraphormer
	- use FreiHAND 130K image

				mean3d_error: 0.64 cm
				auc3d: 0.873


# Reference
## Error Calculation
- Procrustes Analysis (침대에 팔, 다리를 강제로 맞춤)
	1. gt, pred의 평균 값을 계산
	2. 각 평균값으로 빼줌 => 벡터들의 원점을 기준으로 이동
	3. 각 l2 norm을 계산하여 나눠줌 => normalization
	4. pred를 gt와 동일한 scale과 orientation을 갖게 하기위해 변환행렬을 구해줌
	5. 4에서 구한 변환행렬을 pred에 적용해 aligned pred를 얻음
- PA 적용 후 MPJPE 측정
	- PA를 적용함으로써 회전과 스케일을 제외하고 오로지 자세의 차이만 계산
