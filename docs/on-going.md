# Research
## 22.12.17 - now
- 조합들 중에 가장 성능이 우수했던 가상 이미지 13만장만을 가지고 학습
	- Hourglass, HRNet, Simplebaseline
	- ratio of aug를 0.1, 0.2 모델을 재학습 (37만장)
	- ratio of aug 1.0 모델을 학습해서 과연 기존 0.6과의 차이가 있을지 학습

- 3D Hand Pose의 성능을 비교하기 위한 joint error를 확인
	- 2D와 다르게 상대적으로 복잡하여 FreiHAND의 Eval방식을 따르기로 함 (MeshGraphormer 또한 위 방법 사용)


# 3D Hand Pose Estimation
##  FreiHAND
- Ours
	- use FreiHAND 130K image

				mean3d_error: 1.10 cm
				auc3d: 78.1 %

	- use our virtual image 830K