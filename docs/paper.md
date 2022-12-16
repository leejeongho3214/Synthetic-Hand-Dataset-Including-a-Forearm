## Goal
  - Find a 21 joints in 2D RGB-image
  - Not only 2D joints coordinates, but 3D joints
  - Not wrist-attached camera view, but general camera view
  - Make the virtual hand image in general camera view(like almost open hand data)

## Dataset
  - Train-image
     - about 800K 2D virtual RGB images (can increase images as many as we want)
       - If it can train with only virtual image, we use both virtual and each open hand dataset
     - not only 2D joints coordinates, but also 3D joints
     - Futermore, wrist-attached camera view but general camera view
  - Test-image: any open hand dataset 
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
  - lifting the 2D joint into 3D space
     -  Nearast neighbor matihing of a given 2D prediction
     -  A probabilistic 3D pose model based upon PCA bases
     -  Etc.
  - CNN-based
      - Learning to Estimate 3D Hand Pose From Single RGB Images (ICCV 2017)
        - First 3D Hand Pose Model
        - HandSegNet -> [PoseNet(=CPMs)](#convolutional-pose-machines(cvpr-2016)) -> PosePrior
				
  - Transformer-based
      - Mesh Graphormer

## Model
  - DeepPose
  - HourGlass
  - SimpleBaseline
  - HRNet
  - Mesh-Graphormer

## Model(pre-train)
  - MediaPipe


## Etc
### Convolutional Pose Machines(CVPR 2016)
- 순차적 CNN 구조
- Stage 마다 loss를 계산 
  - Vanishing Gradient overcome
- 매 stage마다 이전 stage의 belief map(=heatmap)을 입력으로 함께 넣어줌
  - 찾기 쉬운거 관절 먼저 찾아주고 찾은 관절을 참고로 다른 관절들을 찾아나감
- Stage에 conv & pooling을 반복하여 점차 receptive field를 넓힘
  - 자연스레 다른 관절과의 상관관계도 고려
### 
