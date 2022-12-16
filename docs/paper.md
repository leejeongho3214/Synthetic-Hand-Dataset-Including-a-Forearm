## Goal
  1. Find a 21 joints in 2D RGB-image
  2. Not only 2D joints coordinates, but 3D joints
  3. Not wrist-attached camera view, but general camera view
  4. Make the virtual hand image in general camera view(like almost open hand data)

## Our Data
  1. Train-image: about 800K 2D virtual RGB images (can increase images as many as we want)
     - If it can train with only virtual image, we use both virtual and each open hand dataset
  1. not only 2D joints coordinates, but also 3D joints
  2. Futermore, wrist-attached camera view but general camera view

## Dataset
  1. Train-image: about 800K 2D virtual RGB images (can increase images as many as we want)
  2. Test-image: any open hand dataset 
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
  1. lifting the 2D joint into 3D space
     -  Nearast neighbor matihing of a given 2D prediction
     -  A probabilistic 3D pose model based upon PCA bases
     -  Etc.
  2. CNN-based
      - Learning to Estimate 3D Hand Pose From Single RGB Images (ICCV 2017)
        - First 3D Hand Pose Model
        - HandSegNet -> PoseNet(=CPM) -> PosePrior Network
  3.  

## Model
  1. DeepPose
  2. HourGlass
  3. SimpleBaseline
  4. HRNet
  5. Mesh-Graphormer

## Model(pre-train)
  1. MediaPipe


## Etc
1. Convolutional Pose Machines(CVPR 2016)
- s
3. heart
