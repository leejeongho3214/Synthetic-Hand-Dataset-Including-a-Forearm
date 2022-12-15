Goal
  1. Find a 21 joints in 2D RGB-image
  2. Not only 2D joints coordinates, but 3D joints
  3. Not wrist-attached camera view, but general camera view
  4. Make the virtual imaeg for general hand inference

Data
  1. Train-image: about 800K 2D virtual RGB images (can increase images as many as we want)
    => If it can train with only virtual image, we use both virtual and each open hand dataset
  2. Test-image: any open hand dataset
    - FreiHAND (2019 ICCV)
      => first-large scale hand dataset / 130K images / Adobe Research
    - InterHAND (2020 ECCV)
      => first-interaction hand dataset / 2.6M images / Facebook Research
    - HIU-DMTL (2021 ICCV)
      => annoated all images manually / 40K images
    - OneHand10K
    - Coco-WholeBody (2020 ECCV)
      => low-resolution hand images
    - CMU Panoptic
      => Carnegie Mellon Univ.
      => but, low-resolution hand images

Model
  1. DeepPose
  2. HourGlass
  3. SimpleBaseline
  4. HRNet
  5. Mesh-Graphormer

Model(pre-train)
  1. MediaPipe