Goal
  1. Find a 21 joints in 2D RGB-image
<<<<<<< HEAD
  2. Not only 2D joints coordinates, but 3D joints
  3. Not wrist-attached camera view, but general camera view
  4. Make the virtual imaeg for general hand inference

Data
  1. Train-image: about 800K 2D virtual RGB images (can increase images as many as we want)
    => If it can train with only virtual image, we use both virtual and each open hand dataset
=======
  2. not only 2D joints coordinates, but also 3D joints
  3. Futermore, wrist-attached camera view but general camera view


Data
  1. Train-image: about 800K 2D virtual RGB images (can increase images as many as we want)
>>>>>>> ee48b7535f7d6607ebe26a23a08fa3a94ce0cd4d
  2. Test-image: any open hand dataset
    - FreiHAND (2019 ICCV)
      => first-large scale hand dataset / 130K images / Adobe Research
    - InterHAND (2020 ECCV)
      => first-interaction hand dataset / 2.6M images / Facebook Research
    - HIU-DMTL (2021 ICCV)
      => annoated all images manually / 40K images
<<<<<<< HEAD
    - OneHand10K
    - Coco-WholeBody (2020 ECCV)
=======
    - Coco
>>>>>>> ee48b7535f7d6607ebe26a23a08fa3a94ce0cd4d
      => low-resolution hand images
    - CMU Panoptic
      => Carnegie Mellon Univ.
      => but, low-resolution hand images
<<<<<<< HEAD

Model
  1. DeepPose
  2. HourGlass
  3. SimpleBaseline
  4. HRNet
  5. Mesh-Graphormer

Model(pre-train)
  1. MediaPipe
=======
>>>>>>> ee48b7535f7d6607ebe26a23a08fa3a94ce0cd4d
