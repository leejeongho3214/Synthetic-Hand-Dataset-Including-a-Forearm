# Evaluation

## Table of contents

- [Evaluation](#evaluation)
  - [Table of contents](#table-of-contents)
- [2D hand pose](#2d-hand-pose)
  - [Test on our testset](#test-on-our-testset)
    - [Training with our dataset](#training-with-our-dataset)
    - [Training with other dataset](#training-with-other-dataset)
  - [Etc](#etc)
    - [Performance according to ratio of aug](#performance-according-to-ratio-of-aug)
    - [Performance accoriing to kind of aug](#performance-accoriing-to-kind-of-aug)
- [3D hand pose](#3d-hand-pose)
  - [Test on each dataset](#test-on-each-dataset)
    - [Training with our dataset](#training-with-our-dataset-1)
    - [Training with other dataset](#training-with-other-dataset-1)
 
# 2D hand pose
## Test on our testset
### Training with our dataset
- All of these model trained with 300K our image
  
|     Model      | PCKb@0.1 | EPE(mm) |
| :------------: | :------: | :-----: |
|   MediaPipe    |  65.10   |  3.14   |
|   Hourglass    |  79.67   |  3.03   |
|     HRNet      |  79.73   |  3.22   |
| SimpleBaseline |  84.33   |  2.70   |
|      Ours      | <b>88.59 | <b>2.08 |

</br>

### Training with other dataset
- All of these dataset used the same model as our model

|   Model   | images | PCKb@0.1 | EPE(mm) |
| :-------: | :----: | :------: | :-----: |
|   Coco    |  36K   |   6.95   |  10.55  |
| InterHAND |  300K  |   7.78   |  11.44  |
| Panoptic  |  13K   |  10.79   |  10.17  |
|    RHD    |  37K   |  14.22   |  9.52   |
| HIU-DMTL  |  36K   |  23.27   |  9.12   |
| FreiHAND  |  120K  |  69.67   |  2.95   |
|   Ours    |  111K  | <b>88.59 | <b>2.03 |

</br>

## Etc
### Performance according to ratio of aug
| Model | Ratio of aug | PCKb@0.1 | EPE(mm) |
| :---: | :----: | :------: | :-----: |
| Ours  |  0.1   | 81.4925  |   2.64      | 
| Ours  |  0.2   | 76.5375  |      2.84   | 
| Ours  |  0.3   |  79.74   |        2.68 | 
| Ours  |  0.4   | 84.2275  |       2.32  | 
| Ours  |  0.5   |  82.75   |        2.46| 
| Ours  |  0.6   |   84.4   |         2.26|  
| Ours  |  0.7   |  84.485  |         2.31| 
| Ours  |  0.8   |  84.48   |         2.32| 
| Ours  |  0.9   | <b> 85.855  |        2.18 | 
| Ours  |   1    |  82.635  |         2.45| 

</br>

### Performance accoriing to kind of aug
| Model | Catrgory of aug | PCKb@0.1 | EPE(mm) |
| :---: | :----: | :------: | :-----: |
| Ours  | X                          | 27.60 | 6.17|
| Ours  | rot                        | 56.94 | 4.47|
| Ours  | color                      | 44.27 | 4.85|
| Ours  | erase                      | 29.87 | 5.86|
| Ours  | blur                       | 19.59 | 7.79|
| Ours  | rot & color                | <b>82.75 |<b> 2.46|
| Ours  | rot & color & blur         | 82.18 | 2.53|
| Ours  | rot & color & blur & erase | 80.44 | 2.55|

# 3D hand pose
## Test on each dataset
### Training with our dataset

### Training with other dataset
   