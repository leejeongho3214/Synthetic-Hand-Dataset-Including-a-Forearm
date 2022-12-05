# Evaluation

## Table of contents

- [Evaluation](#evaluation)
  - [Table of contents](#table-of-contents)
- [2D hand pose](#2d-hand-pose)
  - [Training with our dataset](#training-with-our-dataset)
  - [Training with other dataset](#training-with-other-dataset)
  - [Training](#training)
 
# 2D hand pose
## Training with our dataset
- All of these model trained with 300K our image
  
|     Model      | PCKb@0.1 | EPE(mm) |
| :------------: | :-----: | :-----: |
|   MediaPipe    |  65.10  |  3.14   |
|   Hourglass    |  79.67  |  3.03   |
|     HRNet      |  79.73  |  3.22   |
| SimpleBaseline |  84.33  |  2.70   |
|      Ours      |  <b>88.59  |  <b>2.08   |

</br>

## Training with other dataset
- All of these dataset used the same model as our model

|   Model   | images|PCKb@0.1 | EPE(mm) |
| :-------: | :-----: |:-----: | :-----: |
|   Coco    | 36K| 6.95  |   10.55|
| InterHAND |  300K|7.78  |   11.44|
| Panoptic  |  13K|10.79  |  10.17 |
|    RHD    |  37K|14.22  |   9.52 |
| HIU-DMTL  |  36K|23.27  |  9.12 |
| FreiHAND  |  120K|69.67  |  2.95  |
|   Ours    |  111K|<b>88.59  |   <b>2.03|

</br>

## Training

   