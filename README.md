# Wearable_Pose_Model ✨✨


This is our research code of [Our_Model](docs/2022_KSCI.pdf). 

Our Model is a transformer-based method for hand pose from an input image. 
This image is obtained from wrist-attached RGB camera.
In this work, We study how to use our synthetic images.

 <img src="docs/model.png" width="650"> 

## Installation(Download)
Build as the below architecture 
```
{$ROOT}
|-- build
|-- src
|-- datasets
|-- models
|-- docs
```

## Installation(Conda)
```bash
git clone https://github.com/leejeongho3214/Wearable_Pose_Model.git
cd Wearable_Pose_Model
conda env create -f requirements.yaml
```

## Model Download
Please download our files that are important to run our code. [download](https://dkuniv-my.sharepoint.com/:f:/g/personal/72210297_dankook_ac_kr/Em6dacaP1AlNmTAmaBbX2osBxkTx8km8k7BeHT2d-TWF5A?e=poqt1A)

## Train
```
run 'src/tools/train.py'
```
## Solution
If it happens error message about path as "src/...", you can insert the below code.
```python
import sys
sys.path.append("/usr/your/path/Wearable_Pose_Model")
```

## Result
 <img src="docs/result1.png" width="650"> 
 <img src="docs/result2.png" width="650"> 
 <img src="docs/result3.png" width="650"> 

## Compare(ing)
### Full FreiHAND Dataset
1st, 2nd column: images

3rd column: distance between g.t and prediction

4th column: a ratio of synthetic image in dataset
|FreiHAND(real)|CISLAB(synthetic)|Total|ratio|MPJPE(mm)|
|:----------------:|:-----------------:|:------:|:------:|:------:|
|120,000|0| 120,000|0%|4.71|
|120,000|5,000|125,000|4%| 3.03|
|120,000|10,000| 130,000|7.7%|2.77|
|120,000|20,000| 140,000|14.3%|2.92|
|120,000|30,000| 150,000|20%|2.82|
|120,000|40,000|160,000|25%|3.07|
|120,000|50,000|170,000|30%|2.80|
|120,000|60,000|180,000|33.3%| 2.91|    
|120,000|70,000|190,000|36.8%| 2.75|
|120,000|80,000|200,000|40%| 2.86|
|120,000|90,000|210,000|42.9%| 2.99|
|120,000|100,000|220,000|45.5%| |
|120,000|110,000|230,000|47.8%| |
|120,000|120,000|240,000|50%|2.91|

### Total 120k Dataset
|FreiHAND(real)|CISLAB(synthetic)|Total|MPJPE(mm)|
|:----------------:|:-----------------:|:------:|:------:|
|120,000|0| 120,000|4.71|
|110,000|10,000| 120,000|**2.35**|
|100,000|20,000|120,000| 2.75|


### Some part of Dataset
Total images are 10,000 images

below number means ratio of dataset
stop to train when count is 50

|Frei|CIS|error|
|:--:|:--:|:--:|
|100|0|8.38|
|95|5|6.73|
|90|10|**2.95**|
|85|15|3.11|
|80|20|3.11|
|75|25|3.42|
|70|30|3.92|
|65|35|3.69|
|60|40|2.97|
|55|45|3.87|
|50|50|3.49|
|45|55|3.28|
|40|60|3.39|

### Visualize

<img src="docs/visualize(some_part).png" width="600"> 

