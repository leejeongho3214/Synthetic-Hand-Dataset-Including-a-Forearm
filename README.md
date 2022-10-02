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
Evaluation Metric: PCK(Percentage of Correct Key-point)

We use 5 Trehshold values\
T = [0.1, 0.2, 0.3, 0.4 ,0.5]

|Model|Train_data|Threshold|PCK|
|-----|------|------|------|
|MediaPipe|Unknown|0.1|62.70|
|MediaPipe|Unknown|0.2|83.52|
|MediaPipe|Unknown|0.3|90.57|
|MediaPipe|Unknown|0.4|94.65|
|MediaPipe|Unknown|0.5|97.28|
|MMPose|OneHand10k|0.1|75.85|
|MMPose|OneHand10k|0.2|88.63|
|MMPose|OneHand10k|0.3|91.91|
|MMPose|OneHand10k|0.4|93.82|
|MMPose|OneHand10k|0.5|95.31|
|Our_Model|HIU_Full|0.1|29.22|
|Our_Model|HIU_Full|0.2|57.13|
|Our_Model|HIU_Full|0.3|71.96|
|Our_Model|HIU_Full|0.4|82.15|
|Our_Model|HIU_Full|0.5|88.70|
|Our_Model|Frei100k|0.1|47.37|
|Our_Model|Frei100k|0.2|73.81|
|Our_Model|Frei100k|0.3|88.20|
|Our_Model|Frei100k|0.4|94.50|
|Our_Model|Frei100k|0.5|98.15|
|Our_Model|Frei120k|0.1|55.53|
|Our_Model|Frei120k|0.2|84.31|
|Our_Model|Frei120k|0.3|94.94|
|Our_Model|Frei120k|0.4|98.64|
|Our_Model|Frei120k|0.5|99.75|
|Our_Model|Frei120k+Synthetic5k|0.1|72.33|
|Our_Model|Frei120k+Synthetic5k|0.2|97.78|
|Our_Model|Frei120k+Synthetic5k|0.3|99.51|
|Our_Model|Frei120k+Synthetic5k|0.4|99.88|
|Our_Model|Frei120k+Synthetic5k|0.5|100|





## Visualize

