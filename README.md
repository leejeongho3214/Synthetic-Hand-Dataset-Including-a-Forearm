# Wearable_Pose_Model ✨✨


This is our research code of [Our_Model](docs/2022_KSCI.pdf). 

Our Model is a transformer-based for hand pose.</br>
The image is obtained from wrist-attached RGB camera.</br>
In this work, We study how to use our synthetic images instead of making real-world dataset.</br></br>

 <img src="docs/model.png" width="650"> </br></br>

## Installation(Download)</br>
Build as the below architecture 
```
{$ROOT}
|-- build
|-- src
|-- datasets
|-- models
|-- docs
```

## Installation(Conda)</br>
```bash
git clone https://github.com/leejeongho3214/Wearable_Pose_Model.git
cd Wearable_Pose_Model
conda env create -f requirements.yaml
```

## Model Download</br>
Please download our files that need to run our code. [download](https://dkuniv-my.sharepoint.com/:f:/g/personal/72210297_dankook_ac_kr/Em6dacaP1AlNmTAmaBbX2osBxkTx8km8k7BeHT2d-TWF5A?e=poqt1A)

## Train</br>
```
run 'src/tools/train.py'
```

## Solution</br>
If it happens error message about path as "src/...", you can insert the below code.
```python
import sys
sys.path.append("/usr/your/path/Wearable_Pose_Model")
```
## Result</br>
Evaluation Metric: PCK(Percentage of Correct Key-point) </br>
[eval.md](#docs/Evaluation.md)
