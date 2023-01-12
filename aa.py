from matplotlib import pyplot as plt
import matplotlib
import cv2
matplotlib.use('Webagg')
import torch
a = torch.ones(224, 224, 3)
b = plt.imread("0.jpg")
plt.imshow(a)
plt.imshow(b)
plt.show()
cv2.waitKey(0)
# sudo apt-get install python3-tk
