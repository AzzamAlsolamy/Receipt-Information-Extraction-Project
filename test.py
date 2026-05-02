import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('/teamspace/studios/this_studio/Receipt-Information-Extraction-Project/Datasets/SROIE2019/train/img/X51008164997.jpg', cv.IMREAD_GRAYSCALE)

ret1, th1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
ret2, th2 = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
blur = cv.GaussianBlur(img, (5, 5), 0)
ret3, th3 = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

fig, axes = plt.subplots(1, 3, figsize=(15, 10))

axes[0].imshow(th1, 'gray')
axes[0].set_title('Global Thresholding (v=127)', fontsize=13)
axes[0].axis('off')

axes[1].imshow(th2, 'gray')
axes[1].set_title("Otsu's Thresholding", fontsize=13)
axes[1].axis('off')

axes[2].imshow(th3, 'gray')
axes[2].set_title("Otsu's Thresholding + Gaussian Blur", fontsize=13)
axes[2].axis('off')

plt.tight_layout()
plt.savefig('receipt_results_only3.png', dpi=150, bbox_inches='tight')
plt.show()