import numpy as np
import imageio
import matplotlib.pyplot as plt
from scipy import ndimage


image_path = "E:\\Nanad\\Backup Redmi\\camera\\IMG_20230630_163141.jpg" 
image = imageio.imread(image_path)

sigma = 2  
blurred_image = ndimage.gaussian_filter(image, sigma=sigma)


plt.figure(figsize=(12, 6))


plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Gambar Asli')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(blurred_image)
plt.title('Gambar Setelah Filter Gaussian')
plt.axis('off')

plt.show()