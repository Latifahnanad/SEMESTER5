import imageio
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve


def sobel_edge_detection(image):
    sobel_x = np.array([[1, 0, -1],
                         [2, 0, -2],
                         [1, 0, -1]])
    
    sobel_y = np.array([[1, 2, 1],
                         [0, 0, 0],
                         [-1, -2, -1]])
    
    
    grad_x = convolve(image, sobel_x)
    grad_y = convolve(image, sobel_y)
    
    
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    return gradient_magnitude

def apply_threshold(image, threshold):
    binary_image = image > threshold
    return binary_image.astype(np.uint8)


image_path = "E:\\Gallery Redmi\\Instagram\\nanad.jpg"
image = imageio.imread(image_path)


if len(image.shape) == 3:
    image_gray = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
else:
    image_gray = image

edges = sobel_edge_detection(image_gray)
edges_normalized = (edges / edges.max() * 255).astype(np.uint8)

threshold_value = 100
segmented_image = apply_threshold(edges_normalized, threshold_value)


plt.figure(figsize=(12, 8))
plt.subplot(1, 3, 1)
plt.title('Gambar Asli')
plt.imshow(image_gray, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Deteksi Tepi (Sobel)')
plt.imshow(edges_normalized, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Hasil Segmentasi')
plt.imshow(segmented_image, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()