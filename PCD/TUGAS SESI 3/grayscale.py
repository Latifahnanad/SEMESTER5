import imageio.v3 as iio 
import numpy as np
import matplotlib.pyplot as plt

image = iio.imread("E:\\Nanad\\Backup Redmi\\camera\\IMG_20230210_133950.jpg")

gray_image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)

histogram, bin_edges = np.histogram(gray_image, bins=256, range=(0, 255))

total_pixels = np.sum(histogram)
max_intensity = np.argmax(histogram)
max_frequency = histogram[max_intensity]


fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].imshow(image)
axes[0].set_title('Gambar Asli (RGB)')
axes[0].axis('off')

axes[1].imshow(gray_image, cmap='gray')
axes[1].set_title('Gambar Grayscale')
axes[1].axis('off')

axes[2].plot(bin_edges[0:-1], histogram, color='blue')
axes[2].set_title('Histogram Gambar Grayscale')
axes[2].set_xlabel('Nilai Intensitas')
axes[2].set_ylabel('Frekuensi')
axes[2].grid(axis='y', linestyle='--', alpha=0.7)

axes[2].text(0, max_frequency + 500, f'Total Piksel: {total_pixels}', fontsize=12, color='blue')
axes[2].text(max_intensity, max_frequency, 
             f'Max Intensity: {max_intensity}\nFrequency: {max_frequency}', 
             ha='center', fontsize=12, color='red')


plt.tight_layout()
plt.show()