import numpy as np
import imageio as img
import matplotlib.pyplot as plt

# Path gambar
path = "E:\\Nanad\\Backup Redmi\\AdobeLightroom\\IMG_20230221_151301.jpg"
image = img.imread(path)

# Dapatkan dimensi gambar
height, width = image.shape[:2]

# Buat salinan kosong untuk mirroring vertikal dan horizontal
mirror_both = np.zeros_like(image)

# Proses mirroring vertikal dan horizontal bersamaan
for y in range(height):
    for x in range(width):
        mirror_both[y, x] = image[height - 1 - y, width - 1 - x]

# Visualisasi gambar asli dan hasil mirroring
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image)
plt.subplot(1, 2, 2)
plt.title("Vertical & Horizontal Mirroring")
plt.imshow(mirror_both)
plt.show()
