import imageio as img
import numpy as np
import matplotlib.pyplot as plt

def rotateImage(image, degree):
    radian_deg = np.radians(degree)
    cos_deg, sin_deg = np.cos(radian_deg), np.sin(radian_deg)
    height, width = image.shape[:2]

    # Hitung dimensi gambar output berdasarkan rotasi
    max_dim_x = int(abs(width * cos_deg) + abs(height * sin_deg))
    max_dim_y = int(abs(width * sin_deg) + abs(height * cos_deg))
    
    # Buat gambar output dengan ukuran yang baru
    outputImage = np.zeros((max_dim_y, max_dim_x, 3), dtype=image.dtype)

    for y in range(height):
        for x in range(width):
            # Tentukan posisi baru untuk setiap piksel menggunakan titik (0,0) sebagai pivot
            newX = int(cos_deg * x - sin_deg * y)
            newY = int(sin_deg * x + cos_deg * y)
            
            # Pastikan koordinat baru ada dalam batas dimensi gambar output
            if 0 <= newX < max_dim_x and 0 <= newY < max_dim_y:
                outputImage[newY, newX] = image[y, x]

    return outputImage

# Baca gambar
image = img.imread("E:\\Nanad\\Backup Redmi\\AdobeLightroom\\IMG_20230221_151301.jpg")

# Rotasi gambar
rotated_image = rotateImage(image, 45)

# Visualisasi gambar asli dan hasil rotasi
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image)

plt.subplot(1, 2, 2)
plt.title("Rotated Image (Pivot: Top-Left)")
plt.imshow(rotated_image)

plt.show()
