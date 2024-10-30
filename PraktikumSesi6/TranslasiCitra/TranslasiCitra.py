import imageio.v3 as img
import numpy as np
import matplotlib.pyplot as plt

def Translasi(image, shiftX, shiftY):
    imgTranslasi = np.roll(image, shift=shiftY, axis=0) # Geser vertikal
    imgTranslasi = np.roll(imgTranslasi, shift=shiftX, axis=1) # Geser horizontal
    # Mengisi bagian yang kosong dengan warna hitam (0)
    if shiftY > 0:
        imgTranslasi[:shiftY, :] = 0 # Bagian atas jika geser ke bawah
    elif shiftY < 0:
        imgTranslasi[shiftY:, :] = 0 # Bagian bawah jika geser ke atas
    if shiftX > 0:
        imgTranslasi[:, :shiftX] = 0 # Bagian kiri jika geser ke kanan
    elif shiftX < 0:
        imgTranslasi[:, shiftX:] = 0 # Bagian kanan jika geser ke kiri
    return imgTranslasi

image = img.imread("E:\\Nanad\\Backup Redmi\\AdobeLightroom\\1676895868825.jpg")
imgResult = Translasi(image, shiftX=50, shiftY=-300)

plt.figure(figsize=(10,10))
plt.subplot(2,1,1)
plt.imshow(image)
plt.subplot(2,1,2)
plt.imshow(imgResult)
plt.show()