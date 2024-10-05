import numpy as np
import imageio.v3 as image

paths = {
    "kenikir": "E:\\Nanad\\Tugas\\Python\\daun_kenikir.jpg",
    "singkong": "E:\\Nanad\\Tugas\\Python\\daun_singkong.jpeg",
    "pepaya": "E:\\Nanad\\Tugas\\Python\\daun_pepaya.jpeg"
}

for name, path in paths.items():
    # Membaca gambar
    image_source = image.imread(path)
    
    # Konversi ke grayscale menggunakan rumus luminance
    grayscale_image = np.dot(image_source[...,:3], [0.299, 0.587, 0.114]).astype(np.uint8)
    
    # Menerapkan thresholding (biner) dengan threshold 128
    threshold_image = np.where(grayscale_image > 128, 255, 0).astype(np.uint8)
    
    # Simpan gambar threshold (biner)
    output_path = f"E:\\Nanad\\Tugas\\Python\\hasil\\{name}_threshold.jpeg"
    image.imwrite(output_path, threshold_image)
    
    print(f"Gambar Threshold (Biner) untuk {name} selesai disimpan di {output_path}")