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
    
    # Simpan gambar grayscale
    output_path = f"E:\\Nanad\\Tugas\\Python\\hasil\\{name}_grayscale.jpeg"
    image.imwrite(output_path, grayscale_image)
    
    print(f"Gambar Grayscale untuk {name} selesai disimpan di {output_path}")