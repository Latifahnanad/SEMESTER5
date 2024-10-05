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
    
    # Pisahkan channel warna merah
    red_ch = image_source[:,:,0]
    
    # Membuat gambar baru hanya dengan channel merah
    image_red = np.zeros_like(image_source)
    image_red[:,:,0] = red_ch
    
    # Simpan gambar channel merah
    output_path = f"E:\\Nanad\\Tugas\\Python\\hasil\\{name}_red_channel.jpeg"
    image.imwrite(output_path, image_red)
    
    print(f"Gambar Channel Merah untuk {name} selesai disimpan di {output_path}")