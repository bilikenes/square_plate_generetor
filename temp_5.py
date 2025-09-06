import os
import random
import shutil


src_folder = r"C:\Users\PC\Desktop\plates\square_plates"

# Hedef klasör (seçilen 3000 resmin taşınacağı yer)
dst_folder = r"C:\Users\PC\Desktop\plates\square_plates_for_test"

# Hedef klasörü oluştur (yoksa)
os.makedirs(dst_folder, exist_ok=True)

# Kaynak klasördeki tüm dosyaları listele
all_files = os.listdir(src_folder)

# Sadece resim dosyalarını filtrele (jpg, png vs.)
image_files = [f for f in all_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# 3000 tane rastgele seç
selected_files = random.sample(image_files, 3000)

# Seçilen dosyaları taşı
for file_name in selected_files:
    src_path = os.path.join(src_folder, file_name)
    dst_path = os.path.join(dst_folder, file_name)
    shutil.move(src_path, dst_path)

print("3000 resim başarıyla taşındı!")
