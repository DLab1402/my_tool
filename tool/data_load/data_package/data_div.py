import os
import shutil
import random
import sys

# Thay đổi encoding stdout để hỗ trợ UTF-8 (nếu Python >= 3.7)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Đường dẫn thư mục nguồn
source_dir = 'D:/ppg_project/Data/valid_data_resampled'

# Tạo thư mục data_train và data_test nếu chưa tồn tại
train_dir = 'D:/ppg_project/Data/data_train'
test_dir = 'D:/ppg_project/Data/data_test'

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Lấy danh sách tất cả file trong thư mục nguồn (chỉ file .json để an toàn)
files = [f for f in os.listdir(source_dir) if f.endswith('.json')]

# Xáo trộn ngẫu nhiên danh sách file
random.shuffle(files)

# Tỷ lệ chia: 80% train, 20% test
split_index = int(0.8 * len(files))
train_files = files[:split_index]
test_files = files[split_index:]

# Di chuyển file vào data_train
for file in train_files:
    src = os.path.join(source_dir, file)
    dst = os.path.join(train_dir, file)
    shutil.move(src, dst)
    print(f"Di chuyển {file} vào data_train")

# Di chuyển file vào data_test
for file in test_files:
    src = os.path.join(source_dir, file)
    dst = os.path.join(test_dir, file)
    shutil.move(src, dst)
    print(f"Di chuyển {file} vào data_test")

# Thống kê thực tế sau khi di chuyển (đếm file .json trong thư mục)
train_count = len([f for f in os.listdir(train_dir) if f.endswith('.json')])
test_count = len([f for f in os.listdir(test_dir) if f.endswith('.json')])
total_moved = train_count + test_count

print(f"Thống kê:")
print(f"Số file train: {train_count}")
print(f"Số file test: {test_count}")
print(f"Tổng file đã di chuyển: {total_moved}")
print(f"Tỷ lệ train/test: {train_count}/{test_count} ({(train_count / total_moved * 100):.1f}% / {(test_count / total_moved * 100):.1f}%)")