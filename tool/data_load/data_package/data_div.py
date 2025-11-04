import os
import shutil
import random
import sys
# Encoding stuff
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Define source and destination directories
source_dir = 'D:/ppg_project/Data/valid_data_resampled'
train_dir = 'D:/ppg_project/Data/data_train'
test_dir = 'D:/ppg_project/Data/data_test'

def clean_directory(directory_path):
    # create directory if not exists
    os.makedirs(directory_path, exist_ok=True)    
    # Delete old files
    print(f"Clean old files in  {directory_path} ---")
    deleted_count = 0
    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)
        # Only delete files, not subdirectories
        if os.path.isfile(item_path):
            os.remove(item_path)
            deleted_count += 1
            
clean_directory(train_dir)
clean_directory(test_dir)
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
    #anh hoa x
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