import nrrd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Đọc file NRRD
nrrd_path = r"D:\Project\ACL\my_tool\Thuan dot 1\HUYNH THANH PHONG (25919398)\Segmentation.seg.nrrd"
data, header = nrrd.read(nrrd_path)

# Kiểm tra dữ liệu 3D
if data.ndim != 3:
    raise ValueError("File NRRD không phải là ảnh 3D.")

num_slices = data.shape[2]

# Tạo figure và image
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)

# Tạo ảnh RGB từ nhãn: đỏ nếu là 1, đen nếu là 0
def get_rgb_slice(index):
    slice_data = data[:, :, index]
    rgb = np.zeros((slice_data.shape[0], slice_data.shape[1], 3), dtype=np.uint8)
    rgb[:, :, 0] = slice_data * 255  # kênh đỏ
    return rgb

rgb_slice = get_rgb_slice(0)
img = ax.imshow(rgb_slice)
ax.set_title(f"Slice 1 / {num_slices}")
ax.axis("off")

# Tạo thanh trượt
ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
slider = Slider(ax_slider, 'Slice', 0, num_slices - 1, valinit=0, valstep=1)

# Hàm cập nhật khi kéo thanh trượt
def update(val):
    slice_index = int(slider.val)
    rgb_slice = get_rgb_slice(slice_index)
    img.set_data(rgb_slice)
    ax.set_title(f"Slice {slice_index + 1} / {num_slices}")
    fig.canvas.draw_idle()

slider.on_changed(update)

plt.show()
