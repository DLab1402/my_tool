import nrrd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

# Đọc ảnh gốc và segmentation
image_path = r"D:\Project\ACL\my_tool\Thuan dot 1\KHUAT THI THU HANG (22940170)\8 t2_tse_sag.nrrd"
seg_path = r"D:\Project\ACL\my_tool\Thuan dot 1\KHUAT THI THU HANG (22940170)\Segmentation.seg.nrrd"

image, _ = nrrd.read(image_path)
seg, _ = nrrd.read(seg_path)

# Kiểm tra tính tương thích
if image.shape != seg.shape:
    raise ValueError("Ảnh gốc và segmentation không cùng kích thước.")
if image.ndim != 3 or seg.ndim != 3:
    raise ValueError("Cả ảnh gốc và segmentation phải là 3D.")

# Chuẩn hóa ảnh gốc về [0, 1]
image_norm = (image - np.min(image)) / (np.max(image) - np.min(image))

z_max = image.shape[2] - 1
slice_index = z_max // 2

# Hàm tạo overlay RGB slice
def create_overlay(slice_idx):
    gray = image_norm[:, :, slice_idx]
    label = seg[:, :, slice_idx]

    # Tạo ảnh RGB từ ảnh gray
    rgb = np.stack([gray]*3, axis=-1)

    # Tô đỏ vùng có segmentation
    red_mask = label > 0
    rgb[red_mask, 0] = 1.0   # R
    rgb[red_mask, 1] = 0.0   # G
    rgb[red_mask, 2] = 0.0   # B

    return rgb

# Tạo figure
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.15)

# Hiển thị ảnh overlay đầu tiên
img_display = ax.imshow(create_overlay(slice_index))
ax.set_title(f'Slice {slice_index}/{z_max}')
ax.axis('off')

# Thanh trượt
ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
slider = Slider(ax_slider, 'Slice', 0, z_max, valinit=slice_index, valstep=1)

# Hàm cập nhật khi thay đổi slice
def update(val):
    idx = int(slider.val)
    overlay = create_overlay(idx)
    img_display.set_data(overlay)
    ax.set_title(f'Slice {idx}/{z_max}')
    fig.canvas.draw_idle()

slider.on_changed(update)
plt.show()
