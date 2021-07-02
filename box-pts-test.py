import open3d as o3d
import cv2
import numpy as np
from matplotlib import pyplot as plt
import copy
import random
from PIL import Image

post_ply = o3d.io.read_point_cloud("post_office.ply")
# o3d.visualization.draw_geometries([post_ply])

# convert Open3D.o3d.geometry.PointCloud to numpy array
ply_load = np.asarray(post_ply.points)
color_load = np.asarray(post_ply.colors)

file_name = 'color_temp2.png'
color_temp = cv2.imread(file_name)
original = cv2.imread(file_name, 0)
red = np.where((color_temp[:,:,2]==255) & (color_temp[:,:,1] == 0) & (color_temp[:,:,0] == 0), 255, 0)
red = np.where((color_temp[:,:,2]==185) & (color_temp[:,:,1] == 185) & (color_temp[:,:,0] ==185), 255, red)
# gray = np.where((color_temp[:,:,2]==185) & (color_temp[:,:,1] == 185) & (color_temp[:,:,0] == 0), 255, 185)

# plt.imshow(red)
# plt.show()
# plt.imshow(color_temp[:,:,2])
# plt.show()
# plt.imshow(color_temp[:,:,1])
# plt.show()
# plt.imshow(color_temp[:,:,0])
# plt.show()
MASK_SZ = 7
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MASK_SZ,MASK_SZ))
red1 = cv2.morphologyEx(red.astype(np.uint8), cv2.MORPH_DILATE, kernel, iterations=1)

red2 = cv2.morphologyEx(red.astype(np.uint8), cv2.MORPH_DILATE, kernel, iterations=2)
red2 = cv2.morphologyEx(red2, cv2.MORPH_ERODE, kernel, iterations=1)

red3 = cv2.morphologyEx(red.astype(np.uint8), cv2.MORPH_DILATE, kernel, iterations=3)
red3 = cv2.morphologyEx(red3, cv2.MORPH_ERODE, kernel, iterations=2)

fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharey=True)

# fig -> 창의 이름을 설정
fig.canvas.set_window_title('Sample Pictures')

# 그림마다 픽셀의 눈금 없애기
ax[0][0].axis('off')
ax[0][1].axis('off')
ax[1][0].axis('off')
ax[1][1].axis('off')

# 그림의 비율을 일정하게 유지하기
ax[0][0].imshow(red, aspect="auto")
ax[0][1].imshow(red1, aspect="auto")
ax[1][0].imshow(red2, aspect="auto")
ax[1][1].imshow(red3, aspect="auto")

# (서브플롯 간 간격 조절)그림 사이의 공백 조절
plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.05, hspace=0.05)


color_temp1 = copy.deepcopy(color_temp)
color_temp2 = copy.deepcopy(color_temp)
color_temp3 = copy.deepcopy(color_temp)

color_temp1[:,:,2] = np.where(red1==255, 255, color_temp1[:,:,2])
color_temp1[:,:,1] = np.where(red1==255, 0, color_temp1[:,:,1])
color_temp1[:,:,0] = np.where(red1==255, 0, color_temp1[:,:,0])

color_temp2[:,:,2] = np.where(red2==255, 255, color_temp2[:,:,2])
color_temp2[:,:,1] = np.where(red2==255, 0, color_temp2[:,:,1])
color_temp2[:,:,0] = np.where(red2==255, 0, color_temp2[:,:,0])

color_temp3[:,:,2] = np.where(red3==255, 255, color_temp3[:,:,2])
color_temp3[:,:,1] = np.where(red3==255, 0, color_temp3[:,:,1])
color_temp3[:,:,0] = np.where(red3==255, 0, color_temp3[:,:,0])

fig2, ax2 = plt.subplots(2, 2, figsize=(10, 10), sharey=True)

# fig -> 창의 이름을 설정
fig2.canvas.set_window_title('Sample Pictures')

# 그림마다 픽셀의 눈금 없애기
ax2[0][0].axis('off')
ax2[0][1].axis('off')
ax2[1][0].axis('off')
ax2[1][1].axis('off')

ax2[0][0].imshow(color_temp, aspect="auto")
ax2[0][1].imshow(color_temp1, aspect="auto")
ax2[1][0].imshow(color_temp2, aspect="auto")
ax2[1][1].imshow(color_temp3, aspect="auto")

original[np.where(red3==255)] = 0
_, threshold_img = cv2.threshold(original, 5, 255, cv2.THRESH_BINARY)

cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(threshold_img)

colors = np.zeros((color_temp.shape[0],color_temp.shape[1]), dtype=np.uint8) # ?
colors = cv2.cvtColor(colors, cv2.COLOR_GRAY2BGR)



for index in range(1, cnt): # 각각의 객체 정보에 들어가기 위해 반복문. 범위를 1부터 시작한 이유는 배경을 제외
    (x, y, w, h, area) = stats[index]
    if area < 80:
        continue
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    mask = np.where(labels == index, 255, 0)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=2)
    colors[np.where(mask==255)] = (b,g,r)
    # cv2.rectangle(dst, (x, y, w, h), (0, 255, 255))

colors = cv2.resize(colors, (int(colors.shape[0]*0.8), int(colors.shape[1]*0.8)))
# zeros = cv2.resize(zeros, (int(zeros.shape[0]*0.8), int(zeros.shape[1]*0.8)))

cv2.imshow('colors', colors)
# cv2.imshow('zeros', zeros)

cv2.imwrite('box-test4.png', colors)




plt.show()
cv2.waitKey(-1)

print('xyz_load')
