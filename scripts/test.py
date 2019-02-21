
import dlib
from skimage import io

# 使用人脸检测器
detector = dlib.get_frontal_face_detector()

# 图片所在路径
img = io.imread('./me.jpg')

# 生成Dlib图像窗口
win = dlib.image_window()
win.set_image(img)

# 检测人脸
faces = detector(img, 1)
print(type(faces[0]), '\n')
print('人脸数:',len(faces))

# 坐标
for i, d in enumerate(faces):
    print('left:', d.left(), '\t',
          'right:', d.right(), '\t',
          'top:', d.top(), '\t',
          'bottom:', d.bottom(), '\t')

#绘制矩形框
win.add_overlay(faces)

# 保持图像
dlib.hit_enter_to_continue()

str = './a/b/c'
print(str.split('/')[1])


