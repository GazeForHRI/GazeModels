import sys
sys.path.insert(0,'yolo_head')
from yolo_head.detect import det_head
## 构建字典，遍历每张图片
import cv2
import os
cap = cv2.VideoCapture('video_1.mp4')

def delete_files_in_folder(folder_path):
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"文件夹 '{folder_path}' 不存在")
        return

    # 获取文件夹中的所有文件和子文件夹
    files = os.listdir(folder_path)

    for file in files:
        file_path = os.path.join(folder_path, file)

        if os.path.isfile(file_path):
            # 如果是文件，删除它
            os.remove(file_path)
            print(f"del_file: {file_path}")
        elif os.path.isdir(file_path):
            # 如果是文件夹，递归删除它
            delete_files_in_folder(file_path)
    
    # 删除空文件夹


delete_files_in_folder("result/labels/")
delete_files_in_folder("frames/")
delete_files_in_folder("new_frames/")
frame_id = 0
while   True:
    ret, frame = cap.read()
    if ret:
        cv2.imwrite('frames/%d.jpg' % frame_id, frame)
        frame_id += 1
    else:
        break
    
imgset = 'frames/*.jpg'
det_head(imgset)



