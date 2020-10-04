"""
    获取视频帧数
"""

# import cv2
#
# if __name__ == '__main__':
#
#     video = cv2.VideoCapture("./make_video/origin/import.mkv")
#
#     # Find OpenCV version
#     fps = video.get(cv2.CAP_PROP_FPS)
#     print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
#
#     video.release()


"""
    图片转视频
"""

# import os
# from PIL import Image
#
#
# if __name__ == "__main__":
#
#     imgPath = "./make_video/output_pic"  # 读取图片路径
#     videoPath = "./make_video/output_video/output.avi"  # 保存视频路径
#
#     images = os.listdir(imgPath)
#     fps = 25  # 每秒25帧数
#
#     fourcc = cv2.VideoWriter_fourcc(*"MJPG")
#
#     image = Image.open(os.path.join(imgPath, images[0]))
#     videoWriter = cv2.VideoWriter(videoPath, fourcc, fps, image.size)
#     for im_name in range(len(images)):
#         frame = cv2.imread(os.path.join(imgPath, str(im_name) + ".jpg"))  # 这里的路径只能是英文路径
#         print(im_name)
#         videoWriter.write(frame)
#     print("图片转视频结束！")
#     videoWriter.release()
#     cv2.destroyAllWindows()

'''
    视频转图片
'''

# import cv2
# import os
# import numpy as np
#
#
# def save_image(image, addr, num):
#     address = os.path.join(addr, 'frame' + str(num).zfill(5) + '.jpg')
#     image = cv2.resize(image, (512, 512))
#     cv2.imwrite(address, image)
#     print('save image to:', address)
#
# # 读取视频文件
# videoCapture = cv2.VideoCapture("./make_video/origin/import.mkv")
#
# i = 0
# while True:
#     # 读帧
#     success, frame = videoCapture.read()
#     if success is False:
#         break
#     i = i + 1
#     save_image(frame, './make_video/origin_pic/', i)
