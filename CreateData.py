import shutil
from glob import glob
import cv2
import numpy as np

def non_max_suppression(boxes, overlapThresh):
  '''
  boxes: List các bounding box
  overlapThresh: Ngưỡng overlapping giữa các hình ảnh
  '''
  # Nếu không có bounding boxes thì trả về empty list
  if len(boxes)==0:
    return []
  # Nếu bounding boxes nguyên thì chuyển sang float.
  if boxes.dtype.kind == "i":
    boxes = boxes.astype("float")

  # Khởi tạo list của index được lựa chọn
  pick = []

  # Lấy ra tọa độ của các bounding boxes
  x1 = boxes[:,0]
  y1 = boxes[:,1]
  x2 = boxes[:,2]
  y2 = boxes[:,3]

  # Tính toàn diện tích của các bounding boxes và sắp xếp chúng theo thứ tự từ bottom-right, chính là tọa độ theo y của bounding box
  area = (x2 - x1 + 1) * (y2 - y1 + 1)
  idxs = np.argsort(y2)
  # Khởi tạo một vòng while loop qua các index xuất hiện trong indexes
  while len(idxs) > 0:
    # Lấy ra index cuối cùng của list các indexes và thêm giá trị index vào danh sách các indexes được lựa chọn
    last = len(idxs) - 1
    i = idxs[last]
    pick.append(i)

    # Tìm cặp tọa độ lớn nhất (x, y) là điểm bắt đầu của bounding box và tọa độ nhỏ nhất (x, y) là điểm kết thúc của bounding box
    xx1 = np.maximum(x1[i], x1[idxs[:last]])
    yy1 = np.maximum(y1[i], y1[idxs[:last]])
    xx2 = np.minimum(x2[i], x2[idxs[:last]])
    yy2 = np.minimum(y2[i], y2[idxs[:last]])

    # Tính toán width và height của bounding box
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    # Tính toán tỷ lệ diện tích overlap
    overlap = (w * h) / area[idxs[:last]]

    # Xóa index cuối cùng và index của bounding box mà tỷ lệ diện tích overlap > overlapThreshold
    idxs = np.delete(idxs, np.concatenate(([last],
      np.where(overlap > overlapThresh)[0])))
  # Trả ra list các index được lựa chọn
  return boxes[pick].astype("int")

#Hàm output video phát hiện các đối tượng chuyển động ở trong video
#Hàm output video phát hiện các đối tượng chuyển động ở trong video
def detect_motivation(inputVideoPath, outputVideoPath):
  capture = cv2.VideoCapture(inputVideoPath)
  fps = capture.get(cv2.CAP_PROP_FPS)
  frame_width = int(capture.get(3))
  frame_height = int(capture.get(4))
  fourcc = cv2.VideoWriter_fourcc(*'XVID')
  # fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
  output = cv2.VideoWriter(outputVideoPath, fourcc, fps, (frame_width,frame_height))

  backSub = cv2.createBackgroundSubtractorMOG2()

  #đọc từng frame ảnh
  while True:
    _, frame = capture.read()
    if not _:
        break

    fgMask = backSub.apply(frame)

    contours,_ = cv2.findContours(fgMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    boundingBoxes = []
    for contour in contours:
      x,y,w,h = cv2.boundingRect(contour)
      x1, y1, x2, y2 = x, y, x+w, y+h
      area = cv2.contourArea(contour)
      if area > 300:
        boundingBoxes.append((x1, y1, x2, y2))
    boundingBoxes = [box for box in boundingBoxes if box[:2] != (0, 0)]
    boundingBoxes = np.array(boundingBoxes)
    pick = non_max_suppression(boundingBoxes, 0.1)

    for (startX, startY, endX, endY) in pick:
      cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
    output.write(frame)
  output.release()
  capture.release()

def _cropImage(x1, y1, x2, y2, img):
  if np.ndim(img) == 3:
      crop = img[y1:y2, x1:x2, :]
  else:
      crop = img[y1:y2, x1:x2]
  return crop

def cropImage(inputVideoPath, outputImageFolder, frame_steps=150):
  capture = cv2.VideoCapture(inputVideoPath)

  backSub = cv2.createBackgroundSubtractorMOG2()
  count = 0
  countImg = 0
  # đọc từng frame ảnh
  while True:
      _, frame = capture.read()
      count += 1
      if not _:
          break

      if ((count % frame_steps) == 5):
          print(count)
          fgMask = backSub.apply(frame)

          contours, _ = cv2.findContours(fgMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
          boundingBoxes = []
          for contour in contours:
              x, y, w, h = cv2.boundingRect(contour)
              x1, y1, x2, y2 = x, y, x + w, y + h
              area = cv2.contourArea(contour)
              if area > 300:
                  boundingBoxes.append((x1, y1, x2, y2))
          boundingBoxes = [box for box in boundingBoxes if box[:2] != (0, 0)]
          boundingBoxes = np.array(boundingBoxes)
          pick = non_max_suppression(boundingBoxes, 0.1)

          crop_images = [_cropImage(x1, y1, x2, y2, frame) for (x1, y1, x2, y2) in pick]

          for img in crop_images:
              cv2.imwrite(outputImageFolder + '/image' + str(countImg) + '.jpg', img)
              countImg += 1

  capture.release()

#cropImage('./test_video/cut1.mp4', './cut1_imgs', 30)
#detect_motivation('./test_video/cut2.mp4', './test_video/video-detect.avi')

def lowerVideo(inputpath, outputpath, size = (640, 368)):
    capture = cv2.VideoCapture(inputpath)
    fps = capture.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output = cv2.VideoWriter(outputpath, fourcc, fps, size)

    frame_count = 0
    while True:
        _, frame = capture.read()
        if not _:
            break

        frame_count += 1
        print(frame_count)
        frame = cv2.resize(frame, (640, 368))
        output.write(frame)

    output.release()
    capture.release()
#lowVideo('./test_video/3.mp4', './test_video/3.avi')

img_paths = []
car_paths = glob('./train/Car/*')
bike_paths= glob('./train/Bicycle/*')
moto_paths = glob('./train/Motorcycle/*')
print(len(car_paths))
print(len(bike_paths))
print(len(moto_paths))
for i in range(1000):
  print(i)
  shutil.copyfile(car_paths[i], './image-neg2/car' + str(i) + '.jpg')
  shutil.copyfile(bike_paths[i], './image-neg2/bike' + str(i) + '.jpg')
  shutil.copyfile(moto_paths[i], './image-neg2/moto' + str(i) + '.jpg')