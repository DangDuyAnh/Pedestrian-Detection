#Import thư viện
import json
from time import sleep
import time
import numpy as np
from skimage.feature import hog
import cv2
import pickle

# tuplify
def tup(point):
    return (point[0], point[1]);

# returns true if the two boxes overlap
def overlap(source, target):
  # unpack points
  tl1, br1 = source;
  tl2, br2 = target;

  # checks
  if (tl1[0] >= br2[0] or tl2[0] >= br1[0]):
    return False;
  if (tl1[1] >= br2[1] or tl2[1] >= br1[1]):
    return False;
  return True;

# returns all overlapping boxes
def getAllOverlaps(boxes, bounds, index):
  overlaps = [];
  for a in range(len(boxes)):
    if a != index:
      if overlap(bounds, boxes[a]):
        overlaps.append(a);
  return overlaps;

def mergeNearBy(contours, hierarchy, mergeMargin = -1):
  boxes = [];  # each element is [[top-left], [bottom-right]];
  hierarchy = hierarchy[0]
  for component in zip(contours, hierarchy):
      currentContour = component[0]
      currentHierarchy = component[1]
      x,y,w,h = cv2.boundingRect(currentContour)
      if currentHierarchy[3] < 0:
          boxes.append([[x,y], [x+w, y+h]]);

  # filter out excessively large boxes
  filtered = [];
  max_area = 30000;
  for box in boxes:
      w = box[1][0] - box[0][0];
      h = box[1][1] - box[0][1];
      if w*h < max_area:
          filtered.append(box);
  boxes = filtered;

  # this is gonna take a long time
  finished = False;
  highlight = [[0, 0], [1, 1]];
  points = [[[0, 0]]];
  while not finished:
    # set end con
    finished = True;

    # loop through boxes
    index = len(boxes) - 1;
    while index >= 0:
      # grab current box
      curr = boxes[index];

      # add margin
      tl = curr[0][:];
      br = curr[1][:];
      tl[0] -= mergeMargin;
      tl[1] -= mergeMargin;
      br[0] += mergeMargin;
      br[1] += mergeMargin;

      # get matching boxes
      overlaps = getAllOverlaps(boxes, [tl, br], index);

      # check if empty
      if len(overlaps) > 0:
        # combine boxes
        # convert to a contour
        con = [];
        overlaps.append(index);
        for ind in overlaps:
          tl, br = boxes[ind];
          con.append([tl]);
          con.append([br]);
        con = np.array(con);

        # get bounding rect
        x, y, w, h = cv2.boundingRect(con);

        # stop growing
        w -= 1;
        h -= 1;
        merged = [[x, y], [x + w, y + h]];

        # remove boxes from list
        overlaps.sort(reverse=True);
        for ind in overlaps:
          del boxes[ind];
        boxes.append(merged);

        # set flag
        finished = False;
        break;

      # increment
      index -= 1

  return boxes

def _cropImage(x1, y1, x2, y2, img):
  if np.ndim(img) == 3:
      crop = img[y1:y2, x1:x2, :]
  else:
      crop = img[y1:y2, x1:x2]
  return crop

def detect(svmModelPath, inputVideoPath, outputVideoPath=None, backgroundRatio = 0.9, nMixtures = 5, varThreshold = 16 ,varInit = 15, medianBlur = True, medianSize = 3, erode = False, dilate = False, kernelParam = [3, 3]):
  capture = cv2.VideoCapture(inputVideoPath)
  output = 0
  if outputVideoPath != None:
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_width = int(capture.get(3))
    frame_height = int(capture.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output = cv2.VideoWriter(outputVideoPath+'/detect.avi', fourcc, fps, (frame_width, frame_height))
    output_gray = cv2.VideoWriter(outputVideoPath+'/fgMask.avi', fourcc, fps, (frame_width, frame_height), 0)
  backSub = cv2.createBackgroundSubtractorMOG2(varThreshold = varThreshold)
  backSub.setBackgroundRatio(backgroundRatio)
  backSub.setNMixtures(nMixtures)
  backSub.setVarInit(varInit)

  kernel = np.ones((kernelParam[0], kernelParam[1]), np.uint8)
  frame_count = 0
  error = []
  loaded_model = pickle.load(open(svmModelPath, 'rb'))

  while True:
    _, frame = capture.read()
    if not _:
      break
    frameCopy = frame.copy()
    frame_count += 1
    print('Frame: ' + str(frame_count))
    fgMask = backSub.apply(frame)
    if (medianBlur):
      fgMask = cv2.medianBlur(fgMask, medianSize)
    if (erode):
      fgMask = cv2.erode(fgMask, kernel, iterations=1)
    if (dilate):
      fgMask = cv2.dilate(fgMask, kernel, iterations=1)
    contours, hierarchy = cv2.findContours(fgMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    try:
      boxes = mergeNearBy(contours, hierarchy)
    except:
      error.append(frame_count)
    classify_arr = []
    takenBoxes = []
    for box in boxes:
      if ((box[1][0]-box[0][0]) * (box[1][1] - box[0][1]) > 900):
        img = _cropImage(box[0][0], box[0][1], box[1][0], box[1][1], frame)
        img = np.asarray(img)
        img = cv2.cvtColor(cv2.resize(img,(64,128)),cv2.COLOR_RGB2GRAY)
        img = hog(img)
        classify_arr.append(img)
        takenBoxes.append(box)

    result = []
    classify_arr = np.asarray(classify_arr)
    if (len(classify_arr)):
      result = loaded_model.predict(classify_arr)

    for i in range(len(result)):
      if (result[i]):
        cv2.rectangle(frame, tup(takenBoxes[i][0]), tup(takenBoxes[i][1]), (255, 0, 0), 1)
        cv2.putText(frame, 'Human', (takenBoxes[i][0][0], takenBoxes[i][0][1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)
      else:
        cv2.rectangle(frame, tup(takenBoxes[i][0]), tup(takenBoxes[i][1]), (0, 0, 255), 1)
        cv2.putText(frame, 'Motorcycle/Car', (takenBoxes[i][0][0], takenBoxes[i][0][1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)

    foregroundPart = cv2.bitwise_and(frame, frame, mask=fgMask)
    stacked_frame = np.hstack((frameCopy, foregroundPart, frame))

    if (outputVideoPath == None):
      cv2.imshow('Original Frame, Extracted Foreground and Detected Human', cv2.resize(stacked_frame, None, fx=1, fy=2))
      keyboard = cv2.waitKey(30)
      if keyboard == 'q' or keyboard == 27:
        break
    else:
      output.write(frame)
      output_gray.write(fgMask)
  capture.release()


def detect_and_send_to_ws(ws,
    svmModelPath, inputVideoPath, outputVideoPath=None, 
    backgroundRatio = [0.9], nMixtures = [5], varThreshold = [16] ,varInit = [15], medianBlur = [True], 
    medianSize = [3], erode = [False], dilate = [False], kernel = [np.ones((3, 3), np.uint8)]):#kernelParam = [3, 3]):

  capture = cv2.VideoCapture(inputVideoPath)

  width = 0.0
  height = 0.0
  fps = 0
  if capture.isOpened(): 
    # get vcap property 
    width  = capture.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
    height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
    fps = capture.get(cv2.CAP_PROP_FPS)
    ws.send(json.dumps({ 
        'width'   : width,
        'height'  : height,
        'fps'     : fps
      })
    )
  else:
    raise RuntimeError()

  output = 0
  if outputVideoPath != None:
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_width = int(capture.get(3))
    frame_height = int(capture.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output = cv2.VideoWriter(outputVideoPath+'/detect.avi', fourcc, fps, (frame_width, frame_height))
    output_gray = cv2.VideoWriter(outputVideoPath+'/fgMask.avi', fourcc, fps, (frame_width, frame_height), 0)
  backSub = cv2.createBackgroundSubtractorMOG2(varThreshold = varThreshold[0])
  backSub.setBackgroundRatio(backgroundRatio[0])
  backSub.setNMixtures(nMixtures[0])
  backSub.setVarInit(varInit[0])

  #kernel = np.ones((kernelParam[0], kernelParam[1]), np.uint8)
  frame_count = 0
  error = []
  loaded_model = pickle.load(open(svmModelPath, 'rb'))


  if fps == 0: return

  frameTime = 1 / fps

  curTime = time.time()

  l_varThreshold = varThreshold[0]
  l_backgroundRatio = backgroundRatio[0]
  l_nMixtures = nMixtures[0]
  l_varInit = varInit[0]

  while True:
    #start = time.time()

    if l_varThreshold != varThreshold[0]:
      l_varThreshold = varThreshold[0]
      backSub.setVarThreshold(l_varThreshold)

    if l_backgroundRatio != backgroundRatio[0]:
      l_backgroundRatio = backgroundRatio[0]
      backSub.setBackgroundRatio(l_backgroundRatio)

    if l_nMixtures != nMixtures[0]:
      backSub.setNMixtures(int(l_nMixtures))
      l_nMixtures = nMixtures[0]

    if l_varInit != varInit[0]:
      backSub.setVarInit(l_varInit)
      l_varInit = varInit[0]

    _, frame = capture.read()
    if not _:
      break
    frameCopy = frame.copy()
    frame_count += 1
    #print('Frame: ' + str(frame_count))
    fgMask = backSub.apply(frame)
    if (medianBlur[0]):
      fgMask = cv2.medianBlur(fgMask, medianSize[0])
    if (erode[0]):
      fgMask = cv2.erode(fgMask, kernel[0], iterations=1)
    if (dilate[0]):
      fgMask = cv2.dilate(fgMask, kernel[0], iterations=1)
    contours, hierarchy = cv2.findContours(fgMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    try:
      boxes = mergeNearBy(contours, hierarchy)
    except:
      error.append(frame_count)
    classify_arr = []
    takenBoxes = []
    for box in boxes:
      if ((box[1][0]-box[0][0]) * (box[1][1] - box[0][1]) > 900):
        img = _cropImage(box[0][0], box[0][1], box[1][0], box[1][1], frame)
        img = np.asarray(img)
        img = cv2.cvtColor(cv2.resize(img,(64,128)),cv2.COLOR_RGB2GRAY)
        img = hog(img)
        classify_arr.append(img)
        takenBoxes.append(box)

    result = []
    classify_arr = np.asarray(classify_arr)
    if (len(classify_arr)):
      result = loaded_model.predict(classify_arr)

    for i in range(len(result)):
      if (result[i]):
        cv2.rectangle(frame, tup(takenBoxes[i][0]), tup(takenBoxes[i][1]), (255, 0, 0), 1)
        cv2.putText(frame, 'Human', (takenBoxes[i][0][0], takenBoxes[i][0][1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)
      else:
        cv2.rectangle(frame, tup(takenBoxes[i][0]), tup(takenBoxes[i][1]), (0, 0, 255), 1)
        cv2.putText(frame, 'Motorcycle/Car', (takenBoxes[i][0][0], takenBoxes[i][0][1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)

    foregroundPart = cv2.bitwise_and(frame, frame, mask=fgMask)
    stacked_frame = np.hstack((frameCopy, foregroundPart, frame))

    img_encode = cv2.imencode('.jpg', stacked_frame)[1]
    data_encode = np.array(img_encode)
    byte_encode = data_encode.tobytes()
    ws.send(byte_encode)

    end = time.time()
    delta = end - curTime
    curTime = end
    sleepTime = max(frameTime - delta, 0)
    if sleepTime != 0: sleep(sleepTime)

  capture.release()

#detect('./SVM_modelv3.h5','./videos/3-trim.mp4', './videos')
#detect('./SVM_modelv3.h5','./videos/3-trim.mp4')