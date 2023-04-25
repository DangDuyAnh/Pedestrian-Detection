from io import FileIO
import json
from time import sleep

import numpy as np

import threading
from threading import Thread, Lock

from processing.main import detect_and_send_to_ws

from flask import Flask, render_template
from flask_sock import Sock

app = Flask(__name__)
sock = Sock(app)

fileName: str = ''
blobCount: int = 0
fout: FileIO = None

backgroundRatio = [0.9] 
nMixtures = [5]
varThreshold = [16]
varInit = [15]
medianBlur = [True]
medianSize = [3]
erode = [False]
dilate = [False]
kernel = [np.ones((3, 3), np.uint8)]

params = {
  'backgroundRatio'   : backgroundRatio,
  'nMixtures'         : nMixtures,
  'varThreshold'      : varThreshold,
  'varInit'           : varInit,
  'medianBlur'        : medianBlur,
  'medianSize'        : medianSize,
  'erode'             : erode,
  'dilate'            : dilate,
  'kernel'            : kernel,
}

def ParseNumber(s):
  if type(s) is str:
    if '.' in s:
      try:
        v = float(s)
        return True, v
      except ValueError:
        return False, s
    else:
      try:
        v = int(s)
        return True, v
      except ValueError:
        return False, s

  return False, s
    

def ChangeVariable(data):
  global params
  if 'variable' in data:
    # { 'variable': true, 'name': ..., 'value': ... }
    variable = json.loads(data)

    name = variable['name']
    value = variable['value']

    isNumber, value = ParseNumber(value)

    variable['value'] = value

    print(type(value))

    if name != 'kernel' and (name in params):
      params[name][0] = value
    elif name == 'kernel':
      params['kernel'][0] = np.ones((int(value[0]), int(value[1])), np.uint8)

    print(variable)

def RecvThread(sock, running = [True]):
  global params

  while running[0] == True:
    data = sock.receive()
    if type(data) == str:
      ChangeVariable(data)

  
@sock.route('/')
def ws(sock):
  global blobCount, fileName, fout, backgroundRatio
  global nMixtures, varThreshold, varInit, medianBlur, medianSize, erode, dilate, kernel

  running = [True]
  thread: Thread = None

  try:
    while True:
      data = sock.receive()
      #sock.send(data)
      if type(data) == str:
        ChangeVariable(data)
        
        if data == 'begin-stream-file':
          fileName = f'sample.mp4'
          fout = open(fileName, 'wb')
          blobCount = 0
          
        elif data == 'end-stream-file':
          fout.close()
          fout = None

          running[0] = True
          thread = Thread(target = RecvThread, args = (sock, running, ))
          thread.start()
          
          detect_and_send_to_ws(sock, './linearSVM_model_final.h5', fileName, None, 
            backgroundRatio, nMixtures, varThreshold, varInit, medianBlur, medianSize, erode, dilate, kernel)

          running[0] = False
          thread.join()
          thread = None

          sock.send('end')

      elif type(data) == bytes:
        fout.write(data)
        blobCount += 1

      else:
        raise RuntimeError()

  finally:
    if thread is not None and running[0] == True:
      running[0] = False
      thread.join()
      thread = None

indexFile = None
with open('../../../client/index.html', 'r') as file:
    indexFile = file.read()

@app.route('/', methods=['GET'])
def index():
  return indexFile

app.run()