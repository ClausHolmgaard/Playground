import numpy as np
import cv2
import time

#get two frames
#scale frames
#draw pip
#show frame

WINDOW_SIZE = (1920, 1080)
PIP_SCALE = 0.2

VID1 = '../bunnygood.mkv'
VID2 = '../nature.mkv'

cap1 = None
cap2 = None

frame_list = []

def get_frames():
  # Get frame 1
  # Get frame 2
  # return [frame1, frame2]

  global cap1
  global cap2

  frame1 = None
  frame2 = None

  if cap1 is None:
    print("Opening stream for {}".format(VID1))
    cap1 = cv2.VideoCapture(VID1)

  if cap2 is None:
    print("Opening stream for {}".format(VID2))
    cap2 = cv2.VideoCapture(VID2)
  
  if cap1:
    ret1, frame1 = cap1.read()
  if cap2:
    ret2, frame2 = cap2.read()

  if not ret1:
    print("Error getting frame from stream 1: {}".format(VID1))
  if not ret2:
    print("Error getting frame from stream 2: {}".format(VID2))

  if frame1 is not None:
    #print(frame1.shape)
    pass
  if frame2 is not None:
    #print(frame2.shape)
    pass
    
  return frame1, frame2 
    
def process(f1, f2):
  # scale frame 1
  # scale frame 2
  # draw pip image
  # return pip

  frame1 = cv2.resize(f1, dsize=(WINDOW_SIZE[0], WINDOW_SIZE[1]), fx=0, fy=0)

  pip_x = int(WINDOW_SIZE[0] * PIP_SCALE)
  pip_y = int(WINDOW_SIZE[1] * PIP_SCALE)
  frame2 = cv2.resize(f2, dsize=(pip_x, pip_y), fx=0, fy=0)
  
  return frame1, frame2

def show(frame):
  # show image

  cv2.imshow("Main", frame)
  cv2.waitKey(1)

def done():
  if cap1:
    cap1.release()
  if cap2:
    cap2.release()

def run():
  frames = 100
  time_start = time.time()
  for i in range(frames):
    f1, f2 = get_frames()
    #show(f1)

  time_run = time.time() - time_start
  frame_time_ms = (time_run / frames) * 1000

  print("Run time: {}".format(time_run))
  print("Time / frame: {} ms".format(frame_time_ms)) 

  done()

def test():
  # Get alot of frames

  global frame_list

  frames = 1000
  print_every = 100
  for i in range(frames):
    if i%print_every == 0:
      print("Grabbing image {} of {}".format(i, frames))
    f1, f2 = get_frames()
    frame_list.append([f1, f2])
  print("Done grabbing")

  frame_list = np.array(frame_list)

  time_start = time.time()
  counter = 0
  for frames in frame_list:
    if counter%print_every == 0:
      print("Processing ImportWarninge {} of {}".format(counter, len(frame_list)))
      
    frame1, frame2 = process(frames[0], frames[1])
    counter += 1
  print("Done processing.\n")

  time_run = time.time() - time_start
  frame_time_ms = (time_run / counter) * 1000
 
  print("Processing time: {}".format(time_run))
  print("Processing time / frame: {} ms".format(frame_time_ms)) 
    

if __name__ == '__main__':
  #run()
  test()
  
  
