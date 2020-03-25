import numpy as np
import cv2
from time import sleep

if __name__ == "__main__":
    #import ipdb;ipdb.set_trace()
    frame = 0
    render_delay = 0.01
    while True:
        file = f'frames/tetris-nn={str(frame)}.npy'
        try:
            img = np.load(file)
        except:
            break
        cv2.imshow('image', np.array(img))
        #filename = f'pics/tetris-nn={str(frame)}.jpg'
        #cv2.imwrite(filename, np.array(img)) 
        w = cv2.waitKey(1)
        sleep(render_delay)
        frame += 1
