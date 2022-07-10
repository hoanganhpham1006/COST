import os
import cv2
from tqdm import tqdm
from multiprocessing import Process

ROOT = 'Charades_v1_480/Charades_v1_480/'
SAVE_DIR = 'frames/'
num_thread = 128

def extract_frame(videos):
    for video in tqdm(videos):
        os.mkdir(SAVE_DIR + video[:-4])
        vidcap = cv2.VideoCapture(ROOT + video)
        success,image = vidcap.read()
        count = 0
        while success:
            cv2.imwrite(f"{SAVE_DIR}{video[:-4]}/{video[:-4]}-{str(count).zfill(6)}.png", image)    
            success,image = vidcap.read()
            count += 1

videos = os.listdir(ROOT)
try:
    threads = {}
    for i in range(num_thread):
        threads[i] = Process(target=extract_frame,
            args=(videos[i*len(videos)//num_thread : (i+1)*len(videos)//num_thread], ))
    for i in range(num_thread):
        threads[i].start()
except AssertionError as error:
    print (error)