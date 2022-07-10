import os
import torch
torch.set_num_threads(1)
import torchvision
import numpy as np
import os
import cv2
from tqdm import tqdm
from multiprocessing import Process
import pickle

FRAME_DIR = 'frames/'
SAVE_DIR = 'res101_feat/'
num_thread = 8
batch_infer = 64
num_gpu = 8
mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
std = np.array([0.229, 0.224, 0.224]).reshape(1, 3, 1, 1)
processed_videos = [vid[:-4] for vid in os.listdir(SAVE_DIR)]
videos = [vid for vid in os.listdir(FRAME_DIR) if vid not in processed_videos]


def build_resnet(gpu_id):
    cnn = getattr(torchvision.models, 'resnet101')(pretrained=True)
    model = torch.nn.Sequential(*list(cnn.children())[:-1])
    model = model.to(torch.device(f"cuda:{gpu_id}"))
    # model = model.cuda()
    model.eval()
    print(f"Spawned @ GPU {gpu_id}")
    return model

def generate_pkl(model, videos, gpu_id):
    device = torch.device(f"cuda:{gpu_id}")
    for video in tqdm(videos):
        if os.path.isfile(f"{SAVE_DIR}{video}.pkl"):
            continue
        frames = sorted(os.listdir(os.path.join(FRAME_DIR, video)))
        vid_feats = []
        for i in range(0, len(frames), batch_infer):
            frame_batch = [cv2.imread(os.path.join(FRAME_DIR, video, frames[j])) 
                                for j in range(i, min(i+batch_infer, len(frames)))]
            frame_batch = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in frame_batch]
            frame_batch = [cv2.resize(img, (224, 224)) for img in frame_batch]
            frame_batch = np.array(frame_batch)
            frame_batch = frame_batch.reshape((-1, 3, 224, 224)).astype(np.float32)
            frame_batch = (frame_batch / 255.0 - mean) / std
            frame_batch = torch.FloatTensor(frame_batch).to(device)
            feats = model(frame_batch)
            feats = feats.data.cpu().clone().numpy()
            vid_feats.extend(feats[:, :, 0, 0])
        if len(vid_feats) > 0:
            pickle.dump(vid_feats, open(f"{SAVE_DIR}{video}.pkl", "wb"))

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    models = {}
    for i in range(num_gpu):
        models[i] = build_resnet(i)
    try:
        threads = {}
        for i in range(num_thread):
            threads[i] = Process(target=generate_pkl,
                args=(models[i%num_gpu], videos[i*len(videos)//num_thread : (i+1)*len(videos)//num_thread], i%num_gpu))
        for i in range(num_thread):
            threads[i].start()
    except AssertionError as error:
        print (error)