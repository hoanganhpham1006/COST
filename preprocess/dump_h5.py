import argparse, os
import h5py
import pickle
import numpy as np
from tqdm import tqdm
import json
from transformers import BertTokenizer, BertModel


'''
- Object feature:
+ name: 'appearance_features'; shape: [dataset_size, num_clip, num_obj_tubelet, num_frames, dim] (appearance feature - ROI pooling feature in the paper)   
+ name: 'spatial_features'; shape: [dataset_size, num_clip, num_obj_tubelet, num_frames, dim] (positional features in the paper)
+ name: 'ids'; shape: [dataset_size] (video_id)

- ResNet feature:
+ name: 'resnet_features'; shape: [dataset_size, num_clip, num_frames, dim] (ResNet feature in the paper)
+ name: 'ids'; shape: [dataset_size] (video_id)
'''


SAVE_DIR = "h5_id/"
FRAME_DIR = "frames/"
TUBELET_DIR = "tubelet_id/"
RES_PKL_DIR = "res101_feat/"
I3D_FLOW_PKL_DIR = "i3d_flow/"

MINIMUM_APP = 2
NUMBER_OBJ = 30
APP_FEAT_DIM = 1024
RES_FEATURE_DIM = 2048
MAX_LENGTH = 90
SPATIAL_FEATURE_DIM = 7 # xmin/W, ymin/H, xmax/W, ymax/H, w/W, h/H, wh/WH 
NUMBER_CLIPS = 8
NUMBER_FRAMES = 22
SAMPLE_RATE = 5
DEBUG = False

def generate_h5(video_ids, number_clip=NUMBER_CLIPS, number_frames_per_clip=NUMBER_FRAMES, out_i3d="out_i3d_8c22f_full.h5", out_res="out_res_8c22f_full.h5", out_obj="out_obj_8c22f_full.h5"):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # model = BertModel.from_pretrained("bert-base-uncased")

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    dataset_size = len(video_ids)

    f_res = h5py.File(SAVE_DIR + out_res, 'w')
    f_i3d = h5py.File(SAVE_DIR + out_i3d, 'w')
    f_obj = h5py.File(SAVE_DIR + out_obj, 'w')

    i3d_feat_dset = None
    # i3d_video_ids_dset = None
    res_feat_dset = None
    # res_video_ids_dset = None
    obj_app_feat_dset = None
    obj_spatial_feat_dset = None
    obj_name_enc_dset = None
    obj_name_ids_dset = None
    # obj_video_ids_dset = None
    i0 = 0

    EMPTY_BBOX = [0, 0, 0, 0, 0, 0, 0]
    EMPTY_FEAT = np.zeros((APP_FEAT_DIM, ))
    map_id_name = {}
    for v_id, video_id in enumerate(tqdm(video_ids)):
        pointer = 0
        map_id_name[v_id] = video_id
        vid_res_feat = np.array(pickle.load(open(RES_PKL_DIR + video_id + ".pkl", "rb")))
        vid_i3d_feat = np.load(I3D_FLOW_PKL_DIR + video_id + ".npy")
        len_obj_feat = len(os.listdir(FRAME_DIR + video_id))
        len_i3d_feat = len(vid_i3d_feat)

        ratio_i3d_res = len_i3d_feat / len_obj_feat

        vid_objs_info = open(TUBELET_DIR + video_id + ".txt", "r").readlines()
        start_frame = int(vid_objs_info[0][:-1].split(",")[0])

        res_feat = np.zeros((number_clip, number_frames_per_clip, RES_FEATURE_DIM))
        i3d_feat = np.zeros((number_clip, number_frames_per_clip, RES_FEATURE_DIM))
        obj_app_feat = np.zeros((number_clip, NUMBER_OBJ, number_frames_per_clip, APP_FEAT_DIM))
        obj_spatial_feat = np.zeros((number_clip, NUMBER_OBJ, number_frames_per_clip, SPATIAL_FEATURE_DIM))
        obj_name_enc = np.zeros((MAX_LENGTH, ))
        obj_name_ids = np.zeros((NUMBER_OBJ, ))

        clips_keyframes = np.linspace(start_frame, len_obj_feat, num=number_clip + 2, dtype=np.int32)[1:-1]
        # i3d_keyframes = np.linspace(0, len_i3d_feat, num=number_clip + 2, dtype=np.int32)[1:-1]
        # overlap = False
        # if i3d_startframes[1] - i3d_startframes[0] < number_frames_per_clip:
        #     overlap = True

        objs_life = {}
        for i in range(number_clip):
            clip_frames = []
            i3d_frames = []

            for j in range(-number_frames_per_clip//2, number_frames_per_clip//2):
                clip_frames.append(clips_keyframes[i] + j*SAMPLE_RATE)
                i3d_frames.append(int((clips_keyframes[i] + j*SAMPLE_RATE)*ratio_i3d_res))
            
            clip_frames = np.clip(clip_frames, 0, len(vid_res_feat) - 1)
            i3d_frames = np.clip(i3d_frames, 0, len_i3d_feat - 1)

            try:
                res_feat[i] += vid_res_feat[clip_frames] # num_frames, dim
                i3d_feat[i] += vid_i3d_feat[i3d_frames] # num_frames, dim
            except:
                import pdb;pdb.set_trace()

            while pointer < len(vid_objs_info) and int(vid_objs_info[pointer][:-1].split(",")[0]) <= clip_frames[-1]:
                line = vid_objs_info[pointer][:-1].split(",")
                frame_id, obj_id = line[:2]
                xmin, ymin, w, h, W, H = [float(_index) for _index in line[2:8]] #(xmin, ymin, w, h, W, H)
                bbox = [xmin/W, ymin/H, (xmin + w)/W, (ymin + h)/H, w/W, h/H, w*h/W/H]
                obj_feat = [float(val) for val in line[12:]]

                if int(frame_id) in clip_frames:
                    if obj_id not in objs_life:
                        objs_life[obj_id] = {}
                    objs_life[obj_id][frame_id] = {'bbox': bbox, 'obj_feat': obj_feat}

                pointer += 1

        objs_life = dict(sorted(objs_life.items(), key=lambda item: len(item[1]), reverse=True))

        # BERT Extractor
        text = ""
        obj_index = 0
        word_index = 0
        word_indices = []
        for obj in objs_life:
            if obj_index >= NUMBER_OBJ:
                break
            if len(objs_life[obj]) < MINIMUM_APP:
                continue
            text += obj.split('_')[0] + " "
            word_index += len(tokenizer.encode(obj.split('_')[0])) - 2
            word_indices.append(word_index)
            obj_index += 1
        encoded_input = tokenizer(text, return_tensors='pt')
        # output_bert = model(**encoded_input)
        # output_bert = output_bert.last_hidden_state[0][word_indices].detach().numpy()
        # obj_name_feat[:len(output_bert), :] += output_bert
        
        encoded_input = encoded_input.input_ids[0].detach().numpy()
        obj_name_enc[:len(encoded_input)] += encoded_input
        obj_name_ids[:len(word_indices)] += np.array(word_indices)

        for i in range(number_clip):
            clip_frames = []
            
            for j in range(-number_frames_per_clip//2, number_frames_per_clip//2):
                clip_frames.append(clips_keyframes[i] + j*SAMPLE_RATE)
            clip_frames = np.clip(clip_frames, 0, len(vid_res_feat) - 1)

            obj_app_feat_clip = np.zeros((NUMBER_OBJ, number_frames_per_clip, APP_FEAT_DIM))
            obj_spatial_feat_clip = np.zeros((NUMBER_OBJ, number_frames_per_clip, SPATIAL_FEATURE_DIM))
            
            obj_index = 0
            for obj in objs_life:
                if obj_index >= NUMBER_OBJ:
                    break
                if len(objs_life[obj]) < MINIMUM_APP:
                    continue
            
                for frame_index, clip_frame in enumerate(clip_frames):
                    clip_frame_s = str(clip_frame)
                    if clip_frame_s not in objs_life[obj]:
                        objs_life[obj][clip_frame_s] = {'bbox': EMPTY_BBOX, 'obj_feat': EMPTY_FEAT}
                    obj_app_feat_clip[obj_index][frame_index] += objs_life[obj][clip_frame_s]['obj_feat']
                    obj_spatial_feat_clip[obj_index][frame_index] += objs_life[obj][clip_frame_s]['bbox']
                obj_index += 1
            obj_app_feat[i][:len(obj_app_feat_clip), :]  += obj_app_feat_clip
            obj_spatial_feat[i][:len(obj_app_feat_clip), :]  += obj_spatial_feat_clip

        i3d_feat = np.asarray(i3d_feat) # numclips, num_frames, dim
        if i3d_feat_dset is None:
                C, F, D = i3d_feat.shape
                i3d_feat_dset = f_i3d.create_dataset('resnet_features', (dataset_size, C, F, D),
                                                dtype=np.float32)
        # if i3d_video_ids_dset is None:
        #     i3d_video_ids_dset = f_i3d.create_dataset('ids', shape=(dataset_size,), dtype=str)

        res_feat = np.asarray(res_feat) # numclips, num_frames, dim
        if res_feat_dset is None:
                C, F, D = res_feat.shape
                res_feat_dset = f_res.create_dataset('resnet_features', (dataset_size, C, F, D),
                                                dtype=np.float32)
        # if res_video_ids_dset is None:
        #     res_video_ids_dset = f_res.create_dataset('ids', shape=(dataset_size,), dtype=str)
            
        obj_app_feat = np.asarray(obj_app_feat)
        obj_spatial_feat = np.asarray(obj_spatial_feat)
        if obj_app_feat_dset is None:
            C, O, F, D = obj_app_feat.shape
            obj_app_feat_dset = f_obj.create_dataset('appearance_features', (dataset_size, C, O, F, D),
                                                dtype=np.float32)
        if obj_spatial_feat_dset is None:
            C, O, F, D = obj_spatial_feat.shape
            obj_spatial_feat_dset = f_obj.create_dataset('spatial_features', (dataset_size, C, O, F, D),
                                            dtype=np.float32)
        # if obj_video_ids_dset is None:
        #     obj_video_ids_dset = f_obj.create_dataset('ids', shape=(dataset_size,), dtype=str)

        if obj_name_enc_dset is None:
            D = obj_name_enc.shape[0]
            obj_name_enc_dset = f_obj.create_dataset('name_encoded', (dataset_size, D),
                                            dtype=np.int32)
        if obj_name_ids_dset is None:
            O = obj_name_ids.shape[0]
            obj_name_ids_dset = f_obj.create_dataset('name_ids', (dataset_size, O),
                                            dtype=np.int32)


        i1 = i0 + 1
        i3d_feat_dset[i0:i1] = i3d_feat
        # i3d_video_ids_dset[i0:i1] = video_id
        res_feat_dset[i0:i1] = res_feat
        # res_video_ids_dset[i0:i1] = video_id
        obj_app_feat_dset[i0:i1] = obj_app_feat
        obj_spatial_feat_dset[i0:i1] = obj_spatial_feat
        # obj_video_ids_dset[i0:i1] = video_id

        # obj_name_feat_dset[i0:i1] = obj_name_feat
        # f_obj.attrs[str(i0)] = text

        obj_name_enc_dset[i0:i1] = obj_name_enc
        obj_name_ids_dset[i0:i1] = obj_name_ids
        i0 = i1

    f_res.close()
    f_obj.close()
    f_i3d.close()
    print(map_id_name)
    pickle.dump(map_id_name, open(SAVE_DIR + 'map_id_name_8c22f_full.pkl', 'wb'))


if __name__ == '__main__':
    video_ids = [i[:-4] for i in os.listdir(TUBELET_DIR)]
    generate_h5(video_ids)