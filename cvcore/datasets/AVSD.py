import numpy as np
import json
import pickle
import torch
import math
import h5py
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate


I3D_MTN = "i3d_flow/"
USE_I3D = True

def invert_dict(d):
    return {v: k for k, v in d.items()}

def load_vocab(path):
    with open(path, 'r') as f:
        vocab = json.load(f)
        vocab['question_idx_to_token'] = invert_dict(vocab['question_token_to_idx'])
    return vocab


class AVSDDataset(Dataset):
    def __init__(self, cfg, mode="train"):
        print('Loading vocab from %s' % (cfg.VOCAB_FILE))
        self.vocab = load_vocab(cfg.INPUT_DIR + cfg.VOCAB_FILE)
        self.is_training = False
        if mode == "train":
            question_pt_path = cfg.INPUT_DIR + cfg.TRAIN_QUESTION_PT
            self.is_training = True
        elif mode == "val":
            question_pt_path = cfg.INPUT_DIR + cfg.VAL_QUESTION_PT
        elif mode == "test":
            question_pt_path = cfg.INPUT_DIR + cfg.TEST_QUESTION_PT

        print('Loading questions from %s' % (question_pt_path))
        with open(question_pt_path, 'rb') as f:
            obj = pickle.load(f)
            dialogs = obj['dialogs']
            dialogs_len = obj['dialogs_len']
            questions = obj['questions']
            questions_len = obj['questions_len']
            q_video_indices = obj['video_ids']
            answers = obj['answers']
            answers_len = obj['answers_len']
            # glove_matrix = obj['glove']

        self.all_dialogs = torch.from_numpy(np.asarray(dialogs)).long()
        self.all_dialogs_len = torch.from_numpy(np.asarray(dialogs_len)).long()
        
        self.all_questions = torch.from_numpy(np.asarray(questions)).long()
        self.all_questions_len = torch.from_numpy(
            np.asarray(questions_len)).long()

        self.all_answers = torch.from_numpy(np.asarray(answers)).long()
        self.all_answers_len = torch.from_numpy(
            np.asarray(answers_len)).long()

        self.map_id_name = pickle.load(open(cfg.INPUT_DIR + cfg.MAP_ID_NAME, 'rb'))
        self.map_name_id = invert_dict(self.map_id_name)
        
        all_idx = np.asarray([self.map_name_id[qname] for qname in q_video_indices])
        self.all_q_video_idxs = torch.from_numpy(all_idx).long()
        
        print('loading appearance feature from %s' % (cfg.RESNET_FEATURE_H5))
        self.app_feature_h5 = h5py.File(cfg.INPUT_DIR + cfg.RESNET_FEATURE_H5, 'r')

        print('loading motion feature from %s' % (cfg.I3D_FEATURE_H5))
        self.motion_feature_h5 = h5py.File(cfg.INPUT_DIR + cfg.I3D_FEATURE_H5, 'r')
        
        print('loading obj feature from %s' % (cfg.OBJECT_FEATURE_H5))
        self.obj_feature_h5 = h5py.File(cfg.INPUT_DIR + cfg.OBJECT_FEATURE_H5, 'r')
        
        # Only 1/3 datasets are used for training
        # available_vid = pickle.load(open('available_vid_test.pkl', 'rb'))
        # available_vid_map = {}
        # for vid in q_video_indices:
        #     available_vid_map[vid] = False
        # for vid in available_vid:
        #     available_vid_map[vid] = True
        # available_indicies = []
        # for i in range(len(q_video_indices)):
        #     if available_vid_map[q_video_indices[i]]:
        #         available_indicies.append(i)
        # self.all_dialogs = self.all_dialogs[available_indicies]
        # self.all_dialogs_len = self.all_dialogs_len[available_indicies]
        # self.all_questions = self.all_questions[available_indicies]
        # self.all_questions_len = self.all_questions_len[available_indicies]
        # self.all_answers = self.all_answers[available_indicies]
        # self.all_answers_len = self.all_answers_len[available_indicies]
        # self.all_q_video_idxs = self.all_q_video_idxs[available_indicies]
        print(f"# {mode} video: {len(list(set(self.all_q_video_idxs)))}")
        print(f"# {mode} sample: {len(self.all_questions)}")
        

    def __getitem__(self, index):
        dialog = self.all_dialogs[index]
        dialog_len = self.all_dialogs_len[index]

        answer = self.all_answers[index].numpy()
        answer_len = self.all_answers_len[index]

        end_idx = answer_len - 1
        if self.is_training:
            cut_a_p = 0.5
            pr = np.random.uniform()
            if pr >= (1-cut_a_p):
                end_idx = np.random.choice(range(1, answer_len), 1)[0]
       
        answer_in = list(answer[:end_idx])
        answer_out = list(answer[1:end_idx])
        answer_out.append(answer[end_idx])
        answer_len = len(answer_in)

        # max_answer_length = 57
        # while len(answer_in) < max_answer_length:
        #     answer_in.append(self.vocab['answer_token_to_idx']['<NULL>'])
        #     answer_out.append(self.vocab['answer_token_to_idx']['<NULL>'])

        question = self.all_questions[index]
        question_len = self.all_questions_len[index]
        video_idx = self.all_q_video_idxs[index].item()

        obj_app_feat = self.obj_feature_h5['appearance_features'][video_idx]
        obj_spatial_feat = self.obj_feature_h5['spatial_features'][video_idx]
        obj_name_enc = self.obj_feature_h5['name_encoded'][video_idx]
        obj_name_ids = self.obj_feature_h5['name_ids'][video_idx]

        if USE_I3D:
            # motion_feat = np.ones((256,2048), dtype=np.float32)
            # motion_feat_load = np.load(f"{I3D_MTN}{self.map_id_name[video_idx]}.npy")
            # motion_feat[:min(motion_feat_load.shape[0], 256), :] = motion_feat_load[:256, :]
            motion_feat = np.load(f"{I3D_MTN}{self.map_id_name[video_idx]}.npy")
        else:
            motion_feat = self.motion_feature_h5['resnet_features'][video_idx]
        appearance_feat = self.app_feature_h5['resnet_features'][video_idx]
        obj_app_feat = torch.from_numpy(obj_app_feat).float()
        obj_spatial_feat = torch.from_numpy(obj_spatial_feat).float()
        obj_name_enc = torch.from_numpy(obj_name_enc).long()
        obj_name_ids = torch.from_numpy(obj_name_ids).long()

        appearance_feat = torch.from_numpy(appearance_feat)
        # motion_feat = torch.from_numpy(motion_feat)
        # return ( video_idx, torch.from_numpy(np.array(answer_in)), torch.from_numpy(np.array(answer_out)), answer_len, 
        #             question, question_len,
        #             dialog, dialog_len,
        #             obj_app_feat, obj_spatial_feat,
        #             appearance_feat,
        #             motion_feat )
        return ( video_idx, answer_in, answer_out, answer_len, 
                    question, question_len,
                    dialog, dialog_len,
                    obj_app_feat, obj_spatial_feat, 
                    obj_name_enc, obj_name_ids,
                    appearance_feat,
                    motion_feat )

    def __len__(self):
        return len(self.all_questions)
