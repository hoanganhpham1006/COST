import torch
import numpy as np

def collate_batch(batch):
    batch_size = len(batch)

    batch_new = []
    for j in range(14):
        data = [batch[i][j] for i in range(batch_size)]
        if (j == 1 or j == 2) or j == 13:
            batch_new.append(data)
        elif j in [0, 3, 5]:
            batch_new.append(torch.from_numpy(np.array(data)))
        else:
            data_placeholder = torch.ones((batch_size, *data[0].shape), dtype=data[0].dtype)
            for i in range(batch_size):
                data_placeholder[i] = data[i]
            batch_new.append(data_placeholder)

    answer_in, answer_out, motion_feat_np = batch_new[1], batch_new[2], batch_new[13]

    answer_len = batch_new[3]
    motion_len = [m.shape[0] for m in motion_feat_np]

    answer_in_new = torch.ones((batch_size, max(answer_len)), dtype=torch.int64)
    answer_out_new = torch.ones((batch_size, max(answer_len)), dtype=torch.int64)
    motion_feat_new = torch.ones((batch_size, max(motion_len), motion_feat_np[0].shape[-1]), dtype=torch.float32)

    for i in range(batch_size):
        answer_in_new[i, :answer_len[i]] = torch.from_numpy(np.array(answer_in[i]))
        answer_out_new[i, :answer_len[i]] = torch.from_numpy(np.array(answer_out[i]))
        motion_feat_new[i, :motion_len[i]] = torch.from_numpy(np.array(motion_feat_np[i]))
    
    return  (batch_new[0], answer_in_new, answer_out_new, *batch_new[3:13], motion_feat_new)