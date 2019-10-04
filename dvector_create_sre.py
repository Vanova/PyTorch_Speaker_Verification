import os
import torch
import kaldiio
from torch.utils.data import DataLoader
from hparam import hparam as hp
from speech_embedder_net import SpeechEmbedder
import data_load as DL
import time

device = torch.device(hp.device)
print('[INFO] device: %s' % device)
dataset_name = os.path.basename(os.path.normpath(hp.data.eval_path))
print('[INFO] dataset: %s' % dataset_name)
# Load model
embed_net = SpeechEmbedder().to(device)
embed_net.load_state_dict(torch.load(hp.model.model_path))
embed_net.eval()
# Features
eval_gen = DL.ARKUtteranceGenerator()
eval_loader = DataLoader(eval_gen, batch_size=hp.test.M, shuffle=False,
                         num_workers=hp.test.num_workers,
                         drop_last=True)
dwriter = kaldiio.WriteHelper('ark,scp:%s_dvecs.ark,%s_dvecs.scp' % (dataset_name, dataset_name))

cnt = 0
processed = []
for key_bt, feat_bt in eval_loader:
    t_start = time.time()
    print(key_bt)
    print(feat_bt.shape)
    # feat dim [M_files, n_chunks_in_file, frames, n_mels]
    feat_bt = feat_bt.to(device)
    # stack M_files in one array: [M_files x n_chunks_in_file, frames, n_mels]
    stack_shape = (feat_bt.size(0) * feat_bt.size(1), feat_bt.size(2), feat_bt.size(3))
    feat_stack = torch.reshape(feat_bt, stack_shape)

    dvec_stack = embed_net(feat_stack)
    dvec_bt = torch.reshape(dvec_stack, (hp.test.M, dvec_stack.size(0) // hp.test.M, dvec_stack.size(1)))

    for key, dvec in zip(key_bt, dvec_bt):
        mean_dvec = torch.mean(dvec, dim=0).detach()
        mean_dvec = mean_dvec.cpu().numpy()
        dwriter(key, mean_dvec)
        processed.append(key)
        print('%d. Processed: %s' % (cnt, key))
        cnt += 1
    t_end = time.time()
    print('Elapsed: %.4f' % (t_end - t_start))

# Process the rest of the files
rest_files = set(eval_gen.utter_list) - set(processed)
for key in rest_files:
    id = eval_gen.utter_list.index(key)
    _, feat = eval_gen._generate_data(id)
    dvec = embed_net(feat)
    mean_dvec = torch.mean(dvec, dim=0).detach()
    mean_dvec = mean_dvec.cpu().numpy()
    dwriter(key, mean_dvec)
    print('%d. Post processed: %s' % (cnt, key))
    cnt += 1

dwriter.close()
