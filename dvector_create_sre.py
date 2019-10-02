#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import numpy as np
import kaldiio
from hparam import hparam as hp
from speech_embedder_net import SpeechEmbedder
import data_load as DL

# Load model
embed_net = SpeechEmbedder()
embed_net.load_state_dict(torch.load(hp.model.model_path))
embed_net.eval()

# Features
eval_gen = DL.ARKUtteranceGenerator()
dwriter = kaldiio.WriteHelper('ark,scp:sre18_dev_dvecs.ark,sre18_dev_dvecs.scp')

cnt = 0
for key, feat in eval_gen:
    dvec = embed_net(feat) # N x D
    mean_dvec = np.mean(dvec.detach().numpy(), axis=0)
    dwriter(key, mean_dvec)
    print('%d. Processed: %s' % (cnt, key))
    cnt += 1
dwriter.close()
