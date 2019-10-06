"""
    Extract d-vector features
    =======================================================
    Usage:
        run_sltc.py [--datadir DIR] [--outdir DIR]
        run_sltc.py (-h | --help)

    Options:
        --datadir DIR   Model configuration file.
        --outdir DIR    Model configuration file            [default: ./].
        -h --help               Show this screen.
"""
import os
import time
import torch
import kaldiio
from torch.utils.data import DataLoader
from docopt import docopt
from hparam import hparam as hp
from speech_embedder_net import SpeechEmbedder
import data_load as DL


def main(args):
    if args['--datadir']:
        data_dir = args['datadir']
    else:
        data_dir = hp.data.eval_path
    device = torch.device(hp.device)
    print('[INFO] device: %s' % device)
    dataset_name = os.path.basename(os.path.normpath(data_dir))
    print('[INFO] dataset: %s' % dataset_name)

    # Load model
    embed_net = SpeechEmbedder().to(device)
    embed_net.load_state_dict(torch.load(hp.model.model_path))
    embed_net.eval()
    # Features
    eval_gen = DL.ARKUtteranceGenerator(data_dir)
    eval_loader = DataLoader(eval_gen, batch_size=hp.test.M, shuffle=False,
                             num_workers=hp.test.num_workers,
                             drop_last=False)
    dwriter = kaldiio.WriteHelper('ark,scp:%s_dvecs.ark,%s_dvecs.scp' % (dataset_name, dataset_name))

    cnt = 0
    processed = []
    for key_bt, feat_bt in eval_loader:
        feat_bt = feat_bt.to(device)
        t_start = time.time()
        # feat dim [M_files, n_chunks_in_file, frames, n_mels]
        mf, nchunks, frames, nmels = feat_bt.shape
        print(feat_bt.shape)
        stack_shape = (mf * nchunks, frames, nmels)

        feat_stack = torch.reshape(feat_bt, stack_shape)
        dvec_stack = embed_net(feat_stack)
        dvec_bt = torch.reshape(dvec_stack, (mf, dvec_stack.size(0) // mf, dvec_stack.size(1)))

        for key, dvec in zip(key_bt, dvec_bt):
            mean_dvec = torch.mean(dvec, dim=0).detach()
            mean_dvec = mean_dvec.cpu().numpy()
            dwriter(key, mean_dvec)
            processed.append(key)
            print('%d. Processed: %s' % (cnt, key))
            cnt += 1
        t_end = time.time()
        print('Elapsed: %.4f' % (t_end - t_start))


if __name__ == '__main__':
    arguments = docopt(__doc__)
    main(arguments)

