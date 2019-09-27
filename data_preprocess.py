#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Modified from https://github.com/JanhHyun/Speaker_Verification
import glob
import os
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from p_tqdm import p_imap
import kaldi_python_io as kio
import kaldiio
from hparam import hparam as hp
import utils.io as uio
import features.speech as F


class DataProcessor(object):
    def __init__(self, data_dir, subsets, seconds=3, feat_type=[],
                 feat_params=[], pad=True, cache=True):
        self.data_dir = data_dir
        self.subsets = subsets  # let all sets are combined
        self.len_allowed = seconds
        self.feat_type = feat_type
        self.feat_params = feat_params
        self.pad = pad
        self.cache = cache
        self.meta_data = None
        print('[Initialising] SRE dataset: length = {}s and subsets = {}'.format(seconds, subsets))

    @property
    def meta(self):
        if not self.meta_data:
            self.initialize()
        return self.meta_data

    def initialize(self):
        """
        Load data, prepare data meta information
        """
        cached_df = []
        found_cache = {s: False for s in self.subsets}
        if self.cache:
            for s in self.subsets:
                subset_index_path = 'data/%s.index.csv' % s
                if os.path.exists(subset_index_path):
                    cached_df.append(pd.read_csv(subset_index_path))
                    found_cache[s] = True

        # Index the remaining subsets if any
        if all(found_cache.values()) and self.cache:
            self.meta_data = pd.concat(cached_df)
        else:

            df = pd.DataFrame(columns=['file_id', 'speaker_id', 'file_path', 'pipeline',
                                       'samples', 'length', 'channel_id', 'subset'])
            for subset, found in found_cache.items():
                if not found:
                    subset_path = os.path.join(self.data_dir, subset)
                    tmp_df = self.index_subset(subset_path)
                    tmp_df['subset'] = [subset] * len(tmp_df)
                    # Merge individual audio files with indexing dataframe
                    df = df.append(tmp_df)

            # Concatenate with existing dataframe
            self.meta_data = pd.concat(cached_df + [df])
        # Dump index files
        for s in self.subsets:
            self.meta_data[self.meta_data['subset'] == s].to_csv('data/{}.index.csv'.format(s), index=False)
        # Trim too-short files
        if not self.pad:
            self.meta_data = self.meta_data[self.meta_data['length'] > self.len_allowed]

        self.meta_data = self.meta_data.reset_index(drop=True)
        self.meta_data.set_index('file_id', inplace=True)

        self.n_speakers = len(self.meta_data['speaker_id'].unique())
        self.n_utterances = len(self.meta_data)
        print('Utterances: %d, unique speakers: %d.' % (self.n_utterances, self.n_speakers))

    def extract_features(self):
        # TODO it is very slooow compare to Kaldi
        # TODO test with one CPU without Pool :)
        print('[Extraction] features from %d files' % len(self.meta_data))
        # prepare extractor
        extractor = F.prepare_extractor(feats=self.feat_type[0], params=self.feat_params[self.feat_type[0]])
        fids = self.meta_data.index.values.tolist()
        iterator = p_imap(lambda fid: self._extraction_job(fid, extractor), fids)

        hdf_file = os.path.join(data_dir, '%s_%s.hdf' % ('_'.join(self.subsets), self.feat_type[0]))
        writer = uio.HDFWriter(file_name=hdf_file)
        for result in iterator:
            fid = result['file_id']
            if result['feat'] is None:
                fmeta = self.meta_data.loc[fid]
                print('[WARN] empty %s: %s' % (self.feat_type[0], fmeta))
            else:
                writer.append(file_id=fid, feat=result['feat'])
        writer.close()
        del writer

    def apply_kaldi_vad(self):
        """
        Apply VAD to Kaldi fbanks and save to HDF
        """
        subset_path = os.path.join(self.data_dir, self.subsets[0])
        print('[Process] Kaldi features with VAD %s...' % subset_path)
        depends = [os.path.join(subset_path, x) for x in ['vad.scp', 'feats.scp']]
        for depend in depends:
            if not os.path.exists(depend):
                raise RuntimeError('Missing file {}!'.format(depend))

        vads = kaldiio.load_scp(depends[0])
        feats = kaldiio.load_scp(depends[1])
        hdf_file = os.path.join(subset_path, 'feats_vad.hdf')
        writer = uio.HDFWriter(file_name=hdf_file)

        cached_fids = self.meta_data.index.values.tolist()
        cnt = 0
        for fid in cached_fids:
            vad = vads[fid]
            feat = feats[fid]
            feat = feat[vad > 0]
            writer.append(file_id=fid, feat=feat)
            cnt += 1
            print("%d. processed: %s" % (cnt, fid))
            if sum(vad) < 50:  # 8000/200 * 0.025 = 1 sec | 8000 * 0.025 = 200 samples in one frame
                print('[WARN] too short audio after VAD: %s' % self.meta_data.loc[fid]['file_path'])
        writer.close()

    @staticmethod
    def index_subset(subset_path):
        """
        Index a subset by looping through all files and dumping their speaker ID, filepath and length.
            subset_path: Name of the subset
        """
        print('Indexing %s...' % subset_path)
        depends = [os.path.join(subset_path, x) for x in ['utt2spk', 'wav.scp']]
        for depend in depends:
            if not os.path.exists(depend):
                raise RuntimeError('Missing file {}!'.format(depend))

        utt2spk = kio.Reader(depends[0], num_tokens=-1)
        wavscp = uio.load_dictionary(depends[1], delim=' ')
        assert set(wavscp.keys()) == set(utt2spk.index_keys)

        # format of wav.scp: {<file_id> : ['sph2pipe', '-f', 'wav', '-p', '-c', '2', '<file_path>', '|']}
        uall = []
        sall = []
        fpaths = []
        pipes = []
        ch_ids = []
        for u, s in utt2spk.index_dict.items():
            if len(wavscp[u]) == 8:  # TODO fix for .flac
                uall.append(u)
                sall.append(s[0])
                # fpaths.append(wavscp[u][6])
                fpaths.append(os.path.join(subset_path, 'wav', wavscp[u][6]))
                pipes.append(wavscp[u])
                ch_ids.append(wavscp[u][5])
            else:
                print('[Warning] unknown pipeline format for %s: %s' % (u, ' '.join(wavscp[u])))

        df = pd.DataFrame(columns=['file_id', 'speaker_id', 'file_path', 'pipeline', 'samples', 'length'])
        df['file_id'] = uall
        df['speaker_id'] = sall
        df['file_path'] = fpaths
        df['pipeline'] = pipes
        df['channel_id'] = ch_ids
        # go through the data
        iterator = p_imap(lambda fn: DataProcessor._indexing_job(fn),
                          df['file_path'].values.tolist())

        for result in iterator:
            fp = result['file_path']
            df['samples'][df['file_path'] == fp] = result['samples']
            df['length'][df['file_path'] == fp] = result['length']

        print('Files processed: %d' % len(df))
        assert not df.isnull().any().any()
        return df

    def _extraction_job(self, fid, extractor):
        file_path = self.meta_data.loc[fid]['file_path']
        ch = self.meta_data.loc[fid]['channel_id']
        x, fs = sf.read(file_path)
        # signal, fs = uio.load_sph(full_fpath)
        if len(x.shape) > 1:
            x = x[:, ch - 1]
        feat = extractor.extract(x, fs)
        return {'file_id': fid,
                'feat': feat}

    @staticmethod
    def _indexing_job(file_path):
        signal, fs = sf.read(file_path)
        # signal, fs = uio.load_sph(full_fpath)
        assert fs == hp.data.sr
        return {'file_path': file_path,
                'length': len(signal) / float(hp.data.sr),
                'samples': len(signal)}


if __name__ == "__main__":
    # train_set = 'toy_dataset'
    train_set = 'swbd_sre_small_fbank'
    data_dir = '/home/vano/wrkdir/projects_data/sre_2019/'
    feat_type = ['vad', 'fbank', 'fuse_manner', 'fuse_place']
    feat_params = {
        'vad': {
            'vad_type': 'energy',  # 'webrtc'
            'min_len': 1,  # seconds
            'energy_lvl': 30  # db
        },
        'fbank': {
            'sample_rate': hp.data.sr,
            'win_length_seconds': hp.data.window,
            'hop_length_seconds': hp.data.hop,
            'bands': hp.data.nmels,
            'fmin': 0,
            'fmax': hp.data.sr // 2,
            'include_delta': False,
            'include_acceleration': False,
            'n_fft': hp.data.nfft,
            'mono': True,
            'window': 'hamming_asymmetric',
            'vad_type': 'energy'  # 'webrtc'
        }
    }

    dp = DataProcessor(data_dir=data_dir, subsets=[train_set])
    dp.initialize()
    # dp.extract_features()
    dp.apply_kaldi_vad()
