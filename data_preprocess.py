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
from hparam import hparam as hp
import utils.io as uio
import features.speech as F


# audio_path = glob.glob(os.path.dirname(hp.unprocessed_data))


class DataProcessor(object):

    def __init__(self, data_dir, subsets, seconds=3, feat_type=['vad', 'fbank', 'fuse_manner', 'fuse_place'],
                 feat_params={}, pad=True, cache=True):
        self.data_dir = data_dir
        self.subsets = subsets
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
        # TODO extract features with VAD either
        # TODO note, it is better to extract fbank features in Kaldi!!!
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


def save_spectrogram_tisv():
    """ Full preprocess of text independent utterance. The log-mel-spectrogram is saved as numpy file.
        Each partial utterance is splitted by voice detection using DB
        and the first and the last 180 frames from each partial utterance are saved. 
        Need : utterance data set (VTCK)
    """
    print("start text independent utterance feature extraction")
    os.makedirs(hp.data.train_path, exist_ok=True)  # make folder to save train file
    os.makedirs(hp.data.test_path, exist_ok=True)  # make folder to save test file

    utter_min_len = (hp.data.tisv_frame * hp.data.hop + hp.data.window) * hp.data.sr  # lower bound of utterance length
    total_speaker_num = len(audio_path)
    train_speaker_num = (total_speaker_num // 10) * 9  # split total data 90% train and 10% test
    print("total speaker number : %d" % total_speaker_num)
    print("train : %d, test : %d" % (train_speaker_num, total_speaker_num - train_speaker_num))
    for i, folder in enumerate(audio_path):
        print("%dth speaker processing..." % i)
        utterances_spec = []
        for utter_name in os.listdir(folder):
            if utter_name[-4:] == '.WAV':
                utter_path = os.path.join(folder, utter_name)  # path of each utterance
                utter, sr = librosa.core.load(utter_path, hp.data.sr)  # load utterance audio
                intervals = librosa.effects.split(utter, top_db=30)  # voice activity detection
                for interval in intervals:
                    if (interval[1] - interval[0]) > utter_min_len:  # If partial utterance is sufficient long,
                        utter_part = utter[interval[0]:interval[1]]  # save first and last 180 frames of spectrogram.
                        S = librosa.core.stft(y=utter_part, n_fft=hp.data.nfft,
                                              win_length=int(hp.data.window * sr), hop_length=int(hp.data.hop * sr))
                        S = np.abs(S) ** 2
                        mel_basis = librosa.filters.mel(sr=hp.data.sr, n_fft=hp.data.nfft, n_mels=hp.data.nmels)
                        S = np.log10(np.dot(mel_basis, S) + 1e-6)  # log mel spectrogram of utterances
                        utterances_spec.append(S[:, :hp.data.tisv_frame])  # first 180 frames of partial utterance
                        utterances_spec.append(S[:, -hp.data.tisv_frame:])  # last 180 frames of partial utterance

        utterances_spec = np.array(utterances_spec)
        print(utterances_spec.shape)
        if i < train_speaker_num:  # save spectrogram as numpy file
            np.save(os.path.join(hp.data.train_path, "speaker%d.npy" % i), utterances_spec)
        else:
            np.save(os.path.join(hp.data.test_path, "speaker%d.npy" % (i - train_speaker_num)), utterances_spec)


if __name__ == "__main__":
    train_set = 'toy_dataset'
    data_dir = '/home/vano/wrkdir/projects_data/sre_2019/'
    feat_params = {'vad': {
        'vad_type': 'energy',  # 'webrtc'
        'min_len': 1,  # seconds
        'energy_lvl': 30 # db
    }}
    dp = DataProcessor(data_dir=data_dir, subsets=[train_set],
                       feat_type=['vad'], feat_params=feat_params)
    dp.initialize()
    dp.extract_features()
