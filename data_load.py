#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 20:55:52 2018

@author: harry
"""
import os
import glob
import numpy as np
import random as rnd
import torch
import h5py
import utils
from torch.utils.data import Dataset
import kaldi_python_io as kio
import kaldiio
from hparam import hparam as hp
from utils.net_utils import mfccs_and_spec


class SpeakerDatasetTIMIT(Dataset):

    def __init__(self):

        if hp.training:
            self.path = hp.data.train_path_unprocessed
            self.utterance_number = hp.train.M
        else:
            self.path = hp.data.test_path_unprocessed
            self.utterance_number = hp.test.M
        self.speakers = glob.glob(os.path.dirname(self.path))
        rnd.shuffle(self.speakers)

    def __len__(self):
        return len(self.speakers)

    def __getitem__(self, idx):

        speaker = self.speakers[idx]
        wav_files = glob.glob(speaker + '/*.WAV')
        rnd.shuffle(wav_files)
        wav_files = wav_files[0:self.utterance_number]

        mel_dbs = []
        for f in wav_files:
            _, mel_db, _ = mfccs_and_spec(f, wav_process=True)
            mel_dbs.append(mel_db)
        return torch.Tensor(mel_dbs)


class SpeakerDatasetTIMITPreprocessed(Dataset):

    def __init__(self, shuffle=True, utter_start=0):

        # data path
        if hp.training:
            self.path = hp.data.train_path
            self.utter_num = hp.train.M
        else:
            self.path = hp.data.test_path
            self.utter_num = hp.test.M
        self.file_list = os.listdir(self.path)
        self.shuffle = shuffle
        self.utter_start = utter_start

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):

        np_file_list = os.listdir(self.path)

        if self.shuffle:
            selected_file = rnd.sample(np_file_list, 1)[0]  # select random speaker
        else:
            selected_file = np_file_list[idx]

        utters = np.load(os.path.join(self.path, selected_file))  # load utterance spectrogram of selected speaker
        if self.shuffle:
            utter_index = np.random.randint(0, utters.shape[0], self.utter_num)  # select M utterances per speaker
            utterance = utters[utter_index]
        else:
            utterance = utters[
                        self.utter_start: self.utter_start + self.utter_num]  # utterances of a speaker [batch(M), n_mels, frames]

        utterance = utterance[:, :, :160]  # TODO implement variable length batch size

        utterance = torch.tensor(np.transpose(utterance, axes=(0, 2, 1)))  # transpose [batch, frames, n_mels]
        return utterance


class ARKDataGenerator(Dataset):
    """
    Train on Kaldi ark features, apply VAD on the fly
    """
    def __init__(self, shuffle=True, wnd_size=170, utter_start=0):
        # data path
        if hp.training:
            self.path = hp.data.train_path
            self.utter_num = hp.train.M
        else:
            self.path = hp.data.test_path
            self.utter_num = hp.test.M

        depends = [os.path.join(self.path, x) for x in ['feats.scp', 'spk2utt', 'vad.scp']]
        for depend in depends:
            if not os.path.exists(depend):
                raise RuntimeError('Missing file {}!'.format(depend))

        self.shuffle = shuffle
        self.utter_start = utter_start

        self.wnd_size = wnd_size  # (140, 180)
        self.feat_reader = kaldiio.load_scp(depends[0])
        self.vadscp = kaldiio.load_scp(depends[2])
        self.spk2utt = kio.Reader(depends[1], num_tokens=-1)
        self.speakers = self.spk2utt.index_keys

    def __len__(self):
        return len(self.speakers)

    def __getitem__(self, idx):
        """
        Sample one speaker with M utterances
        index: Integer, batch index
        """
        if self.shuffle:
            tmp_speaker = rnd.sample(self.speakers, 1)[0]  # select random speaker
        else:
            tmp_speaker = self.speakers[idx][0]

        return self._generate_data(tmp_speaker)

    def _generate_data(self, tmp_id):
        utt_sets = self.spk2utt[tmp_id]
        utt_sets = self._remove_empty_utterances(utt_sets)

        # utterances of a speaker [batch(M), n_mels, frames]
        if self.shuffle:
            # select M utterances per speaker
            utter_ids = np.random.choice(utt_sets, self.utter_num)
            # without repetition
            # utter_ids = rnd.sample(utt_sets, self.utter_num)
        else:
            utter_ids = utt_sets[:, self.utter_num]

        chunks = []
        for uttid in utter_ids:
                chunk = self._get_chunk(uttid)
                chunks.append(chunk)

        # dimensions [batch, frames, n_mels]
        utterance = np.stack(chunks)
        utterance = torch.tensor(utterance)
        return utterance

    def _get_chunk(self, uttid):
        utt = self.feat_reader[uttid]
        utt = self._apply_vad(uttid, utt)

        pad = utt.shape[0] - self.wnd_size
        if pad > 0:  # random chunk of spectrogram
            start = rnd.randint(0, pad)
            chunk = utt[start:start + self.wnd_size]
        else:
            chunk = np.pad(utt, ((-pad, 0), (0, 0)), 'edge')
        return chunk

    def _remove_empty_utterances(self, utt_sets):
        non_empty = []
        for uid in utt_sets:
            vad = self.vadscp[uid]
            if sum(vad):
                non_empty.append(uid)
            else:
                print('[WARN] utterance is 0 len: %s' % uid)
        return non_empty

    def _apply_vad(self, uttid, utt):
        vad = self.vadscp[uttid]
        feat = utt[vad > 0]
        return feat


class HDFDataGenerator(Dataset):

    def __init__(self, shuffle=True, wnd_size=170, utter_start=0):

        # data path
        if hp.training:
            self.path = hp.data.train_path
            self.utter_num = hp.train.M
        else:
            self.path = hp.data.test_path
            self.utter_num = hp.test.M
        # TODO add subsets

        print('Initialising HDFDatasetGenerator with %d utterances.' % self.utter_num)

        depends = [os.path.join(self.path, x) for x in ['feats.scp', 'spk2utt', 'feats_vad.hdf']]
        for depend in depends:
            if not os.path.exists(depend):
                raise RuntimeError('Missing file {}!'.format(depend))

        self.shuffle = shuffle
        self.utter_start = utter_start

        self.wnd_size = wnd_size
        self.spk2utt = kio.Reader(depends[1], num_tokens=-1)
        self.speakers = self.spk2utt.index_keys
        self.hdf_file = depends[2]

    def __len__(self):
        return len(self.speakers)

    def __getitem__(self, idx):
        """
        Sample one speaker with M utterances
        index: Integer, batch index
        """
        if self.shuffle:
            tmp_speaker = rnd.sample(self.speakers, 1)[0]  # select random speaker
        else:
            tmp_speaker = self.speakers[idx][0]

        return self._generate_data(tmp_speaker)

    def _generate_data(self, tmp_id):
        # load utterance spectrogram of selected speaker
        utt_sets = self.spk2utt[tmp_id]

        if len(utt_sets) < self.utter_num:
            raise RuntimeError('Speaker {} can not got enough utterance with M = {:d}'.
                               format(tmp_id, self.utter_num))

        # utterances of a speaker [batch(M), n_mels, frames, 1]
        if self.shuffle:
            # select M utterances per speaker
            utter_ids = rnd.sample(utt_sets, self.utter_num)
        else:
            utter_ids = utt_sets[:, self.utter_num]

        chunks = []
        for uttid in utter_ids:
            with h5py.File(self.hdf_file, 'r') as feat_reader:
                utt = np.array(feat_reader[uttid], dtype=np.float32)

            pad = utt.shape[0] - self.wnd_size
            if pad > 0:  # random chunk of spectrogram
                start = rnd.randint(0, pad)
                chunks.append(utt[start:start + self.wnd_size])
            else:
                chunk = np.pad(utt, ((-pad, 0), (0, 0)), 'edge')
                chunks.append(chunk)

        # utterance = utterance[:, :, :160]  # TODO implement variable length batch size
        # dimensions [batch, frames, n_mels]
        utterance = np.stack(chunks)
        return torch.tensor(utterance)
