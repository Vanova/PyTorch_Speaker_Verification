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
from itertools import filterfalse
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
    TODO add caching
    TODO features stacking: fbank + fusion attributes
    """

    def __init__(self, shuffle=True, wnd_size=170, utter_start=0, apply_vad=True, cache=0):
        # data path
        if hp.training:
            self.path = hp.data.train_path
            self.utter_num = hp.train.M
        else:
            self.path = hp.data.test_path
            self.utter_num = hp.test.M

        self.shuffle = shuffle
        self.wnd_size = wnd_size  # (140, 180)
        self.utter_start = utter_start
        self.apply_vad = apply_vad

        depends = [os.path.join(self.path, x) for x in ['feats.scp', 'spk2utt']]

        self.feat_reader = kaldiio.load_scp(depends[0])
        self.spk2utt = kio.Reader(depends[1], num_tokens=-1)
        self.speakers = self.spk2utt.index_keys

        if self.apply_vad:
            vadf = os.path.join(self.path, 'vad.scp')
            self.vadscp = kaldiio.load_scp(vadf)
            print('[INFO] applying VAD: %s' % vadf)
            # TODO self._remove_silent_utter()
        else:
            print('[INFO] do not apply VAD, expect non silent files are fed')

    def __len__(self):
        return len(self.speakers)

    def __getitem__(self, idx):
        """
        Sample one speaker with M utterances
        index: Integer, batch index
        """
        # select random speaker
        if self.shuffle:
            tmp_spk = rnd.sample(self.speakers, 1)[0]
        else:
            tmp_spk = self.speakers[idx][0]

        return self._generate_data(tmp_spk)

    def _generate_data(self, tmp_id):
        utt_sets = self.spk2utt[tmp_id]
        if self.apply_vad: # TODO clean silent files during initialization!!!
            utt_sets = self._remove_silent_utter(utt_sets)

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

        # dimensions [batch(M), frames, n_mels]
        utterance = np.stack(chunks)
        utterance = torch.tensor(utterance)
        return utterance

    def _get_chunk(self, uttid):
        utt = self.feat_reader[uttid]
        if self.apply_vad:
            utt = self._apply_vad(uttid, utt)

        pad = utt.shape[0] - self.wnd_size
        if pad > 0:  # random chunk of spectrogram
            start = rnd.randint(0, pad)
            chunk = utt[start:start + self.wnd_size]
        else:
            chunk = np.pad(utt, ((-pad, 0), (0, 0)), 'edge')
        return chunk

    def _remove_silent_utter(self, utt_sets):
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


class ARKUtteranceGenerator(Dataset):
    def __init__(self, data_dir, wnd_size=170, rnd_chunks=True, apply_vad=True, cache=0):
        # data path
        self.path = data_dir
        self.wnd_size = wnd_size
        self.hop_size = wnd_size // 2
        self.rnd_chunks = rnd_chunks
        self.apply_vad = apply_vad
        self.num_chunks = 128  # int(self._average_vad() // self.hop_size)
        print('[INFO] number of windows sampled from every file: %d' % self.num_chunks)

        depends = [os.path.join(self.path, x) for x in ['feats.scp', 'vad.scp']]

        self.feat_reader = kaldiio.load_scp(depends[0])
        self.utter_list = list(self.feat_reader.keys())

        if self.apply_vad:
            vadf = os.path.join(self.path, 'vad.scp')
            self.vadscp = kaldiio.load_scp(vadf)
            print('[INFO] applying VAD: %s' % vadf)
            self._remove_silent_utter()
        else:
            print('[INFO] do not apply VAD, expect non silent files are fed')

    def __len__(self):
        return len(self.utter_list)

    def __getitem__(self, idx):
        """
        Sample one speaker with M utterances
        index: Integer, batch index
        """
        # for API consistency
        return self._generate_data(idx)

    def _generate_data(self, ut_id):
        ut_id = self.utter_list[ut_id]
        # sliding utterance
        if self.rnd_chunks:
            chunks = self._get_random_chunks(ut_id)
        else:
            chunks = self._get_sequential_chunks(ut_id)
        # dimensions [n_chunks, frames, n_mels]
        return ut_id, torch.tensor(chunks)

    def _get_random_chunks(self, uttid):
        """ Randomly sample num_chunks of wnd_size"""
        # random wnd_size
        utt = self.feat_reader[uttid]
        if self.apply_vad:
            utt = self._apply_vad(uttid, utt)

        T, F = utt.shape
        n_frames = T - self.wnd_size
        if n_frames <= 0:
            print('[WARN] file is shorter than wnd: %s, %d/%d' % (uttid, T, self.wnd_size))
            # pad until the ful chunk
            chunk = np.pad(utt, ((-n_frames, 0), (0, 0)), 'edge')
            # fil with the copy of the chunk
            chunks = np.repeat(chunk[np.newaxis, :, :], self.num_chunks, axis=0)
            chunks = chunks.astype(np.float32)
        else:
            starts = np.random.randint(0, n_frames-1, size=self.num_chunks)
            starts.sort()
            chunks = np.zeros((self.num_chunks, self.wnd_size, F), dtype=np.float32)
            for id, s in enumerate(starts):
                chunks[id] = utt[s:s + self.wnd_size]
        return chunks

    def _get_sequential_chunks(self, uttid):
        """ Sliding utterance with wnd_size """
        utt = self.feat_reader[uttid]
        if self.apply_vad:
            utt = self._apply_vad(uttid, utt)

        T, F = utt.shape
        # step: half chunk
        S = self.hop_size
        N = (T - self.wnd_size) // S + 1
        if N <= 0:
            print('[WARN] file is shorter than wnd: %s' % uttid)
            # pad until the ful chunk
            pad = self.wnd_size - T
            chunk = np.pad(utt, ((pad, 0), (0, 0)), 'edge')
            # fil with the copy of the chunk
            chunks = np.repeat(chunk[np.newaxis, :, :], self.num_chunks, axis=0)
            chunks = chunks.astype(np.float32)
            return chunks
        elif N == 1:
            return utt[:self.wnd_size]
        else:
            chunks = np.zeros((N, self.wnd_size, F))
            for n in range(N):
                chunks[n] = utt[n * S:n * S + self.wnd_size]
            return chunks

    def _remove_silent_utter(self):
        old_list = self.utter_list.copy()
        self.utter_list[:] = filterfalse(self._is_silent_utter, self.utter_list)
        if len(old_list) != len(self.utter_list):
            silfn = set(old_list) - set(self.utter_list)
            print('[WARN] Silent files are removed: %d' % len(silfn))
            print(silfn)

    def _is_silent_utter(self, uid):
        vad = self.vadscp[uid]
        return sum(vad) == 0

    def _apply_vad(self, uttid, utt):
        vad = self.vadscp[uttid]
        feat = utt[vad > 0]
        return feat

    def _average_vad(self):
        cum = 0
        for vid in self.vadscp:
            cum += sum(self.vadscp[vid])
        return cum // len(self.vadscp)


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
