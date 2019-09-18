import librosa
import numpy as np
from base.feature import BaseFeatureExtractor

MFCC_DEFAULT = {
    'type': 'mfcc',
    'win_length_seconds': 0.04,  # def: 0.04
    'hop_length_seconds': 0.02,  # def: 0.02
    'include_mfcc0': False,
    'include_delta': False,
    'include_acceleration': False,
    'window': 'hamming_asymmetric',  # [hann_asymmetric, hamming_asymmetric]
    'n_mfcc': 14,  # Number of MFCC coefficients
    'n_mels': 40,  # Number of MEL bands used
    'n_fft': 1024,  # FFT length
    'fmin': 0,  # Minimum frequency when constructing MEL bands
    'fmax': 8000,  # def: 24000     # Maximum frequency when constructing MEL band
    'delta': {'width': 9},
    'acceleration': {'width': 9}
}

FBANK_DEFAULT = {
    'type': 'fbank',
    'win_length_seconds': 0.04,
    'hop_length_seconds': 0.02,
    'bands': 64,
    'fmin': 0,  # Minimum frequency when constructing MEL bands
    'fmax': 22050,  # def: 24000   # Maximum frequency when constructing MEL band
    'include_delta': False,
    'include_acceleration': False,
    'delta': {'width': 15},
    'acceleration': {'width': 15},
    'n_fft': 2048,
    'mono': True,
    'window': 'hamming_asymmetric'  # [hann_asymmetric, hamming_asymmetric]
}

STFT_DEFAULT = {
    'type': 'stft',
    'win_length_seconds': 0.025,
    'hop_length_seconds': 0.01,
    'fmin': 0,  # Minimum frequency when constructing MEL bands
    'fmax': 22050,  # def: 24000   # Maximum frequency when constructing MEL band
    'include_delta': False,
    'include_acceleration': False,
    'delta': {'width': 15},
    'acceleration': {'width': 15},
    'n_fft': 1024,
    'mono': True,
    'window': 'hamming_asymmetric'  # [hann_asymmetric, hamming_asymmetric]
}

VAD_DEFAULT = {
    'type': 'vad',
    'vad_type': 'energy', # 'webrtc'
    'min_len': 1 # seconds
}


# TODO fix according to factory pattern: https://krzysztofzuraw.com/blog/2016/factory-pattern-python.html

def prepare_extractor(feats='mfcc', params=None):
    if feats == 'mfcc':
        return MFCCExtractor(params=params)
    elif feats == 'fbank':
        return FbankExtractor(params=params)
    elif feats == 'stft':
        return STFTExtractor(params=params)
    elif feats == 'vad':
        return VADExtractor(params=params)
    else:
        raise ValueError("Unknown feature type [" + feats + "]")


class MFCCExtractor(BaseFeatureExtractor):
    def __init__(self, *args, **kwargs):
        super(MFCCExtractor, self).__init__(*args, **kwargs)
        self.params = kwargs.get('params')
        # either params or default
        if self.params is None:
            self.params = MFCC_DEFAULT

    def extract(self, x, sample_rate):
        """
        x: ndarray. 1D numpy array.
        smp_rate: Integer. Sample rate.
        return: 3D array, features [bands; frames; channels]
        """
        wnd_len = int(self.params['win_length_seconds'] * sample_rate)
        hop_len = int(self.params['hop_length_seconds'] * sample_rate)
        # Extract features, Mel Frequency Cepstral Coefficients
        wnd = self._window(wtype=self.params['window'], smp_size=wnd_len)
        # calculate static mfss coefficients
        stft = np.abs(librosa.stft(x + self.EPS,
                                   n_fft=self.params['n_fft'],
                                   win_length=wnd_len,
                                   hop_length=hop_len,
                                   window=wnd)) ** 2

        mel_basis = librosa.filters.mel(sr=sample_rate,
                                        n_fft=self.params['n_fft'],
                                        n_mels=self.params['n_mels'],
                                        fmin=self.params['fmin'],
                                        fmax=self.params['fmax'])
        stft_windowed = np.dot(mel_basis, stft)
        mfcc = librosa.feature.mfcc(S=librosa.amplitude_to_db(stft_windowed))

        emfcc = mfcc[:, :, np.newaxis]
        sh = emfcc.shape
        # consider delta features as the 2nd "image" channel
        if self.params['include_delta']:
            # append delta features as an additional channel: [bands; frames; channels]
            buf = np.zeros(sh)
            emfcc = np.concatenate((emfcc, buf), axis=2)
            emfcc[:, :, 1] = librosa.feature.delta(emfcc[:, :, 0], **self.params['delta'])
        if self.params['include_acceleration']:
            buf = np.zeros(sh)
            emfcc = np.concatenate((emfcc, buf), axis=2)
            emfcc[:, :, 2] = librosa.feature.delta(emfcc[:, :, 1], **self.params['acceleration'])
        return emfcc


class FbankExtractor(BaseFeatureExtractor):
    """
    Log-Mel filter bank feature extractor, steps:
    1. Frame the signal into short frames.
    2. For each frame calculate the periodogram estimate of the power spectrum.
    3. Apply the mel filterbank to the power spectra, sum the energy in each filter.
    4. Take the logarithm of all filterbank energies.
    """

    def __init__(self, *args, **kwargs):
        super(FbankExtractor, self).__init__(*args, **kwargs)
        self.params = kwargs.get('params')
        if not self.params:
            self.params = FBANK_DEFAULT
        self.vad_type = self.params['vad_type']

    def extract(self, x, smp_rate):
        """
        x: ndarray. 1D numpy array.
        smp_rate: Integer. Sample rate.
        return: 3D array, features [bands; frames; channels]
        """
        wnd_len = int(self.params['win_length_seconds'] * smp_rate)
        hop_len = int(self.params['hop_length_seconds'] * smp_rate)
        # Extract features, Mel Frequency Cepstral Coefficients
        wnd = self._window(wtype=self.params['window'], smp_size=wnd_len)

        if self.vad_type == 'energy':
            min_len = wnd_len * hop_len * 0.75  # about 2 sec
            times = librosa.effects.split(x, top_db=30)
            x = self._apply_vad(x, times, min=min_len)
        elif self.vad_type == 'webrtc':
            raise NotImplemented

        # calculate static filter bank coefficients
        stft = np.abs(librosa.stft(x + self.EPS,
                                   n_fft=self.params['n_fft'],
                                   win_length=wnd_len,
                                   hop_length=hop_len,
                                   window=wnd)) ** 2

        mel_basis = librosa.filters.mel(sr=smp_rate,
                                        n_fft=self.params['n_fft'],
                                        n_mels=self.params['bands'],
                                        fmin=self.params['fmin'],
                                        fmax=self.params['fmax'])
        mel_spec = np.dot(mel_basis, stft)
        logmel = librosa.amplitude_to_db(mel_spec)
        elogmel = logmel[:, :, np.newaxis]
        sh = elogmel.shape
        # consider delta features as the 2nd channel
        if self.params['include_delta']:
            # append delta features as an additional channel: [bands; frames; channels]
            buf = np.zeros(sh)
            elogmel = np.concatenate((elogmel, buf), axis=2)
            elogmel[:, :, 1] = librosa.feature.delta(elogmel[:, :, 0], **self.params['delta'])
        if self.params['include_acceleration']:
            buf = np.zeros(sh)
            elogmel = np.concatenate((elogmel, buf), axis=2)
            elogmel[:, :, 2] = librosa.feature.delta(elogmel[:, :, 1], **self.params['acceleration'])
        return elogmel

    def _apply_vad(self, x, times, min):
        dif = times[:, 1] - times[:, 0]
        times = times[dif > min]
        buf = []
        for t in times:
            buf.append(x[t[0]:t[1]])
        buf = np.hstack(buf)
        return buf


class STFTExtractor(BaseFeatureExtractor):
    def __init__(self, *args, **kwargs):
        super(STFTExtractor, self).__init__(*args, **kwargs)
        self.params = kwargs.get('params')
        if self.params is None:
            self.params = FBANK_DEFAULT

    def extract(self, x, smp_rate):
        """
        x: ndarray. 1D numpy array.
        smp_rate: Integer. Sample rate.
        return: 3D array, features [bands; frames; channels]
        """
        wnd_len = int(self.params['win_length_seconds'] * smp_rate)
        hop_len = int(self.params['hop_length_seconds'] * smp_rate)
        # Extract features, Mel Frequency Cepstral Coefficients
        wnd = self._window(wtype=self.params['window'], smp_size=wnd_len)
        # calculate static mfss coefficients
        stft = np.abs(librosa.stft(x + self.EPS,
                                   n_fft=self.params['n_fft'],
                                   win_length=wnd_len,
                                   hop_length=hop_len,
                                   window=wnd)) ** 2
        # estft = np.expand_dims(stft, axis=2)
        estft = stft[:, :, np.newaxis]
        sh = estft.shape
        # consider delta features as the 2nd "image" channel
        if self.params['include_delta']:
            # append delta features as an additional channel: [bands; frames; channels]
            buf = np.zeros(sh)
            estft = np.concatenate((estft, buf), axis=2)
            estft[:, :, 1] = librosa.feature.delta(estft[:, :, 0])
        if self.params['include_acceleration']:
            buf = np.zeros(sh)
            estft = np.concatenate((estft, buf), axis=2)
            estft[:, :, 2] = librosa.feature.delta(estft[:, :, 1])
        return estft


class VADExtractor(BaseFeatureExtractor):
    """
    Log-Mel filter bank feature extractor, steps:
    1. Frame the signal into short frames.
    2. For each frame calculate the periodogram estimate of the power spectrum.
    3. Apply the mel filterbank to the power spectra, sum the energy in each filter.
    4. Take the logarithm of all filterbank energies.
    """

    def __init__(self, *args, **kwargs):
        super(VADExtractor, self).__init__(*args, **kwargs)
        self.params = kwargs.get('params')
        if not self.params:
            self.params = VAD_DEFAULT
        self.vad_type = self.params['vad_type']

    def extract(self, x, smp_rate):
        """
        x: ndarray. 1D numpy array.
        smp_rate: Integer. Sample rate.
        return: 3D array, features [bands; frames; channels]
        """
        min_len = int(self.params['min_len'] * smp_rate)
        if self.vad_type == 'energy':
            level = self.params['energy_lvl']
            times = self._energy_vad(x, min=min_len, lvl=level)
            return times if len(times) else None
        elif self.vad_type == 'webrtc':
            raise NotImplemented

    @staticmethod
    def _energy_vad(x, min, lvl):
        times = librosa.effects.split(x, top_db=lvl)
        dif = times[:, 1] - times[:, 0]
        times = times[dif > min]
        return times