"""Main module for twa-extractor."""
import logging

import matplotlib.pyplot as plt
import numpy as np
import wfdb
from scipy.signal import butter, filtfilt


class TWAExtractor(pydantic.BaseModel):
    """Extract T-wave alternans from ECG signals.

    Parameters
    ----------
    signal : array_like
        ECG signal.
    fs : int
        Sampling frequency.
    """

    path: str
    fs: int

    def extract(self):
        """Extract TWA from ECG signal.

        Returns
        -------
        twa : array_like
            TWA signal.
        """
        logging.info("loading signal ...")
        self._load()
        logging.info("filtering signal ...")
        self._filter()
        logging.info("detecting beats ...")
        self._detect_beats()
        logging.info("building beat matrix ...")
        self._build_beat_matrix()
        logging.info("correcting baseline ...")
        self._correct_baseline()
        logging.info("replacing low correlation beats ...")
        self._replace_low_correlation_beats()
        logging.info(f"abnormal beat percentage :, np.sum(b, axis=0)/len(qrs)*100")
        logging.info("building t wave vector ...")
        self._build_t_wave_vector()
        logging.info("calculating K-score ...")
        self._calculate_k_score()

    def _load(self):
        record = wfdb.rdrecord(signal_path)
        signal_raw = record.p_signal
        Fs = record.fs
        num_signals = signal_raw.shape[1]

    def _filter(self):
        """Filter ECG signal."""
        lowcut = 0.5
        highcut = 40
        order = 3
        sos = butter(order, [lowcut, highcut], btype="bandpass", fs=Fs, output="sos")
        signal_filt = sosfiltfilt(sos, signal_raw)

    def _detect_beats(self):
        """Detect beats in ECG signal."""
        qrs = ecg.gqrs_detect(signal=signal_filt[:, 0], sampling_rate=Fs)["qrs"]

    def _build_beat_matrix(self):
        """Build beat matrix from ECG signal."""
        beat_matrix = np.zeros((len(qrs), 2 * Fs))
        for i, beat in enumerate(qrs):
            beat_matrix[i, :] = signal_filt[beat - Fs : beat + Fs, 0]

    def _correct_baseline(self):
        """Correct baseline in ECG signal."""
        signal_corrected = ecg.correct_baseline_wander(
            signal=signal_filt, sampling_rate=Fs, method="polynomial", order=3
        )
