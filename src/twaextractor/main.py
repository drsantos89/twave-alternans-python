"""Main module for twa-extractor."""
import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pydantic
import wfdb
from scipy.signal import butter, sosfiltfilt


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
    signal_raw: np.ndarray[Any, Any] | None = None
    fs: int | None = None
    num_signals: int | None = None

    def extract(self) -> None:
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

    def _load(self) -> None:
        record = wfdb.rdrecord(self.path)
        self.signal_raw = record.p_signal
        self.fs = record.fs
        self.num_signals = self.signal_raw.shape[1]

    def _filter(self) -> None:
        """Filter ECG signal."""
        lowcut = 0.5
        highcut = 40
        order = 3
        sos = butter(
            order, [lowcut, highcut], btype="bandpass", fs=self.fs, output="sos"
        )
        self.signal_filt = sosfiltfilt(sos, self.signal_raw)

    def _detect_beats(self) -> None:
        """Detect beats in ECG signal."""
        self.qrs = wfdb.ecg.gqrs_detect(
            signal=self.signal_filt[:, 0], sampling_rate=self.fs
        )["qrs"]

    def _build_beat_matrix(self) -> None:
        """Build beat matrix from ECG signal."""
        beat_matrix = np.zeros((len(self.qrs), 2 * self.fs))
        for i, beat in enumerate(self.qrs):
            beat_matrix[i, :] = self.signal_filt[beat - self.fs : beat + self.fs, 0]

    def _correct_baseline(self) -> None:
        """Correct baseline in ECG signal."""
        signal_corrected = wfdb.ecg.correct_baseline_wander(
            signal=self.signal_filt, sampling_rate=self.fs, method="polynomial", order=3
        )
