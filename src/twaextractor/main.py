"""Main module for twa-extractor."""
import logging
from typing import Any

import numpy as np
import pydantic
import wfdb
from scipy.signal import butter, sosfiltfilt
from wfdb import processing


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
    n_sig: int | None = None
    sig_name: list[str] | None = None
    sig_len: int | None = None
    signal_filt: np.ndarray[Any, Any] | None = None
    qrs: np.ndarray[Any, Any] | None = None
    beat_matrix: np.ndarray[Any, Any] | None = None
    beat_matrix_corrected: np.ndarray[Any, Any] | None = None
    beat_median: np.ndarray[Any, Any] | None = None
    beat_median_odd: np.ndarray[Any, Any] | None = None
    beat_median_even: np.ndarray[Any, Any] | None = None
    beat_corrected_bool: np.ndarray[Any, Any] | None = None

    class Config:
        """Pydantic config class."""

        arbitrary_types_allowed = True

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

        logging.info("calculating beat medians ...")
        self._calc_beat_medians()

        logging.info("replacing low correlation beats ...")
        self._replace_low_correlation_beats()

        logging.info("building t wave vector ...")
        self._build_t_wave_vector()

        logging.info("calculating K-score ...")
        self._calculate_k_score()

    def _load(self) -> None:
        record = wfdb.rdrecord(self.path)
        self.signal_raw = record.p_signal.T
        self.fs = record.fs
        self.n_sig = record.n_sig
        self.sig_len = record.sig_len
        self.sig_name = record.sig_name

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
        if not isinstance(self.signal_filt, np.ndarray):
            raise TypeError("signal_filt must be a numpy array")
        self.qrs = processing.gqrs_detect(sig=self.signal_filt[0, :], fs=self.fs)

    def _build_beat_matrix(self) -> None:
        """Build beat matrix from ECG signal."""
        if self.fs is None:
            raise ValueError("fs must be defined")
        if not isinstance(self.qrs, np.ndarray):
            raise TypeError("qrs must be a numpy array")
        if not isinstance(self.signal_filt, np.ndarray):
            raise TypeError("signal_filt must be a numpy array")

        self.beat_matrix = np.zeros((self.n_sig, len(self.qrs), int(0.5 * self.fs)))
        for i, beat in enumerate(self.qrs):
            if beat - int(0.1 * self.fs) < 0:
                continue
            if beat + int(0.4 * self.fs) > self.sig_len:
                continue
            self.beat_matrix[:, i, :] = self.signal_filt[
                :, beat - int(0.1 * self.fs) : beat + int(0.4 * self.fs)
            ]

    def _correct_baseline(self) -> None:
        """Correct baseline in beat matrix."""
        baselines = np.expand_dims(np.mean(self.beat_matrix, axis=2), 2)
        self.beat_matrix_corrected = self.beat_matrix - baselines

    def _calc_beat_medians(self) -> None:
        """Calculate beat medians from beat matrix."""
        self.beat_median = np.median(self.beat_matrix_corrected, axis=0)
        self.beat_median_odd = self.beat_median[1::2]
        self.beat_median_even = self.beat_median[0::2]

    def _replace_low_correlation_beats(self) -> None:
        """Replace low correlation beats in ECG signal."""
        if not isinstance(self.qrs, np.ndarray):
            raise TypeError("qrs must be a numpy array")
        if not isinstance(self.beat_matrix_corrected, np.ndarray):
            raise TypeError("beat_matrix_corrected must be a numpy array")

        self.beat_corrected_bool = np.zeros(len(self.qrs))
        for i, beat in self.beat_matrix_corrected[1::2]:
            if np.corrcoef(beat, self.beat_median_odd)[0, 1] < 0.9:
                self.beat_corrected_bool[i] = 1
                self.beat_matrix_corrected[i, :] = self.beat_median_odd
        for i, beat in self.beat_matrix_corrected[0::2]:
            if np.corrcoef(beat, self.beat_median_even)[0, 1] < 0.9:
                self.beat_corrected_bool[i] = 1
                self.beat_matrix_corrected[i, :] = self.beat_median_even

    def _build_t_wave_vector(self) -> None:
        """Build T-wave vector from ECG signal."""
        if self.fs is None:
            raise ValueError("fs must be defined")
        if not isinstance(self.qrs, np.ndarray):
            raise TypeError("qrs must be a numpy array")
        if not isinstance(self.beat_matrix_corrected, np.ndarray):
            raise TypeError("beat_matrix_corrected must be a numpy array")

        self.t_wave_vector = np.zeros((len(self.qrs), self.num_signals))
        for i, beat in enumerate(self.beat_matrix_corrected):
            self.t_wave_vector[i, :] = beat[int(0.2 * self.fs) : int(0.4 * self.fs)]

    def _calculate_k_score(self) -> None:
        """Calculate K-score (spectral method) from ECG signal."""
        if self.num_signals is None:
            raise ValueError("num_signals must be defined")
        self.k_score = np.zeros(self.num_signals)
        for i in range(self.num_signals):
            self.k_score[i] = np.std(self.t_wave_vector[:, i]) / np.mean(
                self.t_wave_vector[:, i]
            )
