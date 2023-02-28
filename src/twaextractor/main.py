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
    t_wave_vector: np.ndarray[Any, Any] | None = None
    k_score: np.ndarray[Any, Any] | None = None

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
        if not isinstance(self.beat_matrix_corrected, np.ndarray):
            raise TypeError("beat_matrix_corrected must be a numpy array")

        self.beat_median = np.median(self.beat_matrix_corrected, axis=1)
        self.beat_median_odd = np.median(self.beat_matrix_corrected[:, 1::2, :], axis=1)
        self.beat_median_even = np.median(
            self.beat_matrix_corrected[:, 0::2, :], axis=1
        )

    def _replace_low_correlation_beats(self) -> None:
        """Replace low correlation beats in ECG signal."""
        if not isinstance(self.qrs, np.ndarray):
            raise TypeError("qrs must be a numpy array")
        if not isinstance(self.n_sig, int):
            raise TypeError("n_sig must be an integer")
        if not isinstance(self.beat_matrix_corrected, np.ndarray):
            raise TypeError("beat_matrix_corrected must be a numpy array")
        if not isinstance(self.beat_median_odd, np.ndarray):
            raise TypeError("beat_median_odd must be a numpy array")
        if not isinstance(self.beat_median_even, np.ndarray):
            raise TypeError("beat_median_even must be a numpy array")

        # placeholder for beat replacement boolean
        self.beat_corrected_bool = np.zeros((self.n_sig, len(self.qrs)))

        # replace odd beats with low correlation
        for i in range(1, len(self.qrs) - 1, 2):
            beat = self.beat_matrix_corrected[:, i, :]
            for c in range(self.n_sig):
                if np.corrcoef(beat[c, :], self.beat_median_odd[c, :])[0, 1] < 0.9:
                    self.beat_corrected_bool[c, i] = 1
                    self.beat_matrix_corrected[c, i, :] = self.beat_median_odd[c, :]

        # replace even beats with low correlation
        for i in range(0, len(self.qrs) - 1, 2):
            beat = self.beat_matrix_corrected[:, i, :]
            for c in range(self.n_sig):
                if np.corrcoef(beat[c, :], self.beat_median_even[c, :])[0, 1] < 0.9:
                    self.beat_corrected_bool[c, i] = 1
                    self.beat_matrix_corrected[c, i, :] = self.beat_median_even[c, :]

    def _build_t_wave_vector(self) -> None:
        """Build T-wave vector from ECG signal."""
        if self.fs is None:
            raise ValueError("fs must be defined")
        if not isinstance(self.qrs, np.ndarray):
            raise TypeError("qrs must be a numpy array")
        if not isinstance(self.beat_matrix_corrected, np.ndarray):
            raise TypeError("beat_matrix_corrected must be a numpy array")

        self.t_wave_vector = np.zeros((self.n_sig, len(self.qrs), int(0.3 * self.fs)))
        for i in range(len(self.qrs)):
            self.t_wave_vector[:, i, :] = self.beat_matrix_corrected[
                :, i, int(0.3 * self.fs) : int(0.5 * self.fs)
            ]

    def _calculate_k_score(self) -> None:
        """Calculate K-score (spectral method) from ECG signal."""
        if not isinstance(self.n_sig, int):
            raise ValueError("n_sig must be defined")
        if not isinstance(self.t_wave_vector, np.ndarray):
            raise TypeError("t_wave_vector must be a numpy array")

        # placeholder for K-score results
        self.k_score = np.zeros(self.n_sig)

        # calculate K-score for each signal
        for i in range(self.n_sig):
            self.k_score[i] = np.std(self.t_wave_vector[:, i]) / np.mean(
                self.t_wave_vector[:, i]
            )
