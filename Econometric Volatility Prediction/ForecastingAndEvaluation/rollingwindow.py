from typing import List, Tuple
import numpy as np

class rolling_window:
    """
    Rolling (sliding) window splitter for time-ordered data.

    Parameters
    ----------
    window : int, default 5
        Number of observations in each training window.
    horizon : int, default 1
        Number of observations in each test window (forecast horizon).
    period : int, default 1
        Step size (stride) to move the window forward.

    Attributes
    ----------
    window : int
        Training window length.
    horizon : int
        Test window length (forecast horizon).
    period : int
        Step size between consecutive windows.

    Notes
    -----
    - This splitter does not shuffle; it preserves temporal order.
    - Indices are 0-based and inclusive on the training side, exclusive on the
      right end as produced by `np.arange`.
    """

    window: int
    horizon: int
    period: int

    def __init__(self, window: int = 5, horizon: int = 1, period: int = 1) -> None:
        self.window = window
        self.horizon = horizon
        self.period = period

    def split(self, data: np.ndarray) -> List[Tuple[List[int], List[int]]]:
        """
        Generate rolling train/test index pairs.

        Parameters
        ----------
        data : np.ndarray
            Array-like with first dimension representing time; only its length is used.

        Returns
        -------
        list[tuple[list[int], list[int]]]
            A list where each item is (train_indices, test_indices).
        """
        data_length = data.shape[0]
        output_train: List[List[int]] = []
        output_test: List[List[int]] = []
        start = 0
        while start + self.window + self.horizon <= data_length:
            train_indices = list(np.arange(start, start + self.window))
            test_indices = list(np.arange(start + self.window, start + self.window + self.horizon))
            output_train.append(train_indices)
            output_test.append(test_indices)
            start += self.period
        index_output = [(train, test) for train, test in zip(output_train, output_test)]
        return index_output
