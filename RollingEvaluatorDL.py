from typing import Any, Dict, List, Sequence, Tuple, Callable
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from .FactorAndDmTest import ForecastUtils
from .ValueAtRisk import ValueAtRisk
from DataAndRealizedKernel.PDFmodels import PDFModels
from FeatureCreation.Features import vol_features


class _SeqDataset(Dataset):
    """
    Simple (X_seq, y) dataset for PyTorch DataLoader.

    Parameters
    ----------
    X : np.ndarray
        Input sequence tensor of shape (n_seq, lookback, n_features).
    y : np.ndarray
        Target tensor of shape (n_seq, n_horizons, 1).

    Attributes
    ----------
    X : torch.Tensor
        Stored inputs as float tensor.
    y : torch.Tensor
        Stored targets as float tensor.
    """

    X: torch.Tensor
    y: torch.Tensor

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int: return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]: return self.X[idx], self.y[idx]


def _make_sequences(X: np.ndarray, y: np.ndarray, lookback: int, horizons: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (n_seq, lookback, n_features) X_seq and (n_seq, n_horizons, 1) y_seq.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (N, n_features).
    y : np.ndarray
        Target vector of length N (single target scaled to 1-D).
    lookback : int
        Number of past steps per sequence.
    horizons : sequence of int
        Forecast horizons (positive integers).

    Returns
    -------
    (np.ndarray, np.ndarray)
        Tuple `(X_seq, y_seq)` where:
          - X_seq has shape (n_seq, lookback, n_features)
          - y_seq has shape (n_seq, n_horizons, 1)
    """
    Xs: List[np.ndarray] = []; ys: List[List[float]] = []
    max_h = max(horizons)
    for t in range(lookback, len(X) - max_h + 1):
        Xs.append(X[t - lookback:t])
        ys.append([y[t + h - 1] for h in horizons])
    X_seq = np.asarray(Xs, dtype=np.float32)
    y_arr = np.asarray(ys, dtype=np.float32)
    y_seq = y_arr[..., np.newaxis]
    return X_seq, y_seq


class DeepLearningEvaluator:
    """
    Rolling window walk-forward evaluator for deep learning volatility models.

    This evaluator:
      1) Creates engineered features via `vol_features`.
      2) Scales inputs/targets to stabilize training.
      3) Trains each model on an initial train/validation split.
      4) Performs rolling refits and multi-horizon predictions.
      5) Computes metrics (RMSE/MAE) on realized kernel (factor-adjusted).
      6) Computes VaR forecasts, hit sequences, and backtests.

    Parameters
    ----------
    log_ret : np.ndarray
        Raw log returns (1-D).
    kernel : np.ndarray
        Realized kernel series (1-D).
    sentiment : np.ndarray
        Sentiment categorical array aligned with returns.
    window : int
        Rolling window length in sequences.
    lookback : int
        Lookback length used to form sequences.
    ma_windows : Sequence[int]
        Moving average window sizes for feature engineering.
    models : Sequence[Tuple[str, Callable[..., torch.nn.Module] | torch.nn.Module]]
        Iterable of (name, builder_or_instance). If builder is callable, it is
        called with `self` to build a model instance.
    batch_size : int
        Mini-batch size for training.
    base_epochs : int
        Epochs for initial (train+val) training per model.
    tune_epochs : int
        Epochs for rolling refit at each step.
    learning_rate : float
        Optimizer learning rate for Adam.
    confidence_level : float
        VaR confidence level (e.g., 0.99 for 1% left tail).
    date_index : Sequence[Any]
        Date index aligned with raw series for reporting/plots.
    horizons : Sequence[int]
        Forecast horizons (positive integers).
    val_split : float
        Fraction of the first window reserved for validation (0 < val_split < 1).

    Attributes
    ----------
    device : torch.device
        CUDA if available, else CPU.
    X_raw, y_raw : np.ndarray
        Engineered features and target (kernel) before scaling.
    X, y : np.ndarray
        Scaled features/target (y flattened to 1-D for sequencing).
    X_seq, y_seq : np.ndarray
        Sequenced inputs/targets for training and evaluation.
    X_train0, y_train0, X_val0, y_val0 : np.ndarray
        Initial train/validation splits in sequence space.
    models : Dict[str, torch.nn.Module]
        Registered models by name.
    _optimisers : Dict[str, torch.optim.Optimizer]
        Optimizers keyed by model name.
    param_counts : Dict[str, int]
        Trainable parameter counts by model.
    forecasts, VaR, hits : dict
        Rolling evaluation outputs per horizon and model.
    realised_kernel_adj, test_returns : dict
        Realized targets and returns aligned to predictions.
    test_dates : list
        Dates aligned with the start of each prediction step.
    validation_errors, training_errors : Dict[str, List[float]]
        Per-epoch MSE on original scale for initial training.
    """

    # Core data & config
    log_ret_raw: np.ndarray
    kernel_raw: np.ndarray
    sentiment: np.ndarray
    window_len: int
    lookback: int
    ma_windows: Sequence[int]
    horizons: Sequence[int]
    batch_size: int
    base_epochs: int
    tune_epochs: int
    learning_rate: float
    CL: float
    alpha: float
    device: torch.device
    date_index: Sequence[Any]

    # Feature engineering and scaling
    X_raw: np.ndarray
    y_raw: np.ndarray
    offset: int
    X_scaler: StandardScaler
    y_scaler: StandardScaler
    X: np.ndarray
    y: np.ndarray
    X_seq: np.ndarray
    y_seq: np.ndarray

    # Initial split
    X_train0: np.ndarray
    y_train0: np.ndarray
    X_val0: np.ndarray
    y_val0: np.ndarray

    # Risk and utilities
    risk: ValueAtRisk

    # Model registry
    models: Dict[str, torch.nn.Module]
    _optimisers: Dict[str, torch.optim.Optimizer]
    param_counts: Dict[str, int]

    # Outputs
    forecasts: Dict[int, Dict[str, List[float]]]
    VaR: Dict[int, Dict[str, List[float]]]
    hits: Dict[int, Dict[str, List[int]]]
    realised_kernel_adj: Dict[int, List[float]]
    test_returns: Dict[int, List[float]]
    test_dates: List[Any]
    validation_errors: Dict[str, List[float]]
    training_errors: Dict[str, List[float]]

    def __init__(self, log_ret: np.ndarray, kernel: np.ndarray, sentiment: np.ndarray, window: int, lookback: int, ma_windows: Sequence[int], models: Sequence[Tuple[str, Callable[..., torch.nn.Module] | torch.nn.Module]], batch_size: int, base_epochs: int, tune_epochs: int, learning_rate: float, confidence_level: float, date_index: Sequence[Any], horizons: Sequence[int], val_split: float) -> None:
        self.log_ret_raw = log_ret
        self.kernel_raw = kernel
        self.sentiment = sentiment
        self.window_len = window
        self.lookback = lookback
        self.ma_windows = ma_windows
        self.horizons = horizons
        self.batch_size = batch_size
        self.base_epochs = base_epochs
        self.tune_epochs = tune_epochs
        self.learning_rate = learning_rate
        self.param_counts = {}
        self.CL = confidence_level
        self.alpha = 1.0 - confidence_level
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.risk = ValueAtRisk(PDFModels())
        self.date_index = date_index

        self.X_raw, self.y_raw = vol_features(self.log_ret_raw, self.kernel_raw, self.sentiment, self.lookback, self.ma_windows, horizons)
        self.offset = len(self.log_ret_raw) - len(self.y_raw)

        self.X_scaler = StandardScaler().fit(self.X_raw)
        self.y_scaler = StandardScaler().fit(self.y_raw.reshape(-1, 1))
        self.X = self.X_scaler.transform(self.X_raw)
        self.y = self.y_scaler.transform(self.y_raw.reshape(-1, 1)).ravel()

        self.X_seq, self.y_seq = _make_sequences(self.X, self.y, self.lookback, self.horizons)
        self._split_initial_train_val(val_split)

        self.forecasts = {h: defaultdict(list) for h in self.horizons}
        self.VaR = {h: defaultdict(list) for h in self.horizons}
        self.hits = {h: defaultdict(list) for h in self.horizons}
        self.realised_kernel_adj = {h: [] for h in self.horizons}
        self.test_returns = {h: [] for h in self.horizons}
        self.test_dates = []
        self.models = {}
        self._optimisers = {}

        for name, builder in models:
            m = builder(self) if callable(builder) else builder
            self.register_model(name, m)

        self.validation_errors = {name: [] for name in self.models}
        self.training_errors = {name: [] for name in self.models}

    def _split_initial_train_val(self, val_split: float) -> None:
        """
        Split the first window into train/validation sequences.

        Parameters
        ----------
        val_split : float
            Fraction of the first window dedicated to validation (0 < val_split < 1).
        """
        n0 = self.window_len
        val_n = int(n0 * val_split)
        train_n = n0 - val_n
        self.X_train0 = self.X_seq[:train_n]
        self.y_train0 = self.y_seq[:train_n]
        self.X_val0 = self.X_seq[train_n:n0]
        self.y_val0 = self.y_seq[train_n:n0]

    def register_model(self, name: str, model: torch.nn.Module) -> None:
        """
        Register a model and its optimizer.

        Parameters
        ----------
        name : str
            Model key.
        model : torch.nn.Module
            Instantiated PyTorch model.
        """
        self.models[name] = model
        self._optimisers[name] = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        model.to(self.device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.param_counts[name] = int(n_params)

    def _fit(self, name: str, X: np.ndarray, y: np.ndarray, epochs: int) -> None:
        """
        Train a model for a fixed number of epochs on provided sequences.

        Parameters
        ----------
        name : str
            Model key.
        X : np.ndarray
            Input sequences (N, lookback, n_features).
        y : np.ndarray
            Targets (N, n_horizons, 1).
        epochs : int
            Number of epochs.
        """
        model = self.models[name]
        opt = self._optimisers[name]
        loader = DataLoader(_SeqDataset(X, y), batch_size=self.batch_size, shuffle=True)
        model.train()
        for _ in range(epochs):
            for xb, yb in loader:
                xb = xb.to(self.device); yb = yb.to(self.device)
                opt.zero_grad()
                loss = F.mse_loss(model(xb), yb)
                loss.backward()
                opt.step()

    def _fit_with_validation(self, name: str, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, epochs: int) -> None:
        """
        Train with per-epoch train/validation MSE tracked on original scale.

        Parameters
        ----------
        name : str
            Model key.
        X_train, y_train, X_val, y_val : np.ndarray
            Train/validation sequence tensors.
        epochs : int
            Number of epochs.
        """
        model = self.models[name]
        opt = self._optimisers[name]
        train_loader = DataLoader(_SeqDataset(X_train, y_train), batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(_SeqDataset(X_val, y_val), batch_size=self.batch_size, shuffle=False)

        for _ in range(epochs):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                loss = F.mse_loss(model(xb), yb)
                loss.backward()
                opt.step()

            model.eval()
            preds_t: List[float] = []; true_t: List[float] = []
            with torch.no_grad():
                for xb, yb in train_loader:
                    xb = xb.to(self.device)
                    ps = model(xb).cpu().numpy().reshape(-1, 1)
                    ts = yb.cpu().numpy().reshape(-1, 1)
                    ps = self.y_scaler.inverse_transform(ps).ravel()
                    ts = self.y_scaler.inverse_transform(ts).ravel()
                    preds_t.extend(ps); true_t.extend(ts)
            train_mse = float(np.mean((np.array(preds_t) - np.array(true_t)) ** 2))
            self.training_errors[name].append(train_mse)

            preds_v: List[float] = []; true_v: List[float] = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(self.device)
                    ps = model(xb).cpu().numpy().reshape(-1, 1)
                    ts = yb.cpu().numpy().reshape(-1, 1)
                    ps = self.y_scaler.inverse_transform(ps).ravel()
                    ts = self.y_scaler.inverse_transform(ts).ravel()
                    preds_v.extend(ps); true_v.extend(ts)
            val_mse = float(np.mean((np.array(preds_v) - np.array(true_v)) ** 2))
            self.validation_errors[name].append(val_mse)

    def _predict(self, name: str, X: np.ndarray) -> Dict[int, float]:
        """
        Predict multi-horizon targets for a single sequence and invert scaling.

        Parameters
        ----------
        name : str
            Model key.
        X : np.ndarray
            Input sequence of shape (1, lookback, n_features).

        Returns
        -------
        dict[int, float]
            Mapping horizon -> predicted realized kernel on original scale.
        """
        model = self.models[name]
        model.eval()
        with torch.no_grad():
            y_hat_scaled = model(torch.from_numpy(X).to(self.device)).cpu().numpy()
        batch, n_h, _ = y_hat_scaled.shape
        y_hat_unflat = self.y_scaler.inverse_transform(y_hat_scaled.reshape(batch * n_h, 1))
        y_hat = y_hat_unflat.reshape(batch, n_h)
        y0 = y_hat[0]
        return {h: float(y0[i]) for i, h in enumerate(self.horizons)}

    def run(self) -> Dict[str, Any]:
        """
        Execute initial training, rolling refits, forecasting, VaR, and metrics.

        Returns
        -------
        dict
            Contains first-window param counts, forecasts, VaR, hits, metrics,
            backtest results, realized kernel, test returns, dates, and errors.
        """
        for name in self.models:
            self._fit_with_validation(name, self.X_train0, self.y_train0, self.X_val0, self.y_val0, self.base_epochs)

        for t in range(self.window_len, len(self.X_seq)):
            print(f"\nStep {t - self.window_len + 1}/{len(self.X_seq) - self.window_len} (window index {t})")
            start = t - self.window_len

            for name in self.models:
                self._fit(name, self.X_seq[start:t], self.y_seq[start:t], self.tune_epochs)

            raw_idx = self.offset + self.lookback + t
            self.test_dates.append(self.date_index[raw_idx])

            seq_for_pred = self.X_seq[t:t + 1]

            for i, h in enumerate(self.horizons):
                target_idx = raw_idx + h - 1
                if target_idx >= len(self.kernel_raw):
                    continue
                factor = ForecastUtils.calculate_factor(self.log_ret_raw[:raw_idx], self.kernel_raw[:raw_idx])
                rk_adj = float(self.kernel_raw[target_idx] * factor)
                self.realised_kernel_adj[h].append(rk_adj)
                self.test_returns[h].append(float(self.log_ret_raw[target_idx]))

            for name in self.models:
                forecast_dict = self._predict(name, seq_for_pred)
                for i, h in enumerate(self.horizons):
                    target_idx = raw_idx + h - 1
                    if target_idx >= len(self.kernel_raw):
                        continue
                    self.forecasts[h][name].append(forecast_dict[h])
                    var_next = float(self.risk.calculate_var_ML(forecast_dict[h], self.CL, self.log_ret_raw[:raw_idx], self.kernel_raw[:raw_idx]))
                    self.VaR[h][name].append(var_next)
                    self.hits[h][name].append(int(self.test_returns[h][-1] < var_next))

        metrics = {h: {name: {"rmse": float(np.sqrt(mean_squared_error(self.realised_kernel_adj[h], self.forecasts[h][name]))), "mae": float(mean_absolute_error(self.realised_kernel_adj[h], self.forecasts[h][name]))} for name in self.forecasts[h]} for h in self.horizons}
        backtests = {h: {name: self.risk.backtest_var(self.hits[h][name], self.alpha) for name in self.hits[h]} for h in self.horizons}

        return {
            "first_window_results": self.param_counts,
            "forecasts": self.forecasts,
            "var_results": self.VaR,
            "hit_var_results": self.hits,
            "metrics": metrics,
            "backtest_results": backtests,
            "realized_kernel": self.realised_kernel_adj,
            "test_log_returns": self.test_returns,
            "test_dates": self.test_dates,
            "validation_errors": self.validation_errors,
            "training_errors": self.training_errors,
        }
