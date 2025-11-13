import numpy as np
import pandas as pd
import random

data = pd.read_csv(filename)
data.index = pd.to_datetime(data["Date"])
data = data.drop("Date", axis=1)


# Feature engineering class


class VolatilityFeatureBuilder:
    @staticmethod
    def build_volatility_features(
        df,
        returns_col: str = "log returns",
        rk_col: str = "Kernel",
        max_lag: int = 5,
        rolling_windows=(5, 22),
        dropna: bool = True,
    ):
        """
        Feature engineering for volatility forecasting using log returns and realized kernel.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with a DatetimeIndex, a log-return column, and a realized-kernel column.
        returns_col : str, default "log_return"
            Name of the log return column in df.
        rk_col : str, default "rk"
            Name of the realized kernel (integrated volatility proxy) column in df.
        max_lag : int, default 5
            Number of lags to create for returns and realized kernel.
        rolling_windows : iterable of int, default (5, 22)
            Window lengths (in days) for rolling features (mean/std).
        dropna : bool, default True
            If True, drop rows with NaNs created by lags/rolling windows.

        Returns
        -------
        features : pd.DataFrame
            DataFrame with engineered features and the target column 'y'.
        """
        df = df.copy()

        # Volatility-type transforms of returns
        df[returns_col + "_sq"] = df[returns_col] ** 2
        df[returns_col + "_abs"] = df[returns_col].abs()

        # Lags
        for lag in range(1, max_lag + 1):
            # Lags of returns
            df[f"{returns_col}_lag{lag}"] = df[returns_col].shift(lag)
            df[f"{returns_col}_sq_lag{lag}"] = df[returns_col + "_sq"].shift(lag)
            df[f"{returns_col}_abs_lag{lag}"] = df[returns_col + "_abs"].shift(lag)

            # Lags of realized kernel
            df[f"{rk_col}_lag{lag}"] = df[rk_col].shift(lag)
            df[f"log_{rk_col}_lag{lag}"] = df[rk_col].shift(lag)

        # Rolling features
        for window in rolling_windows:
            # Rolling mean/std of |returns|
            df[f"{returns_col}_abs_rollmean_{window}"] = (
                df[returns_col + "_abs"].rolling(window=window).mean()
            )
            df[f"{returns_col}_abs_rollstd_{window}"] = (
                df[returns_col + "_abs"].rolling(window=window).std()
            )

            # Rolling mean/std of realized kernel
            df[f"{rk_col}_rollmean_{window}"] = df[rk_col].rolling(window=window).mean()
            df[f"{rk_col}_rollstd_{window}"] = df[rk_col].rolling(window=window).std()

            df[f"log_{rk_col}_rollmean_{window}"] = df[rk_col].rolling(window=window).mean()
            df[f"log_{rk_col}_rollstd_{window}"] = df[rk_col].rolling(window=window).std()

        # Time Features
        df["dow"] = df.index.dayofweek
        df["month"] = df.index.month
        df["year"] = df.index.year

        # Cyclical encoding
        df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)

        # Target
        df["y"] = df[rk_col]

        if dropna:
            df = df.dropna()

        return df


# build features 
features = VolatilityFeatureBuilder.build_volatility_features(data,returns_col="log returns",rk_col="Kernel",max_lag=5,rolling_windows=(5, 22),dropna=True,)

nb_train = 1300
X_train = features.iloc[:, :-1][:nb_train]
y_train = features["y"][:nb_train]
X_test = features.iloc[:, :-1][nb_train:]
y_test = features["y"][nb_train:]


# Random forest from scratch

class TimeSeriesRandomForest:
    @staticmethod
    def draw_block_bootstrap(X_train, y_train, block_size):
        """
        Moving block bootstrap for time-series data.
        """
        n = len(X_train)
        n_blocks = int(np.ceil(n / block_size))

        bootstrap_indices = []

        for _ in range(n_blocks):
            # choose start index for the block
            start = np.random.randint(0, n - block_size + 1)
            block = list(range(start, start + block_size))
            bootstrap_indices.extend(block)

        # truncate to exactly n observations
        bootstrap_indices = np.array(bootstrap_indices[:n])

        # OOB indices: those never sampled
        in_bag_mask = np.zeros(n, dtype=bool)
        in_bag_mask[bootstrap_indices] = True
        oob_indices = np.where(~in_bag_mask)[0]

        X_bootstrap = X_train.iloc[bootstrap_indices].values
        y_bootstrap = y_train.iloc[bootstrap_indices].values

        X_oob = X_train.iloc[oob_indices].values
        y_oob = y_train.iloc[oob_indices].values

        return X_bootstrap, y_bootstrap, X_oob, y_oob

    @staticmethod
    def draw_bootstrap(X_train, y_train):
        bootstrap_indices = list(np.random.choice(range(len(X_train)), len(X_train), replace=True))
        oob_indices = [i for i in range(len(X_train)) if i not in bootstrap_indices]
        X_bootstrap = X_train.iloc[bootstrap_indices].values
        y_bootstrap = y_train[bootstrap_indices]
        X_oob = X_train.iloc[oob_indices].values
        y_oob = y_train[oob_indices]
        return X_bootstrap, y_bootstrap, X_oob, y_oob

    @staticmethod
    def oob_score(tree, X_test, y_test):
        mis_label = 0
        for i in range(len(X_test)):
            pred = TimeSeriesRandomForest.predict_tree(tree, X_test[i])
            if pred != y_test[i]:
                mis_label += 1
        return mis_label / len(X_test)

    @staticmethod
    def entropy(p):
        if p == 0:
            return 0
        elif p == 1:
            return 0
        else:
            return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

    @staticmethod
    def information_gain(left_child, right_child):
        parent = left_child + right_child
        p_parent = parent.count(1) / len(parent) if len(parent) > 0 else 0
        p_left = left_child.count(1) / len(left_child) if len(left_child) > 0 else 0
        p_right = right_child.count(1) / len(right_child) if len(right_child) > 0 else 0
        IG_p = TimeSeriesRandomForest.entropy(p_parent)
        IG_l = TimeSeriesRandomForest.entropy(p_left)
        IG_r = TimeSeriesRandomForest.entropy(p_right)
        return (
            IG_p
            - len(left_child) / len(parent) * IG_l
            - len(right_child) / len(parent) * IG_r
        )

    @staticmethod
    def find_split_point(X_bootstrap, y_bootstrap, max_features):
        feature_ls = list()
        num_features = len(X_bootstrap[0])

        while len(feature_ls) <= max_features:
            feature_idx = random.sample(range(num_features), 1)
        if feature_idx not in feature_ls:
            feature_ls.extend(feature_idx)

        best_info_gain = -999
        node = None
        for feature_idx in feature_ls:
            for split_point in X_bootstrap[:, feature_idx]:
                left_child = {"X_bootstrap": [], "y_bootstrap": []}
                right_child = {"X_bootstrap": [], "y_bootstrap": []}

                # split children for continuous variables
                if type(split_point) in [int, float]:
                    for i, value in enumerate(X_bootstrap[:, feature_idx]):
                        if value <= split_point:
                            left_child["X_bootstrap"].append(X_bootstrap[i])
                            left_child["y_bootstrap"].append(y_bootstrap[i])
                        else:
                            right_child["X_bootstrap"].append(X_bootstrap[i])
                            right_child["y_bootstrap"].append(y_bootstrap[i])
                # split children for categoric variables
                else:
                    for i, value in enumerate(X_bootstrap[:, feature_idx]):
                        if value == split_point:
                            left_child["X_bootstrap"].append(X_bootstrap[i])
                            left_child["y_bootstrap"].append(y_bootstrap[i])
                        else:
                            right_child["X_bootstrap"].append(X_bootstrap[i])
                            right_child["y_bootstrap"].append(y_bootstrap[i])

                split_info_gain = TimeSeriesRandomForest.information_gain(
                    left_child["y_bootstrap"], right_child["y_bootstrap"]
                )
                if split_info_gain > best_info_gain:
                    best_info_gain = split_info_gain
                    left_child["X_bootstrap"] = np.array(left_child["X_bootstrap"])
                    right_child["X_bootstrap"] = np.array(right_child["X_bootstrap"])

                    node = {
                        "information_gain": split_info_gain,
                        "left_child": left_child,
                        "right_child": right_child,
                        "split_point": split_point,
                        "feature_idx": feature_idx,
                    }

        return node

    @staticmethod
    def terminal_node(node):
        y_bootstrap = node["y_bootstrap"]
        pred = max(y_bootstrap, key=y_bootstrap.count)
        return pred

    @staticmethod
    def split_node(node, max_features, min_samples_split, max_depth, depth):
        left_child = node["left_child"]
        right_child = node["right_child"]

        del node["left_child"]
        del node["right_child"]

        if len(left_child["y_bootstrap"]) == 0 or len(right_child["y_bootstrap"]) == 0:
            empty_child = {
                "y_bootstrap": left_child["y_bootstrap"]
                + right_child["y_bootstrap"]
            }
            node["left_split"] = TimeSeriesRandomForest.terminal_node(empty_child)
            node["right_split"] = TimeSeriesRandomForest.terminal_node(empty_child)
            return

        if depth >= max_depth:
            node["left_split"] = TimeSeriesRandomForest.terminal_node(left_child)
            node["right_split"] = TimeSeriesRandomForest.terminal_node(right_child)
            return node

        if len(left_child["X_bootstrap"]) <= min_samples_split:
            node["left_split"] = node["right_split"] = TimeSeriesRandomForest.terminal_node(
                left_child
            )
        else:
            node["left_split"] = TimeSeriesRandomForest.find_split_point(
                left_child["X_bootstrap"], left_child["y_bootstrap"], max_features
            )
            TimeSeriesRandomForest.split_node(
                node["left_split"], max_depth, min_samples_split, max_depth, depth + 1
            )
        if len(right_child["X_bootstrap"]) <= min_samples_split:
            node["right_split"] = node["left_split"] = TimeSeriesRandomForest.terminal_node(
                right_child
            )
        else:
            node["right_split"] = TimeSeriesRandomForest.find_split_point(
                right_child["X_bootstrap"], right_child["y_bootstrap"], max_features
            )
            TimeSeriesRandomForest.split_node(
                node["right_split"], max_features, min_samples_split, max_depth, depth + 1
            )

    @staticmethod
    def build_tree(X_bootstrap, y_bootstrap, max_depth, min_samples_split, max_features):
        root_node = TimeSeriesRandomForest.find_split_point(
            X_bootstrap, y_bootstrap, max_features
        )
        TimeSeriesRandomForest.split_node(
            root_node, max_features, min_samples_split, max_depth, 1
        )
        return root_node

    @staticmethod
    def random_forest(X_train,y_train,n_estimators,max_features,max_depth,min_samples_split,block_bootstrap):
        tree_ls = list()
        oob_ls = list()

        def select_bootstrap(X_train_y_train):
            if block_bootstrap:
                return TimeSeriesRandomForest.draw_block_bootstrap(X_train, y_train, block_size=22)
            else:
                return TimeSeriesRandomForest.draw_bootstrap(X_train, y_train)

        for i in range(n_estimators):
            X_bootstrap, y_bootstrap, X_oob, y_oob = select_bootstrap((X_train, y_train))
            tree = TimeSeriesRandomForest.build_tree(
                X_bootstrap, y_bootstrap, max_features, max_depth, min_samples_split
            )
            tree_ls.append(tree)
            oob_error = TimeSeriesRandomForest.oob_score(tree, X_oob, y_oob)
            oob_ls.append(oob_error)
        print("OOB estimate: {:.2f}".format(np.mean(oob_ls)))
        return tree_ls

    @staticmethod
    def predict_tree(tree, X_test):
        feature_idx = tree["feature_idx"]

        if X_test[feature_idx] <= tree["split_point"]:
            if type(tree["left_split"]) == dict:
                return TimeSeriesRandomForest.predict_tree(tree["left_split"], X_test)
            else:
                value = tree["left_split"]
                return value
        else:
            if type(tree["right_split"]) == dict:
                return TimeSeriesRandomForest.predict_tree(tree["right_split"], X_test)
            else:
                return tree["right_split"]

    @staticmethod
    def predict_rf(tree_ls, X_test):
        pred_ls = list()
        for i in range(len(X_test)):
            ensemble_preds = [
                TimeSeriesRandomForest.predict_tree(tree, X_test.values[i])
                for tree in tree_ls
            ]
            final_pred = max(ensemble_preds, key=ensemble_preds.count)
            pred_ls.append(final_pred)
        return np.array(pred_ls)

    @staticmethod
    def main():
        n_estimators = 100
        max_features = 3
        max_depth = 10
        min_samples_split = 2

        block_bootstrap_model = TimeSeriesRandomForest.random_forest(
            X_train,
            y_train,
            n_estimators=100,
            max_features=3,
            max_depth=10,
            min_samples_split=2,
            block_bootstrap=True,
        )
        bootstrap_model = TimeSeriesRandomForest.random_forest(
            X_train,
            y_train,
            n_estimators=100,
            max_features=3,
            max_depth=10,
            min_samples_split=2,
            block_bootstrap=False,
        )

        block_bootstrap_model_preds = TimeSeriesRandomForest.predict_rf(
            block_bootstrap_model, X_test
        )
        bootstrap_model_preds = TimeSeriesRandomForest.predict_rf(
            bootstrap_model, X_train
        )

        rmse_block_bootstrap = (1 / len(block_bootstrap_model_preds)) * sum(
            (y_test - block_bootstrap_model_preds) ** 2
        )
        rmse_bootstrap = (1 / len(bootstrap_model_preds)) * sum(
            (y_test - bootstrap_model_preds) ** 2
        )

        print(f"Testing accuracy with block bootstrapping: {rmse_block_bootstrap}")
        print(f"Testing accuracy without block bootstrapping: {rmse_bootstrap}")


if __name__ == "__main__":
    TimeSeriesRandomForest.main()
