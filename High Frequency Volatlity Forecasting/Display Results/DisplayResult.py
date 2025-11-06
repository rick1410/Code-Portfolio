import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from .Plotter import  DMPlotter
import seaborn as sns
import os

class Plotter:
    @staticmethod
    def _get_model_name(obj):
        return getattr(obj, "model_name", str(obj))

    @staticmethod
    def _sanitize_filename(s: str) -> str:
        return "".join(c if c.isalnum() else "_" for c in s)

    def plot_forecasts_by_group(forecasts, realized_kernel, model_classes,group_name, dates, ticker, save_path=None):
        
        plt.figure(figsize=(12, 6))
        plt.plot(dates, realized_kernel, label="Realized Kernel",color="grey", linewidth=1.5, alpha=0.6)
        
        for model_class in model_classes:
            name = Plotter._get_model_name(model_class)
            if name in forecasts:
                plt.plot(dates, forecasts[name], label=name, linewidth=1)
        
        plt.title(f"{ticker}: Forecasts vs Realized Kernel\n{group_name}", pad=15)
        plt.xlabel("Time")
        plt.ylabel("Value")

        
        n_entries = len(model_classes) + 1  
        ncol = min(n_entries, 7)
        plt.legend(loc="upper center",bbox_to_anchor=(0.5, -0.15),ncol=ncol,frameon=False,fontsize="small")
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2)
        if save_path:
            plt.savefig(save_path, dpi=350, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    @staticmethod
    def plot_var_and_returns(var_results, log_returns, model_classes,group_name, dates, ticker, save_path=None):
        plt.figure(figsize=(14, 7))
        plt.plot(dates, log_returns, label="Out‐of‐Sample Returns",color="grey", linewidth=1.5,alpha=0.6)
        
        for model_class in model_classes:
            name = Plotter._get_model_name(model_class)
            if name in var_results:
                plt.plot(dates, var_results[name], label=f"{name} VaR",
                         linewidth=1)
        
        plt.title(f"{ticker}: One‐Day VaR vs Returns\n{group_name}", pad=15)
        plt.xlabel("Time"); plt.ylabel("Value")
        plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15),ncol=len(model_classes)+1, frameon=False)
        plt.tight_layout(); plt.subplots_adjust(bottom=0.2)
        if save_path:
            plt.savefig(save_path, dpi=350, bbox_inches="tight"); plt.close()
        else:
            plt.show()


class ResultDisplayer:
    def __init__(self, results, ticker, groups=None, out_root=None):
        """
        out_root: path to your High Frequency folder, e.g.
          r"...\High Frequency"
        """
        self.results  = results
        self.groups   = groups or {}
        self.ticker   = ticker
        self.out_root = out_root

    def _ensure_folder(self):
        folder = os.path.join(self.out_root, self.ticker, f"{self.ticker} Results")
        os.makedirs(folder, exist_ok=True)
        return folder

    def display_results(self):
        folder = self._ensure_folder()
        dates  = self.results.get("test_dates", None)

        for grp, title in [("group1", "Normal"),("group2", "Student_t"),("group3", "EGB2"),("group4", "AST")]:

            models = self.groups.get(grp, [])
            fname = f"{self.ticker}_{Plotter._sanitize_filename(title)}_forecasts.png"
            path = os.path.join(folder, fname) if folder else None
            Plotter.plot_forecasts_by_group(forecasts = self.results["forecasts"][1],realized_kernel= self.results["realized_kernel"][1],model_classes = models,group_name = f"Models with {title} distribution",dates = dates,ticker = self.ticker,save_path = path)
            
            # VaR plot
            fname = f"{self.ticker}_{Plotter._sanitize_filename(title)}_var.png"
            path = os.path.join(folder, fname) if folder else None
            Plotter.plot_var_and_returns(var_results = self.results["var_results"][1],log_returns = self.results["test_log_returns"][1],model_classes= models,group_name  = f"{title} distribution VaR vs Returns",dates = dates,ticker = self.ticker,save_path = path)

    def display_results_ML(self, name):
        """
        Same saving logic for ML results; uses `group1`.
        """
        folder = self._ensure_folder()
        dates  = self.results.get("test_dates", None)
        models = self.groups.get("group1", [])

        safe = Plotter._sanitize_filename(name)
        # ML Forecasts
        fname = f"{self.ticker}_{safe}_ML_forecasts.png"
        path = os.path.join(folder, fname) if folder else None
        Plotter.plot_forecasts_by_group(forecasts = self.results["forecasts"][1],realized_kernel= self.results["realized_kernel"][1],model_classes  = models,group_name = name,dates = dates,ticker = self.ticker,save_path = path)
        
        fname = f"{self.ticker}_{safe}_ML_var.png"
        path = os.path.join(folder, fname) if folder else None
        Plotter.plot_var_and_returns(var_results = self.results["var_results"][1],log_returns = self.results["test_log_returns"][1],model_classes= models,group_name  = name,dates = dates,ticker = self.ticker,save_path= path)



class TickerReport:
    def __init__(self, ticker, combined, dm_results, adjusted_pvals, results_dir):
        """
        ticker:         e.g. "AAPL"
        combined:       your combined results dict
        dm_results:     Diebold-Mariano results
        adjusted_pvals: dict {'UC':df_uc, 'CC':df_cc, 'IND':df_ind}
        results_dir:    full path to e.g. ".../High Frequency/AAPL/AAPL Results"
        """
        self.ticker         = ticker
        self.combined       = combined
        self.dm_results     = dm_results
        self.adjusted_pvals = adjusted_pvals
        self.results_dir    = results_dir
        os.makedirs(self.results_dir, exist_ok=True)

    def print_report(self):
        self.plot_out_of_sample_metrics()
        self.plot_var_backtests()
        self._print_dm_tests()

    def plot_out_of_sample_metrics(self):
        metrics = ["rmse", "mae"]
        cm = self.combined["metrics"]
        horizons = sorted(cm.keys())
        models   = sorted({m for h in horizons for m in cm[h].keys()})

        for metric in metrics:
            rel = pd.DataFrame(index=models, columns=horizons, dtype=float)
            for h in horizons:
                vals = {m: cm[h][m][metric] for m in cm[h] if metric in cm[h][m]}
                s = pd.Series(vals)
                best = s.min()
                rel[h] = (s / best).reindex(models)

            fig, ax = plt.subplots(figsize=(1.2 * len(horizons), 0.5 * len(models)))
            sns.heatmap(rel, annot=True, fmt=".2f", cmap="coolwarm",cbar_kws={"label": f"Relative {metric.upper()}"}, ax=ax)
            ax.set_title(f"{self.ticker} – {metric.upper()} Relative to Best", fontsize=14)
            ax.set_xlabel("Forecast Horizon")
            ax.set_ylabel("Model")
            plt.tight_layout()

            fname = os.path.join(self.results_dir, f"{self.ticker}_{metric}_relative.png")
            fig.savefig(fname, dpi=350)
            plt.close(fig)

    def plot_var_backtests(self):
        alpha = 0.05
        for test in ["UC", "CC", "IND"]:
            p_adj = self.adjusted_pvals[test]
            p_adj = p_adj.sort_index(axis=0).sort_index(axis=1)

            fig, ax = plt.subplots(figsize=(1.2 * p_adj.shape[1], 0.5 * p_adj.shape[0]))
            sns.heatmap(
                p_adj, annot=True, fmt=".3f", cmap="viridis_r",
                cbar_kws={"label": "FDR-adjusted p-value"},
                linewidths=0.5, linecolor="gray", mask=p_adj.isnull(),
                vmin=0, vmax=1, ax=ax
            )
            ax.set_title(f"{self.ticker}-{test} Test FDR-adj. p-values", fontsize=14)
            ax.set_xlabel("Horizon")
            ax.set_ylabel("Model")

            # highlight p<alpha with a white border
            for i, model in enumerate(p_adj.index):
                for j, hor in enumerate(p_adj.columns):
                    if p_adj.loc[model, hor] < alpha:
                        ax.add_patch(plt.Rectangle((j, i), 1, 1,fill=False,edgecolor="white",lw=2))
            
            plt.tight_layout()
            fname = os.path.join(self.results_dir, f"{self.ticker}_{test}_fdr_pvalues.png")
            fig.savefig(fname, dpi=350)
            plt.close(fig)

    def _print_dm_tests(self,alpha = 0.05):
            cmap = ListedColormap(['orange', 'royalblue'])
            cmap.set_bad('black')

            win_counts = DMPlotter.compute_win_counts(self.dm_results, alpha)
            for h, df in sorted(self.dm_results.items()):
                tops = win_counts[h].nlargest(10).index.tolist()
                n  = len(tops)
                stat_mat = np.full((n, n), np.nan)
                p_mat = np.full((n, n), np.nan)

                for i, m1 in enumerate(tops):
                    for j, m2 in enumerate(tops):
                        if i == j: 
                            continue
                        sub = df[((df['Model 1']==m1)&(df['Model 2']==m2))|((df['Model 1']==m2)&(df['Model 2']==m1))]
                        if not sub.empty:
                            stat_mat[i,j] = sub['DM Statistic'].iloc[0]
                            p_mat[i,j] = sub['P-Value'].iloc[0]

                signif = (p_mat < alpha)
                binary = np.where(np.isnan(stat_mat), np.nan, signif.astype(int))

                fig, ax = plt.subplots(figsize=(1.5*n, 0.5*n))
                ax.imshow(binary, cmap=cmap, aspect='auto')
                ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
                ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
                ax.set_axisbelow(False)
                ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
                ax.tick_params(which='minor', bottom=False, left=False)

                for i in range(n):
                    for j in range(n):
                        v = stat_mat[i,j]
                        if not np.isnan(v):
                            color = 'white' if signif[i,j] else 'black'
                            ax.text(j, i, f"{v:.3f}", ha='center', va='center', color=color)

                ax.set_xticks(np.arange(n))
                ax.set_xticklabels(tops, rotation=45, ha='right')
                ax.set_yticks(np.arange(n))
                ax.set_yticklabels(tops)
                handles = [mpatches.Patch(color='orange', label=f"p < {alpha}"), mpatches.Patch(color='royalblue', label=f"p ≥ {alpha}")]
                ax.legend(handles=handles, bbox_to_anchor=(1.05,1), loc='upper left')
                ax.set_title(f"DM Test Statistics – Horizon {h}", fontsize=14)
                plt.tight_layout()
                fname = os.path.join(self.results_dir, f"{self.ticker}_{h}_DM_results.png")
                fig.savefig(fname, dpi=350)
                plt.close(fig)