import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Patch
import pandas as pd
import seaborn as sns
import numpy as np
from collections import defaultdict
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import os

import matplotlib as mpl

# automatically apply tight_layout on every figure
mpl.rcParams['figure.autolayout']   = True

# when saving, crop whitespace and use zero padding
mpl.rcParams['savefig.bbox']        = 'tight'
mpl.rcParams['savefig.pad_inches']  = 0

# (optional) force edge-to-edge subplots
mpl.rcParams['figure.subplot.left']   = 0
mpl.rcParams['figure.subplot.right']  = 1
mpl.rcParams['figure.subplot.bottom'] = 0
mpl.rcParams['figure.subplot.top']    = 1

class DMPlotter:
    @staticmethod
    def compute_win_counts(dm_by_horizon, alpha=0.05):
  
        # collect all model names
        all_models = set()
        for df in dm_by_horizon.values():
            # Take the union
            all_models |= set(df['Model 1']) | set(df['Model 2'])
  
        all_models = sorted(all_models)

        win_counts = {}
        for h, df in dm_by_horizon.items():
            counts = defaultdict(int)
            sig = df[df['P-Value'] < alpha]
            for _, row in sig.iterrows():

                m1, m2, stat = row['Model 1'], row['Model 2'], row['DM Statistic']
                if stat < 0:
                    counts[m1] += 1
                else:
                    counts[m2] += 1
            # build full Series (zeros for models with no wins)
            s = pd.Series(counts, index=all_models).fillna(0).astype(int)
            win_counts[h] = s

        return win_counts

    @staticmethod
    def plot_top_models(dm_by_horizon, top_n=10, alpha=0.05):
        
        counts = DMPlotter.compute_win_counts(dm_by_horizon, alpha)
        horizons = sorted(counts.keys())
        n  = len(horizons)
        n_cols = 2
        n_rows = (n + 1) // 2

        fig, axs = plt.subplots(n_rows, n_cols,figsize=(6 * n_cols, 5 * n_rows),sharey=True)
        axs = np.array(axs).flatten()

        for ax, h in zip(axs, horizons):
            series = counts[h]
            top = series.nlargest(top_n)
            ax.bar(top.index,top.values,color='royalblue', edgecolor='black')
            ax.set_title(f"Top {top_n} DM Winners (h={h})", fontsize=12, fontweight='bold')
            ax.set_xlabel("Model", fontsize=10)
            ax.set_ylabel("Wins", fontsize=10)
            ax.tick_params(axis='x', rotation=45, labelsize=8)
            ax.grid(axis='y', linestyle='--', alpha=0.6)

        # hide extra panels
        for ax in axs[n:]:
            ax.set_visible(False)

        plt.tight_layout()
        fname = "DM_TOP_MODELS.png"
        fig.savefig(os.path.join(r"C:\Users\rickt\Desktop\Econometrics and Data Science\Thesis\Data\Stock\High Frequency\General Results", fname),dpi=350,bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_dm_network(dm_by_horizon, model_categories, alpha=0.05, figsize=(20, 15)):
    
        #Set automatically removes duplicates
        all_models = set()
        for df in dm_by_horizon.values():
            all_models |= set(df['Model 1']) | set(df['Model 2'])

        # aggregate win counts across horizons
        counts = DMPlotter.compute_win_counts(dm_by_horizon, alpha)
        total_wins = defaultdict(int)
        for series in counts.values():
            for m, c in series.items():
                total_wins[m] += int(c)

        # build graph
        G = nx.DiGraph()
        G.add_nodes_from(all_models)

        # node aesthetics
        pos = nx.spring_layout(G, k=1.0, scale=2.5)
        sizes = []
        max_w  = max(total_wins.values(), default=1)
        for m in all_models:
            sizes.append(300 + 5700 * (total_wins.get(m, 0) / max_w))

        cat_color_map = {"Normal": "royalblue","Student t": "firebrick","AST": "purple","EGB2": "darkgreen","ML":"orange","DL":"gray"}
        colors = [cat_color_map.get(model_categories.get(m, "Normal"), "black") for m in all_models]

        fig, ax = plt.subplots(figsize=figsize)
        nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color=colors, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)

        # edges per horizon
        horizons = sorted(dm_by_horizon.keys())
        cmap  = plt.get_cmap('tab10')
        hcolors  = {h: cmap(i % cmap.N) for i, h in enumerate(horizons)}
        n  = len(horizons)

        for idx, h in enumerate(horizons):
            df = dm_by_horizon[h]
            edgelist = [(r['Model 1'], r['Model 2']) for _, r in df.iterrows()if r['P-Value'] < alpha]
            rad = (idx / max(1, n-1) - 0.5) * 0.6
            nx.draw_networkx_edges(G, pos,edgelist=edgelist,ax=ax,connectionstyle=f'arc3,rad={rad}',edge_color=[hcolors[h]] * len(edgelist),arrows=True,arrowstyle='-|>',arrowsize=12,width=2,alpha=0.7)

        # legends
        node_handles = [Patch(color=c, label=cat) for cat,c in cat_color_map.items()]
        leg1 = ax.legend(handles=node_handles, title='Model Category', loc='upper left')
        ax.add_artist(leg1)
        horizon_handles = [Patch(color=hcolors[h], label=f'h={h}') for h in horizons]
        ax.legend(handles=horizon_handles, title='Horizon', loc='upper right')

        ax.set_title("DM Network with Horizon-Colored Edges", fontsize=16)
        ax.axis('off')
        plt.tight_layout()
        fname = "DM_NETWORK.png"
        fig.savefig(os.path.join(r"C:\Users\rickt\Desktop\Econometrics and Data Science\Thesis\Data\Stock\High Frequency\General Results", fname),dpi=350,bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_top_11_comparison_per_horizon(dm_by_horizon, alpha=0.05):
        cmap = ListedColormap(['orange', 'royalblue'])
        cmap.set_bad('black')

        win_counts = DMPlotter.compute_win_counts(dm_by_horizon, alpha)
        for h, df in sorted(dm_by_horizon.items()):
            tops = win_counts[h].nlargest(10).index.tolist()
            n  = len(tops)
            stat_mat = np.full((n, n), np.nan)
            p_mat = np.full((n, n), np.nan)

            for i, m1 in enumerate(tops):
                for j, m2 in enumerate(tops):
                    if i == j: 
                        continue
                    sub = df[((df['Model 1']==m1)&(df['Model 2']==m2))|((df['Model 1']==m2)&(df['Model 2']==m1))]
                    stat_mat[i,j] = sub['DM Statistic'].iloc[0]
                    p_mat[i,j]    = sub['P-Value'].iloc[0]

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
            fname = f"DM_TOP_TABLE_{h}.png"
            fig.savefig(os.path.join(r"C:\Users\rickt\Desktop\Econometrics and Data Science\Thesis\Data\Stock\High Frequency\General Results", fname),dpi=350,bbox_inches='tight')
            plt.show()


class DumbbellPlotter:

    @staticmethod
    def plot_relative(rmse_df_all, metric):
       
        n = len(rmse_df_all)
        n_cols = 2
        n_rows = (n + 1) // 2

        fig, axs = plt.subplots(n_rows,n_cols,figsize=(6 * n_cols, 6 * n_rows),sharey=True)
        axs = axs.flatten()

        for ax, (h, df) in zip(axs, sorted(rmse_df_all.items())):
            
            rel = df.div(df.mean(axis=1), axis=0)
            
            rel_avg = rel.mean(axis=1)
            
            top_models = rel_avg.nsmallest(10).index.tolist()
            rel = rel.loc[top_models]

            melted = (rel.reset_index().melt(id_vars='index', var_name='Ticker', value_name='RelError').rename(columns={'index': 'Model'}))
            models = melted['Model'].unique()
            cmap = plt.get_cmap('tab10')
            color_map = {m: cmap(i % 10) for i, m in enumerate(models)}

            
            for _, row in melted.iterrows():
                ax.hlines(y=row['Ticker'],xmin=1.0,xmax=row['RelError'],color='gray',alpha=0.5)
                ax.plot(row['RelError'],row['Ticker'],'o',color=color_map[row['Model']],label=row['Model'],alpha=0.8)

            ax.axvline(1.0, color='black', linestyle='--')
            ax.set_title(f"Average {metric} at horizon (h={h})", fontsize=12, fontweight='bold')
            ax.set_xlabel("Relative Error", fontsize=10)

        
        handles, labels = axs[0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        fig.legend(by_label.values(),by_label.keys(),title='Model',loc='lower center',ncol=5)

        plt.tight_layout(rect=[0, 0.1, 1, 1])
        fname = f"DUMBEL_PLOT_{metric}.png"
        fig.savefig(os.path.join(r"C:\Users\rickt\Desktop\Econometrics and Data Science\Thesis\Data\Stock\High Frequency\General Results", fname),dpi=350,bbox_inches='tight')
        plt.show()


class PanelErrorRanking:
    @staticmethod
    def compute_average_metric(results_by_ticker, metric):
        
        sample = next(iter(results_by_ticker.values()))['combined']
        horizons = sorted(sample['metrics'].keys())
        stocks = sorted(results_by_ticker.keys())
        
        avg_metric = {}
        for h in horizons:
            df = pd.DataFrame(index=[], columns=stocks, dtype=float)
            models = sorted({metrics for res in results_by_ticker.values() for _, metrics in res['combined']['metrics'].items()  for metrics in metrics.keys()})
            df = pd.DataFrame(index=models, columns=stocks, dtype=float)
            for ticker, res in results_by_ticker.items():
                mets = res['combined']['metrics'].get(h, {})
                for m, met in mets.items():
                    df.at[m, ticker] = met.get(metric, pd.NA)
            avg_metric[h] = df.mean(axis=1)
        return avg_metric
    
    @staticmethod
    def plot_scaled_average_metric(results_by_ticker, metric='rmse', top_n=10):
       
        avg_metric = PanelErrorRanking.compute_average_metric(results_by_ticker, metric)
        n = len(avg_metric)
        n_cols = 2
        n_rows = (n + 1) // 2
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), sharex=False)
        axs = axs.flatten()
        
        for ax, (h, series) in zip(axs, avg_metric.items()):
            series = series.dropna().sort_values()
            top_series = series.iloc[:top_n]
            scaled = top_series / top_series.iloc[0]
            scaled.plot.barh(ax=ax, color='teal', edgecolor='black')
            ax.set_title(f"Top {top_n} Models by Avg {metric.upper()} (h={h})", fontsize=12, fontweight='bold')
            ax.set_xlabel(f"Relative Avg {metric.upper()} (best = 1.0)", fontsize=10)
            ax.invert_yaxis()
        
        for ax in axs[len(avg_metric):]:
            ax.set_visible(False)
        
        plt.tight_layout()
        fname = f"REL_PLOT_{metric}.png"
        fig.savefig(os.path.join(r"C:\Users\rickt\Desktop\Econometrics and Data Science\Thesis\Data\Stock\High Frequency\General Results", fname),dpi=350,bbox_inches='tight')
        plt.show()



class ModelSelectionTables:
    
    @staticmethod
    def print_selection(results_dict):
        param_first = {ticker: stock_res['parametric']['first_window_results'] for ticker, stock_res in results_dict.items()}
        all_models = sorted({model for md in param_first.values() for model in md})
        aic_df = pd.DataFrame(index=all_models, columns=param_first.keys(), dtype=float)
        bic_df = aic_df.copy()
        ll_df = aic_df.copy()

        for ticker, model_dict in param_first.items():
            for model, res in model_dict.items():
                aic_df.loc[model, ticker] = res.get('aic')
                bic_df.loc[model, ticker] = res.get('bic')
                ll_df.loc[model, ticker] = res.get('log_likelihood')

        print("\n=== AIC ===")
        print(aic_df.round(2).to_string())
        print("\n=== BIC ===")
        print(bic_df.round(2).to_string())
        print("\n=== Log-Likelihood ===")
        print(ll_df.round(2).to_string())


class MLHyperparamCollector:

    @staticmethod
    def print_hyperparams(results):
        ml_first = {ticker: res['ml']['first_window_results'] for ticker, res in results.items()}
        stocks = list(ml_first.keys())
        rows = []

        for model in sorted({m for d in ml_first.values() for m in d}):
            rows.append({'Model': model, 'Hyper-parameter': '—', **{s: '' for s in stocks}})
            all_params = sorted({param for d in ml_first.values() if model in d for param in d[model]})
            for param in all_params:
                row = {'Model': '', 'Hyper-parameter': param}
                for stock in stocks:
                    row[stock] = ml_first[stock].get(model, {}).get(param, np.nan)
                rows.append(row)

        df = pd.DataFrame(rows, columns=['Model', 'Hyper-parameter'] + stocks)
        print("\n=== Optimal Hyperparameters Machine Learning ===")
        print(df.round(2).to_string())



class MCSBordaAnalyzer:
    
    @staticmethod
    def compute_borda(results_by_ticker):
      
        # determine horizons
        sample = next(iter(results_by_ticker.values()))
        horizons = sorted(sample['mcs'].keys())

        # collect all model names
        all_models = sorted({model for res in results_by_ticker.values() for h in res['mcs']for model in res['mcs'][h]['mcs'].index})

        borda_by_horizon = {}
        for h in horizons:
            cumulative = defaultdict(int)
            for res in results_by_ticker.values():
                entry = res['mcs'].get(h, {})
                df = entry.get('mcs')
                status = df['status']

                # everyone starts with score 1
                scores = {m: 1 for m in status.index}
                
                # excluded models ranked worse
                excluded = status[status == 'excluded'].index[::-1]
                
                for rank, m in enumerate(excluded, start=2):
                    scores[m] = rank
                
                # accumulate
                for m, sc in scores.items():
                    cumulative[m] += sc
            # convert to Series
            series = pd.Series({m: cumulative.get(m, 0) for m in all_models},name='Borda')
            # Invert the Borda.
            series = (series.max() - series)

            borda_by_horizon[h] = series
        
        return borda_by_horizon

    @staticmethod
    def plot_borda(results_by_ticker, top_n=10):
        """
        Plot the top-n Borda scores per horizon in a 2x2 grid.
        """
        borda_by_horizon = MCSBordaAnalyzer.compute_borda(results_by_ticker)
        horizons = sorted(borda_by_horizon.keys())
        n = len(horizons)
        n_cols = 2
        n_rows = (n + 1) // 2

        fig, axs = plt.subplots(n_rows,n_cols,figsize=(8 * n_cols, 5 * n_rows),sharex=False,sharey=False)
        axs = axs.flatten()

        for ax, h in zip(axs, horizons):
            series = borda_by_horizon[h]
            top_series = series.nlargest(top_n)

            ax.barh(top_series.index[::-1],top_series.values[::-1],color='teal',edgecolor='black')
            ax.set_xlabel('Cumulative Borda score')
            ax.set_title(f'h = {h}', fontsize=12, fontweight='bold')

        # hide unused axes
        for ax in axs[n:]:
            ax.set_visible(False)

        fig.suptitle('Top-10 Models by MCS Survival (Higher Borda ⇒ Better)',fontsize=14,fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        fname = "BORDA.png"
        fig.savefig(os.path.join(r"C:\Users\rickt\Desktop\Econometrics and Data Science\Thesis\Data\Stock\High Frequency\General Results", fname),dpi=350,bbox_inches='tight')
        plt.show()

class DLResultsAnalyser:
    @staticmethod
    def show_aggregated_learning_curves(results):

        # Extract DL results
        dl_results_all = {ticker: res['dl'] for ticker, res in results.items()}
        base_epochs    = 200
        model_name     = "LSTM"

        # Gather per‐stock curves
        train_curves = []
        val_curves   = []

        for ticker, dl_res in dl_results_all.items():
            train_curves.append(dl_res["training_errors"][model_name])     
            val_curves.append(  dl_res["validation_errors"][model_name])  

        # Stack into (n_stocks, base_epochs)
        train_arr = np.vstack(train_curves)
        val_arr   = np.vstack(val_curves)

        # Compute mean across stocks
        mean_train = train_arr.mean(axis=0)
        mean_val   = val_arr.mean(axis=0)

        # Plot
        plt.figure(figsize=(8, 4))
        epochs = np.arange(1, base_epochs+1)

        plt.plot(epochs, mean_train, label="Train MSE")
        plt.plot(epochs, mean_val,   label="Val MSE")
        plt.xlabel("Epoch")
        plt.ylabel("Mean Squared Error")
        plt.title("Average Train vs. Validation MSE per Epoch across Stocks")
        plt.legend()
        plt.tight_layout()
        fname = f"{model_name}_LEARNING_CURVES.png"
        plt.savefig(os.path.join(r"C:\Users\rickt\Desktop\Econometrics and Data Science\Thesis\Data\Stock\High Frequency\General Results",fname),dpi=350, bbox_inches='tight')
        plt.show()


class TOPSISRanker:

    @staticmethod
    def rank_models(results_by_ticker,dm_results,horizons,weights,alpha,top_n=10):
        

        rmse_df_all = PanelErrorRanking.compute_average_metric(results_by_ticker, metric='rmse')
        mae_df_all =  PanelErrorRanking.compute_average_metric(results_by_ticker, metric='mae')
        dm_win_counts = DMPlotter.compute_win_counts(dm_results,alpha)
        mcs_borda_scores = MCSBordaAnalyzer.compute_borda(results_by_ticker)

        n_cols = 2
        n_rows = 2
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows))
        axs = np.array(axs).flatten()

        for idx, h in enumerate(horizons):
            ax = axs[idx]

            rmse_avg = rmse_df_all[h].mean(axis=1)
            mae_avg  = mae_df_all[h].mean(axis=1)
            models = rmse_avg.index
            dm = dm_win_counts.get(h, pd.Series(0, index=models)).reindex(models, fill_value=0)
            borda = mcs_borda_scores.get(h, pd.Series(0, index=models)).reindex(models, fill_value=0)

            
            crit = pd.DataFrame({'RMSE':rmse_avg,'MAE':mae_avg,'DM_wins': dm,'Borda': borda})
            norm = (crit - crit.min()) / (crit.max() - crit.min())
            norm['RMSE'] = 1 - norm['RMSE']
            norm['MAE']  = 1 - norm['MAE']

            W = norm.values * np.array(weights)
            ideal  = W.max(axis=0)
            nadir  = W.min(axis=0)
            d_best  = np.linalg.norm(W - ideal,  axis=1)
            d_worst = np.linalg.norm(W - nadir,  axis=1)
            score = pd.Series(d_worst / (d_best + d_worst), index=models)

            
            top_series = score.nlargest(top_n)
            ax.barh(top_series.index[::-1],top_series.values[::-1],color='skyblue', edgecolor='black')
            ax.set_title(f'h = {h}', fontsize=12, fontweight='bold')
            ax.set_xlabel('TOPSIS score')
            ax.set_xlim(0, 1)

        # hide any extra axes
        for ax in axs[len(horizons):]:
            ax.set_visible(False)

        fig.suptitle('TOPSIS Ranking Across Horizons', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        fname = "TOPSIS.png"
        fig.savefig(os.path.join(r"C:\Users\rickt\Desktop\Econometrics and Data Science\Thesis\Data\Stock\High Frequency\General Results", fname),dpi=350,bbox_inches='tight')
        plt.show()
