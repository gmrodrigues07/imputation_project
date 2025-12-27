import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

class ImputationVisualizer:
    def __init__(self, dataset_name, metrics_summary=None, monte_carlo_df=None):
        self.raw_name = dataset_name
        self.folder_name = os.path.splitext(dataset_name)[0]
        self.output_dir = os.path.join('../results', self.folder_name)
        
        self.metrics_summary = metrics_summary
        self.mc_df = monte_carlo_df
        
        plt.style.use('ggplot')
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        if not os.path.exists(self.output_dir):
            try:
                os.makedirs(self.output_dir)
            except OSError:
                pass

    def _get_method_color(self, method_name, method_list):
        try:
            m_list_lower = [str(m).lower() for m in method_list]
            idx = m_list_lower.index(str(method_name).lower())
            return self.colors[idx % len(self.colors)]
        except ValueError:
            return 'gray'

    # --- PLOTS DE BARRAS (Fase 1) ---
    def plot_bar_comparison(self, metric='rmse'):
        if self.metrics_summary is None: return
        col_mean = f"{metric.upper()}_mean"
        if col_mean not in self.metrics_summary.columns: return

        df = self.metrics_summary
        methods = df['Method'].tolist()
        values = df[col_mean].tolist()
        
        save_path = os.path.join(self.output_dir, f'bar_comparison_{metric.lower()}.png')
        plt.figure(figsize=(10, 6))
        bars = plt.bar(methods, values, color=[self._get_method_color(m, methods) for m in methods])
        plt.ylabel(f'Mean {metric.upper()}')
        plt.title(f'{metric.upper()} Comparison (Lower is Better)')
        for bar in bars:
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{bar.get_height():.4f}', ha='center', va='bottom')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

    def plot_computation_time(self):
        if self.metrics_summary is None or 'Time_mean' not in self.metrics_summary.columns: return
        df = self.metrics_summary
        methods = df['Method'].tolist()
        times = df['Time_mean'].tolist()
        
        plt.figure(figsize=(10, 6))
        bars = plt.barh(methods, times, color=self.colors[:len(methods)])
        plt.xlabel('Time (seconds)')
        plt.title('Computational Efficiency')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'computation_time.png'), dpi=300)
        plt.close()

    # --- PLOTS DE MONTE CARLO (Fase 1 e 2) ---
    def plot_monte_carlo_stability(self, metric='rmse'):
        if self.mc_df is None: return
        metric_col = next((c for c in self.mc_df.columns if c.lower() == metric.lower()), None)
        if not metric_col: return

        plt.figure(figsize=(12, 7))
        methods = self.mc_df['method'].unique()
        data = [self.mc_df[self.mc_df['method'] == m][metric_col].values for m in methods]
        
        box = plt.boxplot(data, labels=methods, patch_artist=True)
        for patch, m in zip(box['boxes'], methods):
            patch.set_facecolor(self._get_method_color(m, methods))
            patch.set_alpha(0.6)
            
        plt.title(f'Stability Analysis ({metric.upper()})')
        plt.ylabel(metric.upper())
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'mc_stability_{metric.lower()}.png'), dpi=300)
        plt.close()

    def plot_error_density(self, metric='rmse'):
        if self.mc_df is None: return
        metric_col = next((c for c in self.mc_df.columns if c.lower() == metric.lower()), None)
        if not metric_col: return

        fig, ax = plt.subplots(figsize=(12, 6))
        methods = self.mc_df['method'].unique()
        for i, m in enumerate(methods):
            subset = self.mc_df[self.mc_df['method'] == m][metric_col]
            try:
                subset.plot(kind='density', ax=ax, label=m, linewidth=2, color=self._get_method_color(m, methods))
            except: pass
        plt.title(f'Error Density ({metric.upper()})')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'mc_density_{metric.lower()}.png'), dpi=300)
        plt.close()

    def plot_convergence(self, metric='rmse'):
        if self.mc_df is None: return
        metric_col = next((c for c in self.mc_df.columns if c.lower() == metric.lower()), None)
        if not metric_col: return

        plt.figure(figsize=(12, 6))
        methods = self.mc_df['method'].unique()
        for i, m in enumerate(methods):
            subset = self.mc_df[self.mc_df['method'] == m].reset_index(drop=True)
            if not subset.empty:
                plt.plot(subset.index + 1, subset[metric_col].expanding().mean(), 
                         label=m, color=self._get_method_color(m, methods))
        
        plt.title(f'Convergence ({metric.upper()})')
        plt.xlabel('Iterations')
        plt.ylabel(f'Cumulative Mean {metric.upper()}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'mc_convergence_{metric.lower()}.png'), dpi=300)
        plt.close()

    def plot_downstream_uncertainty(self):
        if self.mc_df is None: return
        acc_col = next((c for c in self.mc_df.columns if 'accuracy' in c.lower()), None)
        if not acc_col: return

        plt.figure(figsize=(10, 6))
        methods = self.mc_df['method'].unique()
        data = [self.mc_df[self.mc_df['method'] == m][acc_col].dropna().values for m in methods]
        
        parts = plt.violinplot(data, showmeans=False, showmedians=True)
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(self._get_method_color(methods[i], methods))
            pc.set_alpha(0.7)
            
        plt.xticks(np.arange(1, len(methods) + 1), methods)
        plt.title('Decision Certainty (Model Accuracy)')
        plt.ylabel('Accuracy')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'downstream_uncertainty.png'), dpi=300)
        plt.close()

    # --- PLOT DE SCATTER (Novo!) ---
    def plot_scatter_comparison(self, df_raw, imputed_dict, col_x, col_y):
        """
        Plota Scatter: Pontos Observados (Cinza) vs Pontos Imputados (Coloridos).
        """
        # Verifica se as colunas existem
        if col_x not in df_raw.columns or col_y not in df_raw.columns:
            return

        # Máscara: Onde havia missing ORIGINALMENTE em X ou Y?
        # Esses são os pontos que queremos destacar como "Imputados"
        mask_missing = df_raw[col_x].isna() | df_raw[col_y].isna()
        mask_observed = ~mask_missing
        
        # Dados Observados (Base Comum)
        obs_x = df_raw.loc[mask_observed, col_x]
        obs_y = df_raw.loc[mask_observed, col_y]

        # Configurar subplot: 1 linha, N colunas (uma para cada método)
        methods = sorted(list(imputed_dict.keys()))
        fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 5), sharey=True)
        if len(methods) == 1: axes = [axes]
        
        for ax, method in zip(axes, methods):
            df_filled = imputed_dict[method]
            
            # 1. Plotar fundo cinza (observados)
            ax.scatter(obs_x, obs_y, color='lightgray', alpha=0.5, label='Observed', s=15)
            
            # 2. Plotar imputados (onde era missing)
            imp_x = df_filled.loc[mask_missing, col_x]
            imp_y = df_filled.loc[mask_missing, col_y]
            
            color = self._get_method_color(method, methods)
            ax.scatter(imp_x, imp_y, color=color, alpha=0.8, label='Imputed', s=20, edgecolor='white', linewidth=0.5)
            
            ax.set_title(f'{method}')
            ax.set_xlabel(col_x)
            if ax == axes[0]: ax.set_ylabel(col_y)
            ax.legend()

        plt.suptitle(f'Imputation Plausibility: {col_x} vs {col_y}', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'scatter_{col_x}_vs_{col_y}.png'), dpi=300)
        plt.close()

    def plot_distribution_comparison(self, df_raw, imputed_dict, col):
        if col not in df_raw.columns: return
        
        fig, axes = plt.subplots(1, len(imputed_dict)+1, figsize=(4*(len(imputed_dict)+1), 4))
        
        # Original
        data_orig = df_raw[col].dropna()
        axes[0].hist(data_orig, bins=30, color='black', alpha=0.7, density=True)
        axes[0].set_title('Original')
        
        methods = sorted(list(imputed_dict.keys()))
        for i, m in enumerate(methods, 1):
            data = imputed_dict[m][col]
            axes[i].hist(data, bins=30, color=self._get_method_color(m, methods), alpha=0.7, density=True)
            axes[i].set_title(m)
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'distribution_{col}.png'), dpi=300)
        plt.close()