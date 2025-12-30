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

    def get_method_color(self, method_name, method_list):
        try:
            m_list_lower = [str(m).lower() for m in method_list]
            idx = m_list_lower.index(str(method_name).lower())
            return self.colors[idx % len(self.colors)]
        except ValueError:
            return 'gray'

    # PLOTS DE BARRAS 
    def plot_bar_comparison(self, metric='rmse'):
        if self.metrics_summary is None: return
        col_mean = f"{metric.upper()}_mean"
        if col_mean not in self.metrics_summary.columns: return

        df = self.metrics_summary
        methods = df['Method'].tolist()
        values = df[col_mean].tolist()
        
        save_path = os.path.join(self.output_dir, f'bar_comparison_{metric.lower()}.png')
        plt.figure(figsize=(10, 6))
        bars = plt.bar(methods, values, color=[self.get_method_color(m, methods) for m in methods])
        plt.ylabel(f'Mean {metric.upper()}')
        plt.title(f'{metric.upper()} Comparison') # lower is better
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
        plt.xlabel('Time (sec.)')
        plt.title('Computational Efficiency')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'computation_time.png'), dpi=300)
        plt.close()

    # PLOTS DE MONTE CARLO (fases 1 e 2) 
    def plot_monte_carlo_stability(self, metric='rmse'):
        if self.mc_df is None: return
        metric_col = next((c for c in self.mc_df.columns if c.lower() == metric.lower()), None)
        if not metric_col: return

        plt.figure(figsize=(12, 7))
        methods = self.mc_df['method'].unique()
        data = [self.mc_df[self.mc_df['method'] == m][metric_col].values for m in methods]
        
        box = plt.boxplot(data, labels=methods, patch_artist=True)
        for patch, m in zip(box['boxes'], methods):
            patch.set_facecolor(self.get_method_color(m, methods))
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
                subset.plot(kind='density', ax=ax, label=m, linewidth=2, color=self.get_method_color(m, methods))
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
                         label=m, color=self.get_method_color(m, methods))
        
        plt.title(f'Convergence ({metric.upper()})')
        plt.xlabel('Iterations')
        plt.ylabel(f'Cumulative Mean {metric.upper()}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'mc_convergence_{metric.lower()}.png'), dpi=300)
        plt.close()

    def plot_downstream_uncertainty(self):
        """
        Gera Boxplots da acurácia usando APENAS Matplotlib.
        Cria subplots (um para cada classificador) lado a lado.
        """
        if self.mc_df is None: return
        
        # Encontra colunas de acurácia
        acc_cols = [c for c in self.mc_df.columns if c.startswith('accuracy_')]
        if not acc_cols: return

        # Configura subplots: 1 linha, N colunas (uma por classificador)
        n_plots = len(acc_cols)
        fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 6), sharey=True)
        
        if n_plots == 1: axes = [axes] # Garante que é lista mesmo se for só 1
        
        methods = self.mc_df['method'].unique()
        
        for ax, col in zip(axes, acc_cols):
            clf_name = col.replace('accuracy_', '')
            
            # Prepara dados para o boxplot: lista de listas [[valores_metodo1], [valores_metodo2]...]
            data_to_plot = []
            labels = []
            colors = []
            
            for m in methods:
                vals = self.mc_df[self.mc_df['method'] == m][col].dropna().values
                data_to_plot.append(vals)
                labels.append(m)
                colors.append(self.get_method_color(m, methods))
            
            # Cria o Boxplot
            bplot = ax.boxplot(data_to_plot, patch_artist=True, labels=labels)
            
            # Colora as caixas
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
                
            ax.set_title(clf_name, fontweight='bold')
            ax.grid(True, axis='y', linestyle='--', alpha=0.5)
            ax.tick_params(axis='x', rotation=45)
            
            if ax == axes[0]:
                ax.set_ylabel('Accuracy Distribution')

        plt.suptitle('Downstream Model Stability (Boxplots)', fontsize=14)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9) # Espaço para o título principal
        
        save_path = os.path.join(self.output_dir, 'downstream_uncertainty.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
        
    #  PLOT DE SCATTER 
    def plot_scatter_comparison(self, df_raw, imputed_dict, col_x, col_y):
        """
        Plota Scatter: Pontos Observados a cinza vs Pontos Imputados a cor.
        """
        # Verifica se as colunas existem
        if col_x not in df_raw.columns or col_y not in df_raw.columns:
            return

        mask_missing = df_raw[col_x].isna() | df_raw[col_y].isna()
        mask_observed = ~mask_missing
        
        # dados observados, a base comum
        obs_x = df_raw.loc[mask_observed, col_x]
        obs_y = df_raw.loc[mask_observed, col_y]

        # configurar a subplot. 1 linha, N colunas (uma para cada método)
        methods = sorted(list(imputed_dict.keys()))
        fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 5), sharey=True)
        if len(methods) == 1: axes = [axes]
        
        for ax, method in zip(axes, methods):
            df_filled = imputed_dict[method]
            
            # 1. Plotar fundo cinza (observados)
            ax.scatter(obs_x, obs_y, color='lightgray', alpha=0.5, label='Observed', s=15)
            
            # 2. Plotar imputados (missing)
            imp_x = df_filled.loc[mask_missing, col_x]
            imp_y = df_filled.loc[mask_missing, col_y]
            
            color = self.get_method_color(method, methods)
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
            axes[i].hist(data, bins=30, color=self.get_method_color(m, methods), alpha=0.7, density=True)
            axes[i].set_title(m)
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'distribution_{col}.png'), dpi=300)
        plt.close()
    
    def plot_imputation_vs_downstream(self):
        """
        Plota RMSE vs Acurácia usando APENAS Matplotlib.
        """
        if self.metrics_summary is None: return
        
        # 1. Encontrar colunas de acurácia dinamicamente
        acc_cols = [c for c in self.metrics_summary.columns if c.startswith('accuracy_')]
        if not acc_cols or 'RMSE_mean' not in self.metrics_summary.columns: return
        
        plt.figure(figsize=(10, 7))
        
        # Marcadores nativos do Matplotlib
        markers = ['o', 's', '^', 'D', 'v'] 
        
        # Listas para criar legendas manuais
        classifier_plots = []
        classifier_names = []
        
        # 2. Loop para plotar cada classificador
        for i, acc_col in enumerate(acc_cols):
            clf_name = acc_col.replace('accuracy_', '')
            marker = markers[i % len(markers)]
            
            # Ponto "fantasma" preto para a legenda das Formas
            p, = plt.plot([], [], color='k', marker=marker, linestyle='None', 
                          markersize=8, label=clf_name)
            classifier_plots.append(p)
            classifier_names.append(clf_name)
            
            # Loop pelos métodos (cores)
            for _, row in self.metrics_summary.iterrows():
                method = row['Method']
                rmse = row['RMSE_mean']
                acc = row[acc_col]
                color = self.get_method_color(method, self.metrics_summary['Method'].tolist())
                
                plt.scatter(rmse, acc, color=color, marker=marker, s=120, 
                            edgecolors='k', alpha=0.9, zorder=3)

        # 3. Legenda de Cores (Métodos) - Criação manual
        method_handles = []
        unique_methods = self.metrics_summary['Method'].unique()
        for m in unique_methods:
            c = self.get_method_color(m, unique_methods)
            h, = plt.plot([], [], color=c, marker='o', linestyle='None', markersize=10)
            method_handles.append(h)
            
        # Adicionar as legendas
        first_legend = plt.legend(classifier_plots, classifier_names, title="Classifiers", 
                                  loc='lower right', frameon=True)
        plt.gca().add_artist(first_legend)
        
        plt.legend(method_handles, unique_methods, title="Methods", 
                   loc='lower left', frameon=True)

        plt.title('Imputation RMSE vs Downstream Accuracy', fontsize=14)
        plt.xlabel('Imputation RMSE (Lower is Better)', fontsize=12)
        plt.ylabel('Accuracy (Higher is Better)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        save_path = os.path.join(self.output_dir, 'correlation_rmse_accuracy_combined.png')
        plt.savefig(save_path, dpi=300)
        plt.close()        