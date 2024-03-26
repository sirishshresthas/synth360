import os
import math
import torch
import numpy as np
import pandas as pd
from typing import Any, List, Dict, Optional
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer, CopulaGANSynthesizer
from sdv.evaluation.single_table import run_diagnostic, evaluate_quality
from src.core.utilities import globals
from sdv.evaluation.single_table import get_column_plot
import plotly.io as pio
from plotly.subplots import make_subplots
# from hyperopt import hp, STATUS_OK, fmin, tpe, Trials
import plotly.graph_objects as go
import plotly.io as pio
from pathlib import Path


class SynthLeader(object):

    def __init__(self, df: pd.DataFrame, name: str = ''):
        self.df = df
        self.name = name

        self.data_dir = Path("./data/")
        self.sub_dirs = ['models', 'figures', 'metadata']

        for sub_dir in self.sub_dirs:
            dir_path = self.data_dir / sub_dir
            dir_path.mkdir(parents=True, exist_ok=True)
            setattr(self, f"{sub_dir}_dir", dir_path)

        self._metadata: SingleTableMetadata = self.create_metadata()
        print(
            f"GPU: {torch.cuda.is_available() or torch.backends.mps.is_available()}")

    def __repr__(self):
        """
        Provides a string representation of the SynthLeader instance, including its class type and data name.
        """
        return f"{type(self.__class__.__name__)} {self.name}"

    @property
    def metadata(self) -> SingleTableMetadata:
        return self._metadata

    @metadata.setter
    def metadata(self, metadata: SingleTableMetadata) -> None:
        self._metadata = metadata

    def create_metadata(self) -> SingleTableMetadata:
        metadata: SingleTableMetadata = SingleTableMetadata()

        metadata.detect_from_dataframe(self.df)
        metadata.validate()
        # metadata.save_to_json(filepath=self.metadata_dir / "metadata.json")

        return metadata

    def _get_numeric_cols(self) -> List[str]:
        cols = self.df.select_dtypes(
            include=[np.number]).columns.to_list()  # type: ignore
        return cols

    def generate_corr_matrix(self, df: pd.DataFrame):

        matrix = df[self._get_numeric_cols()].corr()

        return matrix

    def style_correlation_matrix(self, matrix, style="coolwarm"):
        styled_matrix = matrix \
            .style \
            .background_gradient(cmap=style) \
            .format(precision=3)

        return styled_matrix

    def update_metadata(self, cols: List = [], sdtype: str = 'numerical'):

        for c in cols:
            self.metadata.update_column(
                column_name=c,
                sdtype=sdtype,
                computer_representation='Float'
            )

    def train_synthesizer(self, Model: CopulaGANSynthesizer | CTGANSynthesizer | GaussianCopulaSynthesizer, model_name: str = '', force: bool = False, params: Dict[str, Any] = {}):

        if model_name != '':
            model_name = str(self.models_dir / model_name)

        if not os.path.exists(model_name) or force:
            print(
                "Training model .. this may take several minutes or hours depending on the system.")
            synthesizer = Model(**params)
            synthesizer.fit(self.df)

        else:
            print("Retrieving existing model")
            synthesizer = Model.load(
                filepath=model_name)

        synthesizer.save(filepath=model_name)

        return synthesizer

    # Train Gaussian Copula
    def train_copula_synthesizer(self, model_name: str = '', force=False,  enforce_min_max_values=True, enforce_rounding=True, numerical_distributions: Optional[Dict] = None, default_distribution: str = 'norm'):

        params = {
            "metadata": self.metadata,
            "enforce_min_max_values": enforce_min_max_values,
            "enforce_rounding": enforce_rounding,
            "numerical_distributions": numerical_distributions,
            "default_distribution": default_distribution
        }

        synthesizer: GaussianCopulaSynthesizer = self.train_synthesizer(
            Model=GaussianCopulaSynthesizer, model_name=model_name, force=force, params=params)

        return synthesizer

    def train_ctgan_synthesizer(self, model_name: str = '', epochs=500, enforce_rounding=True, enforce_min_max_values=True, verbose=True, force=False, batch_size: int = 512):

        pac: int = 10

        for diviser in (10, 8):
            if batch_size % diviser == 0:
                pac = diviser
                break

        params = {
            "metadata": self.metadata,
            "epochs": epochs,
            "enforce_rounding": enforce_rounding,
            "enforce_min_max_values": enforce_min_max_values,
            "verbose": verbose,
            "cuda": torch.cuda.is_available(),
            # hyperparameters
            "batch_size": batch_size,
            "pac": pac
        }

        synthesizer: CTGANSynthesizer = self.train_synthesizer(
            Model=CTGANSynthesizer, model_name=model_name, force=force, params=params)

        return synthesizer

    def train_var_autoencoder(self, model_name: str = 'bmk2018_var_autoencoder.pkl'):
        # todo:
        # create autoencoder model with demographics
        pass

    def train_copula_gan_synthesizer(self, model_name: str = '', enforce_min_max_value: bool = True, enforce_rounding: bool = False, numerical_distributions: Dict[str, Any] = {}, epochs: int = 500, verbose: bool = True, force: bool = False, batch_size: int = 512):

        pac: int = 10

        for diviser in (10, 8):
            if batch_size % diviser == 0:
                pac = diviser
                break

        params = {

            "metadata": self.metadata,  # required
            "enforce_min_max_values": enforce_min_max_value,
            "enforce_rounding": enforce_rounding,
            "numerical_distributions": numerical_distributions,
            "epochs": epochs,
            "verbose": verbose,
            "batch_size": batch_size,
            "pac": pac
        }

        synthesizer: CopulaGANSynthesizer = self.train_synthesizer(
            Model=CopulaGANSynthesizer, model_name=model_name, force=force, params=params)

        return synthesizer

#     def tune_hyperparameters(self,
#                              epochs: List = [500, 700, 1000],
#                              batch_size: List = [128, 256],
#                              generator_dim: List = [
#                                  (64, 64), (128, 128), (256, 256)],
#                              discriminator_dim: List = [
#                                  (64, 64), (128, 128), (256, 256)],
#                              discriminator_decay=[0, 0.0001, 0.001, 0.01, 0.1]
#                              ):

#         space = {
#             'epochs': hp.choice('epochs', epochs),
#             # this goes from e^-10 to e^-1
#             'generator_lr': hp.loguniform('generator_lr', -10, -1),
#             'discriminator_lr': hp.loguniform('discriminator_lr', -10, -1),
#             'generator_dim': hp.choice('generator_dim', generator_dim),
#             'discriminator_dim': hp.choice('discriminator_dim', discriminator_dim),
#             'discriminator_decay': hp.choice('discriminator_decay', discriminator_decay),
#             'batch_size': hp.choice('batch_size', batch_size)
#         }

#         trials = Trials()
#         best = fmin(fn=self.objective,
#                     space=space,
#                     algo=tpe.suggest,
#                     max_evals=50,
#                     trials=trials)

#         print("Best hyperparameters:", best)
#         return best

#     def objective(self, params):
#         model = CTGANSynthesizer(metadata=self.metadata,
#                                  generator_dim=params['generator_dim'],
#                                  discriminator_dim=params['discriminator_dim'],
#                                  generator_lr=params['generator_lr'],
#                                  discriminator_lr=params['discriminator_lr'],
#                                  batch_size=int(params['batch_size']),
#                                  discriminator_decay=params['discriminator_decay'],
#                                  epochs=int(params['epochs']),
#                                  cuda=torch.cuda.is_available(),
#                                  pac=8)

#         model.fit(self.df)

#         synthetic_data = model.sample(len(self.df))

#         ## need evaluation method
#         score = None

#         return {'loss': -score, 'status': STATUS_OK}

    def generate_synthetic_sample(self, synthesizer, num_rows: int = 100):
        return synthesizer.sample(num_rows=num_rows)

    def run_diagnostic(self, synthetic_data):
        diagnostic = run_diagnostic(
            real_data=self.df,
            synthetic_data=synthetic_data,
            metadata=self.metadata
        )

        return diagnostic

    def run_evaluation(self, synthetic_data):

        quality = evaluate_quality(
            real_data=self.df,
            synthetic_data=synthetic_data,
            metadata=self.metadata
        )

        return quality

    def visualize_evaluation(self, synthetic_data, data_name: str, demo_cols: List):
        from table_evaluator import TableEvaluator

        eval_folder = self.figures_dir / 'evaluations' / data_name

        eval_folder.mkdir(parents=True, exist_ok=True)

        te = TableEvaluator(self.df, synthetic_data, cat_cols=demo_cols)
        te.visual_evaluation(save_dir=eval_folder)

        plot_list = [
            eval_folder/'mean_std.png',
            eval_folder/'cumsums.png',
            eval_folder/'distributions.png',
            eval_folder/'correlation_difference.png',
            eval_folder/'pca.png'
        ]

        return plot_list

    def visualize_cumsum(self, synthetic_data, column_names: str | List, fig_name: Optional[str] = ''):
        if isinstance(column_names, str):
            column_names = [column_names]

        total_plots = len(column_names)
        rows = math.ceil(math.sqrt(total_plots))
        cols = math.ceil(total_plots / rows)

        fig = make_subplots(rows=rows, cols=cols, subplot_titles=column_names)

        for i, c in enumerate(column_names):
            cumsum_real = self.df[c].cumsum()
            cumsum_synthetic = synthetic_data[c].cumsum()

            # Line plot for real data
            real_trace = go.Scatter(
                x=cumsum_real.index, y=cumsum_real, mode='lines', name='Real Data')

            # Line plot for synthetic data
            synthetic_trace = go.Scatter(
                x=cumsum_synthetic.index, y=cumsum_synthetic, mode='lines', name='Synthetic Data')

            fig.add_trace(real_trace, row=(i // cols) + 1, col=(i % cols) + 1)
            fig.add_trace(synthetic_trace, row=(
                i // cols) + 1, col=(i % cols) + 1)

        fig.update_layout(height=rows * 300, width=cols * 300,
                          title_text="Cumulative Sums of feature")

        if fig_name != '':
            fig_name = f"{self.figures_dir}/{fig_name}"
            pio.write_image(fig, format="png", file=fig_name)

        return fig

    def visualize_data(self, synthetic_data, column_names: str | List, fig_name: Optional[str] = '', data_name: str = ''):

        eval_folder = self.figures_dir / 'evaluations' / data_name
        eval_folder.mkdir(parents=True, exist_ok=True)

        if isinstance(column_names, str):
            column_names = [column_names]

        total_plots = len(column_names)
        rows = math.ceil(math.sqrt(total_plots))
        cols = math.ceil(total_plots / rows)

        fig = make_subplots(rows=rows, cols=cols, subplot_titles=column_names)

        for i, c in enumerate(column_names):
            column_fig = get_column_plot(
                real_data=self.df,
                synthetic_data=synthetic_data,
                column_name=c,
                metadata=self.metadata
            )

            for trace in column_fig.data:
                fig.add_trace(
                    trace,
                    row=(i // cols) + 1,
                    col=(i % cols) + 1
                )

        fig.update_layout(height=rows * 300, width=cols *
                          300, title_text="Column Plots")

        if fig_name != '':
            fig_name = f"{eval_folder}/{fig_name}"
            print("Fig name: ", fig_name)
            pio.write_image(fig, format="png", file=fig_name)

        return fig

    def get_corr_diff(self, corr_a, corr_b):

        corr_diff = corr_a - corr_b
        return corr_diff
