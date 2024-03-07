import os
import math
import torch
import numpy as np
import pandas as pd
from typing import Any, List, Dict, Optional, Union
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer, CopulaGANSynthesizer
from sdv.evaluation.single_table import run_diagnostic, evaluate_quality
from src.core.utilities import globals
from sdv.evaluation.single_table import get_column_plot
import plotly.graph_objs as go
from plotly.subplots import make_subplots


class SynthLeader(object):

    def __init__(self, df: pd.DataFrame, name: str = ''):
        self.df = df
        self._metadata: SingleTableMetadata = self.create_metadata()
        self.name = name
        print(f"Cuda: {torch.cuda.is_available()}")

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

    def _get_numeric_cols(self) -> List[str]:
        cols = self.df.select_dtypes(
            include=[np.number]).columns.to_list()  # type: ignore
        return cols

    def generate_corr_matrix(self, df: pd.DataFrame):

        matrix = df[self._get_numeric_cols()].corr()

        return matrix

    def style_correlation_matrix(self, matrix, style="coolwarm"):

        return matrix.style.background_gradient(cmap=style)

    def create_metadata(self) -> SingleTableMetadata:
        metadata: SingleTableMetadata = SingleTableMetadata()
        metadata.detect_from_dataframe(self.df)
        metadata.validate()

        return metadata

    def update_metadata(self, cols: List = [], sdtype: str = 'numerical'):

        for c in cols:
            self.metadata.update_column(
                column_name=c,
                sdtype=sdtype,
                computer_representation='Float'
            )

    def train_synthesizer(self, Model: CopulaGANSynthesizer | CTGANSynthesizer | GaussianCopulaSynthesizer, model_name: str = '', force: bool = False, params: Dict[str, Any] = {}):
        if model_name != '':
            model_name = str(globals.DATA_DIR / model_name)

        if os.path.exists(model_name):
            if force:
                synthesizer = Model(**params)

            else:

                synthesizer = Model.load(
                    filepath=model_name)

        else:
            synthesizer = Model(**params)

        synthesizer.fit(self.df)

        if model_name:
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

    def train_ctgan_synthesizer(self, model_name: str = '', epochs=500, enforce_rounding=True, enforce_min_max_values=True, verbose=True, force=False):

        params = {
            "metadata": self.metadata,
            "epochs": epochs,
            "enforce_rounding": enforce_rounding,
            "enforce_min_max_values": enforce_min_max_values,
            "verbose": verbose,
            "cuda": torch.cuda.is_available()
        }

        synthesizer: CTGANSynthesizer = self.train_synthesizer(
            Model=CTGANSynthesizer, model_name=model_name, force=force, params=params)

        return synthesizer

    def train_var_autoencoder(self, model_name: str = 'bmk2018_var_autoencoder.pkl'):
        # todo:
        # create autoencoder model with demographics
        pass

    def train_copula_gan_synthesizer(self, model_name: str = '', enforce_min_max_value: bool = True, enforce_rounding: bool = False, numerical_distributions: Dict[str, Any] = {}, epochs: int = 500, verbose: bool = True, force: bool = False):

        params = {

            "metadata": self.metadata,  # required
            "enforce_min_max_values": enforce_min_max_value,
            "enforce_rounding": enforce_rounding,
            "numerical_distributions": numerical_distributions,
            "epochs": epochs,
            "verbose": verbose
        }

        synthesizer: CopulaGANSynthesizer = self.train_synthesizer(
            Model=CopulaGANSynthesizer, model_name=model_name, force=force, params=params)

        return synthesizer

    def gan_hyperparameter_tuning(self):
        # todo:
        # create grid-search hyperparameter tuning
        pass

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

    def visualize_data(self, synthetic_data, column_names: str | List):

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

        fig.update_layout(height=rows * 300, width=cols * 300, title_text="Column Plots")
        fig.show()

    def get_corr_diff(self, corr_a, corr_b):

        corr_diff = corr_a - corr_b
        return corr_diff
