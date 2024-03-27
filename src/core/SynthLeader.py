import os
import math
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, List, Dict, Optional

from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer, CopulaGANSynthesizer
from sdv.evaluation.single_table import run_diagnostic, evaluate_quality
from sdv.evaluation.single_table import get_column_plot, DiagnosticReport, QualityReport

import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from table_evaluator import TableEvaluator
# from hyperopt import hp, STATUS_OK, fmin, tpe, Trials


class SynthLeader(object):
    """
    A class to encapsulate functionality for leading synthetic data generation efforts using various synthesizers.

    Attributes:
        df (pd.DataFrame): The original dataframe to synthesize.
        name (str): An optional name for the instance, useful for identification.
        data_dir (Path): The base directory for data-related storage.
        models_dir (Optional[Path]): Directory to store trained models.
        figures_dir (Optional[Path]): Directory to store figures.
        metadata_dir (Optional[Path]): Directory to store metadata.

    Methods:
        create_metadata(): Creates and validates metadata from the dataframe.
        update_metadata(cols, sdtype): Updates the metadata for specified columns.
        train_synthesizer(): Trains a specified synthesizer model with optional parameters.
        train_copula_synthesizer(): Convenience method to train a Gaussian Copula Synthesizer.
        train_ctgan_synthesizer(): Convenience method to train a CTGAN Synthesizer.
        train_copula_gan_synthesizer(): Convenience method to train a CopulaGAN Synthesizer.
        generate_synthetic_sample(): Generates a synthetic sample from a trained model.
        run_diagnostic(): Runs diagnostics to compare real and synthetic data.
        run_evaluation(): Evaluates the quality of synthetic data against real data.
        visualize_evaluation(): Visualizes evaluation metrics and saves the plots.
        visualize_cumsum(): Visualizes cumulative sums for given columns.
        visualize_data(): Generates and visualizes column plots for real vs. synthetic data.
        get_corr_diff(): Calculates the difference between two correlation matrices.
    """

    def __init__(self, df: pd.DataFrame, name: str = '') -> None:
        """
        Initializes the SynthLeader instance with a dataframe and optional name.

        Parameters:
            df (pd.DataFrame): The dataframe to be synthesized.
            name (str): An optional name for the instance.
        """
        self.df = df
        self.name = name

        self.data_dir = Path("./data/")
        self.models_dir: Optional[Path] = None
        self.figures_dir: Optional[Path] = None
        self.metadata_dir: Optional[Path] = None

        self.sub_dirs = ['models', 'figures', 'metadata']

        for sub_dir in self.sub_dirs:
            dir_path = self.data_dir / sub_dir
            dir_path.mkdir(parents=True, exist_ok=True)
            setattr(self, f"{sub_dir}_dir", dir_path)

        self._metadata: SingleTableMetadata = self.create_metadata()
        print(
            f"GPU: {torch.cuda.is_available()}")

    def __repr__(self) -> str:
        """
        Provides a string representation of the SynthLeader instance.

        Returns:
            str: A string that represents the SynthLeader instance.
        """
        return f"{type(self.__class__.__name__)} {self.name}"

    @property
    def metadata(self) -> SingleTableMetadata:
        """A property that returns the metadata of the DataFrame."""
        return self._metadata

    @metadata.setter
    def metadata(self, metadata: SingleTableMetadata) -> None:
        """
        Sets the metadata for the DataFrame.

        Parameters:
            metadata (SingleTableMetadata): The metadata to set for the DataFrame.
        """
        self._metadata = metadata

    def create_metadata(self) -> SingleTableMetadata:
        """
        Creates and returns metadata from the DataFrame.

        Returns:
            SingleTableMetadata: The generated metadata from the DataFrame.
        """
        metadata: SingleTableMetadata = SingleTableMetadata()

        metadata.detect_from_dataframe(self.df)
        metadata.validate()
        # metadata.save_to_json(filepath=self.metadata_dir / "metadata.json")

        return metadata

    def _get_numeric_cols(self) -> List[str]:
        """
        Identifies and returns a list of numeric columns in the DataFrame.

        Returns:
            List[str]: A list of column names that are numeric.
        """
        cols = self.df.select_dtypes(
            include=[np.number]).columns.to_list()  # type: ignore
        return cols

    def generate_corr_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates a correlation matrix for the numeric columns of the DataFrame.

        Parameters:
            df (pd.DataFrame): The DataFrame for which to generate the correlation matrix.

        Returns:
            pd.DataFrame: The correlation matrix.
        """
        matrix = df[self._get_numeric_cols()].corr()

        return matrix

    def style_correlation_matrix(self, matrix, style="coolwarm") -> pd.DataFrame:
        """
        Applies styling to a correlation matrix DataFrame.

        Parameters:
            matrix (pd.DataFrame): The correlation matrix to style.
            style (str): The matplotlib colormap to use for background gradient.

        Returns:
            pd.Styler: The styled correlation matrix.
        """
        styled_matrix = matrix \
            .style \
            .background_gradient(cmap=style) \
            .format(precision=3)

        return styled_matrix

    def update_metadata(self, cols: List = [], sdtype: str = 'numerical') -> None:
        """
        Updates metadata for specified columns with a given data type.

        Parameters:
            cols (List[str]): The columns to update.
            sdtype (str): The synthetic data type to apply.
        """

        for c in cols:
            self.metadata.update_column(
                column_name=c,
                sdtype=sdtype,
                computer_representation='Float'
            )

    def train_synthesizer(self, Model: CopulaGANSynthesizer | CTGANSynthesizer | GaussianCopulaSynthesizer, model_name: str = '', force: bool = False, params: Dict[str, Any] = {}) -> CTGANSynthesizer | CopulaGANSynthesizer | GaussianCopulaSynthesizer:
        """
        Trains a specified synthesizer model or loads an existing one based on the provided parameters.
        If a model with the given name exists and `force` is not True, the existing model will be loaded instead.

        Parameters:
            Model: The synthesizer model class to be trained.
            model_name: The name of the model file to save or load.
            force: If True, forces retraining of the model even if it already exists.
            params: Parameters to pass to the synthesizer model during initialization.

        Returns:
            The trained or loaded synthesizer model.

        Raises:
            ValueError: If `models_dir` is not set, indicating where to save the figures.

        """

        if self.models_dir is None:
            raise ValueError(
                "models_dir is not set. Please specify a directory for saving models.")

        model_path = self.models_dir / model_name if model_name else None

        if model_path and model_path.exists() and not force:
            print("Retrieving existing model")
            synthesizer = Model.load(filepath=model_name)

        else:
            if model_path:
                print(
                    "Training model .. this may take several minutes or hours depending on the system.")
                synthesizer = Model(**params) #type: ignore
                synthesizer.fit(self.df)
                synthesizer.save(filepath=str(model_path))

            else:
                raise ValueError(
                    "Model name must be provided to train or load a model.")

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
            Model=GaussianCopulaSynthesizer, model_name=model_name, force=force, params=params)  # type: ignore

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
            Model=CTGANSynthesizer, model_name=model_name, force=force, params=params)  # type: ignore

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
            Model=CopulaGANSynthesizer, model_name=model_name, force=force, params=params)  # type: ignore

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

    def generate_synthetic_sample(self, synthesizer: CTGANSynthesizer | CopulaGANSynthesizer | GaussianCopulaSynthesizer, num_rows: int = 100) -> pd.DataFrame:
        """
        Generates a synthetic sample using a trained synthesizer model.

        Parameters:
            synthesizer (Union[CTGANSynthesizer, CopulaGANSynthesizer, GaussianCopulaSynthesizer]): The synthesizer model to use for sample generation.
            num_rows (int): The number of rows to generate.

        Returns:
            pd.DataFrame: The generated synthetic sample.
        """
        return synthesizer.sample(num_rows=num_rows)

    def run_diagnostic(self, synthetic_data: pd.DataFrame) -> DiagnosticReport:
        """
        Runs diagnostics to compare the real and synthetic data.

        Parameters:
            synthetic_data (pd.DataFrame): The synthetic data to compare against the real data.

        Returns:
            DiagnosticReport: Diagnostic results.
        """

        diagnostic = run_diagnostic(
            real_data=self.df,
            synthetic_data=synthetic_data,
            metadata=self.metadata
        )

        return diagnostic

    def run_evaluation(self, synthetic_data) -> QualityReport:
        """
        Evaluates the quality of the synthetic data compared to the real data.

        Parameters:
            synthetic_data (pd.DataFrame): The synthetic data to evaluate.

        Returns:
            QualityReport: The quality score of the synthetic data.
        """
        quality = evaluate_quality(
            real_data=self.df,
            synthetic_data=synthetic_data,
            metadata=self.metadata
        )

        return quality

    def visualize_evaluation(self, synthetic_data: pd.DataFrame, data_name: str, demo_cols: List) -> List[Path]:
        """
        Generates and saves visual evaluations of the synthetic data compared to the real data.

        This function uses the `TableEvaluator` to create a series of plots that compare various
        statistics and distributions between the real and synthetic data. It saves these plots in a
        specific folder structure within the `figures_dir`.

        Parameters:
            synthetic_data (pd.DataFrame): The synthetic dataset to be evaluated.
            data_name (str): A name identifier for the dataset, used to create a subdirectory for saving plots.
            demo_cols (List[str]): A list of column names to be treated as categorical during the evaluation.

        Returns:
            List[Path]: A list of paths to the saved plot images.

        Raises:
            ValueError: If `figures_dir` is not set, indicating where to save the figures.
        """
        if self.figures_dir is None:
            raise ValueError(
                "figures_dir is not set. Please specify a directory for saving figures.")

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

    def visualize_data(self, synthetic_data: pd.DataFrame, column_names: str | List, fig_name: Optional[str] = '', data_name: str = '') -> go.Figure:
        """
        Visualizes data comparisons between real and synthetic datasets for specified columns.

        This function creates and saves a series of plots comparing real and synthetic data distributions
        for each specified column name. Plots are saved to the specified figure name if provided.

        Parameters:
            synthetic_data (pd.DataFrame): The synthetic dataset for comparison.
            column_names (Union[str, List[str]]): Column name(s) to visualize. Can be a single column name as a string or a list of names.
            fig_name (Optional[str]): The file name to save the figure to. If not provided, the figure is not saved.
            data_name (str): The name of the data, used to organize saved plots into subdirectories.

        Returns:
            go.Figure: The Plotly figure object containing the generated plots.

        Raises:
            ValueError: If `figures_dir` is not set, indicating where to save the figures.
        """
        if self.figures_dir is None:
            raise ValueError(
                "figures_dir is not set. Please specify a directory for saving figures.")

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
                fig.add_trace(trace, row=(i // cols) + 1, col=(i % cols) + 1)

        fig.update_layout(height=rows * 300, width=cols *
                          300, title_text="Column Plots")

        if fig_name:
            fig_name = f"{eval_folder}/{fig_name}"
            pio.write_image(fig, format="png", file=fig_name)

        return fig

    def get_corr_diff(self, corr_a: pd.DataFrame, corr_b: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the difference between two correlation matrices.

        This method is useful for comparing the correlation matrices of real and synthetic data
        to understand how closely the synthetic data mimics the real data in terms of variable interrelationships.

        Parameters:
            corr_a (pd.DataFrame): The first correlation matrix.
            corr_b (pd.DataFrame): The second correlation matrix to subtract from the first.

        Returns:
            pd.DataFrame: A DataFrame representing the difference between the two correlation matrices.
        """
        corr_diff = corr_a - corr_b
        return corr_diff
