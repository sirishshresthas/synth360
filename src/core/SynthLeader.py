import pandas as pd
from typing import Any, List, Dict
import numpy as np
import os
import torch
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer
from sdv.evaluation.single_table import run_diagnostic, evaluate_quality
from src.core.utilities import globals


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
        cols = self.df.select_dtypes(include=[np.number]).columns.to_list() #type: ignore
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

    # Train Gaussian Copula
    def train_copula_synthesizer(self, model_name: str = '', force = False,  enforce_min_max_values=True, enforce_rounding=True, numerical_distributions={}, default_distribution: str='norm'):

        params = {
            "metadata": self.metadata,
            "enforce_min_max_values": enforce_min_max_values,
            "enforce_rounding": enforce_rounding,
            "numerical_distributions": numerical_distributions,
            "default_distribution": default_distribution
        }

        if model_name:
            model_name = str(globals.DATA_DIR / model_name)

        if os.path.exists(model_name):
            if force:
                synthesizer = GaussianCopulaSynthesizer(**params)

            else:

                synthesizer = GaussianCopulaSynthesizer.load(filepath=model_name)

        else:
            synthesizer = GaussianCopulaSynthesizer(**params)

        synthesizer.fit(self.df)

        if model_name:
            synthesizer.save(filepath=model_name)

        return synthesizer

    
    def train_ctgan_synthesizer(self, model_name: str = '', epochs=500, enforce_rounding=True, enforce_min_max_values=True, verbose=False, force=False):

        params = {
            "metadata": self.metadata,
            "epochs": epochs, 
            "enforce_rounding": enforce_rounding, 
            "enforce_min_max_values": enforce_min_max_values, 
            "verbose": verbose,
            "cuda": torch.cuda.is_available()
        }

        if model_name:
            model_name = str(globals.DATA_DIR / model_name)

        if os.path.exists(model_name):

            if force:
                synthesizer = CTGANSynthesizer(**params)
            else:
                synthesizer = CTGANSynthesizer.load(filepath=model_name)

        else:
            synthesizer = CTGANSynthesizer(**params)

        synthesizer.fit(self.df)

        if model_name:
            synthesizer.save(filepath=model_name)

        return synthesizer
    
    
    def train_var_autoencoder(self, model_name: str = 'bmk2018_var_autoencoder.pkl'):
        ## todo:
        ## create autoencoder model with demographics
        pass
    
    def train_copula_gan_synthesizer(self, model_name: str = ''): 
        
        ## todo: 
        ## create copula GAN synthesizer
        pass
        
    def gan_hyperparameter_tuning(self): 
        ## todo: 
        ## create grid-search hyperparameter tuning
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

    def get_corr_diff(self, corr_a, corr_b):

        corr_diff = corr_a - corr_b
        return corr_diff
