import pandas as pd
from typing import List
import numpy as np
import os
import torch
from sdv.meetadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer
from sdv.evaluation.single_table import run_diagnostic, evaluate_quality
from src.core.utilities import globals


class SynthLeader(object):

    def __init__(self, df: pd.DataFrame, name: str = None):
        self.df = df
        self._metadata = self.create_metadata()
        self.name = name
        print(f"Cuda: {torch.cuda.is_available}")

    def __repr__(self):
        """
        Provides a string representation of the SynthLeader instance, including its class type and data name.
        """
        return f"{type(self.__class__.__name__)} {self.name}"

    @property
    def metadata(self) -> Dict:
        return self._metadata

    @metadata.setter
    def metadata(self, metadata: Dict) -> None:
        self._metadata = metadata

    def _get_numeric_cols(self) -> List:
        cols = self.df.select_dtypes(includ=[np.number]).columns
        return cols

    def generate_corr_matrix(self, colorify: bool = True):

        matrix = self.df[_get_numeric_cols()].corr()

        if colorify:
            matrix.style.background_gradient(cmap="coolwarm")

        return matrix

    def create_metadata(self):
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(self.df)
        metadata.validate()

        return metadata

    def update_metadata(self, cols: List = [], sdtype: str = 'numerical'):

        for c in cols:
            metadata.update_column(
                column_name=c,
                sdtype=sdtype,
                computer_representation='Float'
            )

    # Train Gaussian Copula
    def train_copula_synthesizer(self, model_name: str = None,  enforce_min_max_values=True, enforcing_rounding=True, numerical_distributions={}, default_distribution='norm'):

        if model_name: 
            model_name = globals.DATA_DIR / model_name

        if os.path.exists(model_name): 
             if force:
                synthesizer = self.run_ctgan_synthesizer(epochs=500, enforce_rounding=True, enforce_min_max_values=True, verbose=False)
            else: 
                synthesizer = GaussianCopulaSynthesizer.load(filepath=model_name)

        else: 
            synthesizer = self._run_copula_synthesizer( 
                enforce_min_max_values=enforce_min_max_values, 
                enforcing_rounding=enforce_rounding, 
                numerical_distributions=numerical_distributions,
                default_distribution=default_distribution
            )
            

        synthesizer.fit(self.df)

        if model_name:
            synthesizer.save(filepath=model_name)

        return synthesizer


    def _run_copula_synthesizer(self): 
        synthesizer = GaussianCopulaSynthesizer(
                self.metadata,
                enforce_min_max_values=enforce_min_max_values,
                enforcing_rounding=enforcing_rounding,
                numerical_distributions=numerical_distributions,
                default_distribution=default_distribution, 
                cuda=torch.cuda.is_available()
            )

        return synthesizer

    

    def train_ctgan_synthesizer(self, model_name: str = None, epochs=500, enforce_rounding=True, enforce_min_max_values=True, verbose=False, force=False)
    
        if model_name: 
            model_name = globals.DATA_DIR / model_name


        if os.path.exists(model_name): 

            if force:
                synthesizer = self.run_ctgan_synthesizer(epochs=500, enforce_rounding=True, enforce_min_max_values=True, verbose=False)
            else: 
                synthesizer = CTGANSynthesizer.load(filepath=model_name)
        
        else: 
            synthesizer = self._run_ctgan_synthesizer(
                epochs=epochs, 
                enforce_rounding=enforce_rounding,
                enforce_min_max_values=enforce_min_max_values, 
                verbose=verbose
            )

        synthesizer.fit(self.df)

        if model_name: 
            synthesizer.save(filepath=model_name)

        return synthesizer
    


    def _run_ctgan_synthesizer(self, epochs=500, enforce_rounding=True, enforce_min_max_values=True, verbose=False): 
        synthesizer = CTGANSynthesizer(
                self.metadata, 
                epochs=epochs, 
                enforce_rounding=enforce_rounding,
                enforce_min_max_values=enforce_min_max_values, 
                verbose=verbose, 
                cuda=torch.cuda.is_available()
            )

        return synthesizer

    
    
    def generate_synthetic_sample(self, synthesizer, num_rows: int = 100):
        return synthesizer.sample(num_rows=num_rows)


    def run_diagnostic(self, synthetic_data): 
        diagnostic = run_diagnostic(
            real_data=self.df,
            synthetic_data = synthetic_data, 
            metadata = self.metadata
        )

        return diagnostic


    def run_evaluation(self, synthetic_data): 

        quality = evaluate_quality(
            real_data = self.df, 
            synthetic_data = synthetic_data, 
            metadata = self.metadata
        )

        return quality




    def get_corr_diff(self, corr_a, corr_b): 

        corr_diff = corr_a - corr_b
        return corr_diff