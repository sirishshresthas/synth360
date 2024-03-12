import pandas as pd
from typing import List
from src.core import LoadBlob
from scipy.stats import entropy
import numpy as np


class DataProcessor(object):

    def __init__(self, data: str = "BMK_2018.csv", rater_col_name: str = "RaterType", id_key: str = "ESI_Key"):
        self.data_name: str = data
        self._rater_col_name: str = rater_col_name
        self._id_key: str = id_key
        self._demo_cols: List = []
        self._items_cols: List = []
        self.df: pd.DataFrame = pd.DataFrame()

    def __repr__(self):
        """
        Provides a string representation of the DataProcessor instance, including its class type and blob name.
        """
        return f"{type(self.__class__.__name__)} {self.data_name}"

    @property
    def rater_col_name(self) -> str:
        return self._rater_col_name

    @rater_col_name.setter
    def rater_col_name(self, name: str = "RaterType") -> None:
        self._rater_col_name = name

    @property
    def id_key(self) -> str:
        return self._id_key

    @id_key.setter
    def id_key(self, id_name: str = "ESI_Key") -> None:
        self._id_key = id_name

    @property
    def demo_cols(self) -> List:
        return self._demo_cols

    @demo_cols.setter
    def demo_cols(self, cols: List = []) -> None:
        self._demo_cols = cols

    @property
    def items_cols(self) -> List:
        return self._items_cols

    @items_cols.setter
    def items_cols(self, cols: List = []) -> None:
        self._items_cols = cols

    def _load_data(self):
        self.blob = LoadBlob(self.data_name)
        df = self.blob.load_data()

        df[self.rater_col_name] = df[self.rater_col_name].str.lower()

        if self.demo_cols and self.items_cols:
            df = df.loc[:, self.demo_cols + self.items_cols]

        return df

    def get_data(self):
        self.df = self._load_data()
        return self.df

    def save_data(self, df: pd.DataFrame, name: str):
        self.blob.upload_data(df, name)

    def pivot_rater_data(self, skip=2):
        df_self = self.df[self.df[self.rater_col_name] == 'self']
        df_self = self._remove_duplicates(
            df=df_self, id_key=self.id_key, items_only=True)
        df_others = self.df[self.df[self.rater_col_name] != 'self']

        df_others = df_others.groupby([self.id_key, self.rater_col_name])[
            self.items_cols[skip:]].mean()

        df_others = df_others.unstack(level=self.rater_col_name)
        df_others.columns = ['{}_{}'.format(col[0], col[1])
                             for col in df_others.columns]

        df = pd.merge(df_self, df_others, on=self.id_key, how="left")

        df.columns = [c.replace(' ', '_') for c in df.columns]

        return df

    def _remove_duplicates(self, df: pd.DataFrame, id_key: str = 'ESI_Key', items_only: bool = False, demo_cols_only: bool = False):
        # Remove duplicates based on id_key, keeping the last occurrence
        df = df.drop_duplicates(subset=id_key, keep="last")

        if items_only:
            if not self.items_cols:
                raise ValueError(f"Items columns list is empty")
            # If items_only is True, filter to include only items columns
            df = df.loc[:, self.items_cols]

        elif demo_cols_only:
            if not self.demo_cols:
                raise ValueError(f"Demographic columns list is empty")
            # If demo_cols_only is True, filter to include only demo columns
            df = df.loc[:, self.demo_cols]

        if items_only and demo_cols_only:
            # If both flags are True, filter to include both items and demo columns
            df = df.loc[:, self.demo_cols + self.items_cols]

        return df

    def filter_data_with_all_raters(self, remove_other_raters: bool = True, remove_rater_list: List = ['superior', 'other']) -> pd.DataFrame:

        unique_rater_counts = self.df.groupby(
            self.id_key)[self.rater_col_name].nunique()

        # Get the `ESI_Key` of self raters with more than 4 raters
        esi_full_raters = unique_rater_counts[unique_rater_counts >= 4]

        self.df = self.df[self.df[self.id_key].isin(esi_full_raters.index)]

        if remove_other_raters:
            self.df = self.df[~self.df[self.rater_col_name].isin(
                remove_rater_list)]

        return self.df

    def median_rater_counts(self):

        filtered_df = self.df[self.df[self.rater_col_name] != 'self']

        rater_counts = filtered_df.groupby(
            [self.id_key, self.rater_col_name]).size().unstack(fill_value=0)

        # Calculate the median number of each rater type across all 'self' raters
        return rater_counts.median()

    def calculate_statistics(self, data):
        stats = {
            'mean': pd.Series(data.mean(numeric_only=True)),
            'std': pd.Series(data.std(numeric_only=True)),
            'kurtosis': pd.Series(data.kurtosis(numeric_only=True)), # Fisher's definition (normal ==> 0.0)
            'skewness': pd.Series(data.skew(numeric_only=True)),
            'max': pd.Series(data.max()),
            'min': pd.Series(data.min()),
            'q1': pd.Series(data.quantile(0.25)),
            'q3': pd.Series(data.quantile(0.75))
        }

        # entropy_values = {}
        # for column_name in data.columns:
        #     entropy_values[column_name] = entropy(data[column_name].value_counts(normalize=True), base=2)
        
        # stats['entropy'] = pd.Series(entropy_values)

        # gini_values = {}
        # for column_name in data.columns: 
        #     gini_values[column_name] = self.gini_coefficient(data[column_name])

        # stats['gini'] = pd.Series(gini_values)
        return stats

    def gini_coefficient(self, data):
        """Calculate the Gini coefficient of a numpy array."""
        data = np.sort(data)
        n = len(data)
        index = np.arange(1, n+1)

        return (np.sum((2 * index - n - 1) * data)) / (n * np.sum(data))

    def compare_datasets(self, real_data, synthetic_data, columns):
        real_stats = self.calculate_statistics(real_data[columns])
        synthetic_stats = self.calculate_statistics(synthetic_data[columns])

        # real_stats['gini'] = self.gini_coefficient(real_data[columns])
        # synthetic_stats['gini'] = self.gini_coefficient(synthetic_data[columns])

        differences = {stat: real_stats[stat] -
                       synthetic_stats[stat] for stat in real_stats}
        
        
        stats = pd.concat([
            pd.DataFrame(real_stats),
            pd.DataFrame(synthetic_stats),
            pd.DataFrame(differences)
        ], keys=['Real', 'Synthetic', 'Differences'], axis=1)

        return stats
