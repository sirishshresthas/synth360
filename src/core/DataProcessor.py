import os
import pandas as pd
from typing import List
from src.core.utilities import globals
from typing import Any, Optional
from src.core import LoadBlob


class DataProcessor(object):

    def __init__(self, data: str = "BMK_2018.csv"):
        self.data_name: str = data
        self._rater_col_name = None
        self._id_key = None
        self._demo_cols = []
        self._items_cols = []
        self.df: pd.DataFrame = None

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
        blob = LoadBlob(self.data_name)
        df = blob.load_data()

        if self.demo_cols and self.items_cols:
            df = df.loc[:, self.demo_cols + self.items_cols]

        return df

    def get_data(self):
        self.df = self._load_data()
        return self.df

    def pivot_rater_data(self):
        df_self = self.df[self.df[self.rater_col_name].isin(['self', 'Self'])]
        df_self = self._remove_duplicates(
            df=df_self, id_key=self.id_key, items_only=True)
        df_others = self.df[~self.df[self.rater_col_name].isin(['self', 'Self'])]
        
        df_others = df_others.groupby([self.id_key, self.rater_col_name])[self.items_cols[1:]].mean()


        df_others = df_others.unstack(level=self.rater_col_name)
        df_others.columns = ['{}_{}'.format(col[0], col[1])
                             for col in df_others.columns]

        df = pd.merge(df_self, df_others, on=self.id_key, how="left")

        df.columns = [c.replace(' ', '_') for c in df.columns]

        return df

    def _remove_duplicates(self, df: pd.DataFrame, id_key: str, items_only: bool = False, demo_cols_only: bool = False):
        # Remove duplicates based on id_key, keeping the last occurrence
        df = df.drop_duplicates(subset=id_key, keep="last")
        
        # Adjust DataFrame based on flags
        if items_only and demo_cols_only:
            # If both flags are True, filter to include both items and demo columns
            df = df.loc[:, self.demo_cols + self.items_cols]

        elif items_only:
            # If items_only is True, filter to include only items columns
            df = df.loc[:, self.items_cols]

        elif demo_cols_only:
            # If demo_cols_only is True, filter to include only demo columns
            df = df.loc[:, self.demo_cols]

        print(df.columns)
        return df




        return df

    def filter_data_with_all_raters(self, remove_other_raters: bool = True, remove_rater_list: List = ['Superior', 'Other']) -> pd.DataFrame:

        unique_rater_counts = self.df.groupby(
            self.id_key)[self.rater_col_name].nunique()

        # Get the `ESI_Key` of self raters with more than 4 raters
        esi_full_raters = unique_rater_counts[unique_rater_counts >= 4]

        self.df = self.df[self.df[self.id_key].isin(esi_full_raters.index)]

        if remove_other_raters:
            self.df = self.df[~self.df[self.rater_col_name].isin(
                remove_rater_list)]

        return self.df

    def median_rater_counts(self, id_key: str = "ESI_Key", rater_col_name: str = "RaterType"):

        if not self.rater_col_name:
            self.rater_col_name = rater_col_name

        if self.rater_col_name not in self.df.columns:
            raise ValueError(
                f"The rater column {self.rater_col_name} does not exist in the data.")

        if not self.id_key:
            self.id_key = id_key

        if self.id_key not in self.df.columns:
            raise ValueError(
                f"The rater column {self.id_key} does not exist in the data.")

        filtered_df = self.df[~self.df[self.rater_col_name].isin(
            ['self', 'Self'])]

        rater_counts = filtered_df.groupby(
            [self.id_key, self.rater_col_name]).size().unstack(fill_value=0)

        # Calculate the median number of each rater type across all 'self' raters
        return rater_counts.median()
