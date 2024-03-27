import pandas as pd
from typing import List
from src.core import LoadBlob
import pingouin as pg
import gc

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

    def save_data(self, df: pd.DataFrame, name: str, folder: str):
        self.blob.upload_data(df, name, folder)

    def pivot_rater_data(self):
        df_self = self.df[self.df[self.rater_col_name] == 'self']

        df_self = self._remove_duplicates(
            df=df_self, id_key=self.id_key, items_only=True)

        df_others = self.df[self.df[self.rater_col_name] != 'self']

        # df_others = df_others.groupby([self.id_key, self.rater_col_name])[
        #     self.items_cols[skip:]].mean()
        df_others = df_others.groupby([self.id_key, self.rater_col_name])[
            self.items_cols].mean()

        df_others = df_others.unstack(level=self.rater_col_name)
        df_others.columns = ['{}_{}'.format(col[0], col[1])
                             for col in df_others.columns]
        
        print(df_others.columns)

        df = pd.merge(df_self, df_others, on=self.id_key, how="left")

        df.columns = [c.replace(' ', '_') for c in df.columns]

        return df

    def _remove_duplicates(self, df: pd.DataFrame, id_key: str = 'ESI_Key', items_only: bool = False, demo_cols_only: bool = False):
        # Remove duplicates based on id_key, keeping the last occurrence
        df = df.drop_duplicates(subset=id_key, keep="last")

        # if items_only:
        #     if not self.items_cols:
        #         raise ValueError(f"Items columns list is empty")
        #     # If items_only is True, filter to include only items columns
        #     df = df.loc[:, self.items_cols]

        # elif demo_cols_only:
        #     if not self.demo_cols:
        #         raise ValueError(f"Demographic columns list is empty")
        #     # If demo_cols_only is True, filter to include only demo columns
        #     df = df.loc[:, self.demo_cols]

        # if items_only and demo_cols_only:
        #     # If both flags are True, filter to include both items and demo columns
        #     df = df.loc[:, self.demo_cols + self.items_cols]

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
            'Min': round(pd.Series(data.min()),3),
            'Max': round(pd.Series(data.max()),3),
            'Median': round(pd.Series(data.median()),3),
            'Mean': round(pd.Series(data.mean(numeric_only=True)),3),
            'SD': round(pd.Series(data.std(numeric_only=True)),3),
            'kurtosis': round(pd.Series(data.kurtosis(numeric_only=True)),3),
            'skewness': round(pd.Series(data.skew(numeric_only=True)),3)
        }

        return stats
    
    def calculate_demographic_statistics(self, data, column, categories):
        
        ## dataframe for mean values
        df_mean = data[data[column] \
                    .isin(categories)] \
                    .groupby(column)[self.items_cols] \
                    .mean() \
                    .round(3).T
        
        ## dataframe for std values
        df_std = data[data[column] \
                    .isin(categories)] \
                    .groupby(column)[self.items_cols] \
                    .std() \
                    .round(3).T

        ## cohen's d requires observation, so extracting observations from the data
        cat1_data = data[data[column] == categories[0]][self.items_cols]
        cat2_data = data[data[column] == categories[1]][self.items_cols]

        # Calculate Cohen's d for each column
        cohens_d_values = {col: round(pg.compute_effsize(cat1_data[col], cat2_data[col], eftype='cohen'),3) \
                           for col in self.items_cols}

        df_cohens_d = pd.DataFrame(cohens_d_values, index=[0])
        df_cohens_d.index = ['Cohen\'s d']
        df_cohens_d = df_cohens_d.transpose()

        combined_mean_std = df_mean.astype(str) + ' (' + df_std.astype(str) + ')'
        combined_mean_std = combined_mean_std.reset_index()

        final_df = pd.merge(combined_mean_std, df_cohens_d, right_index=True, left_on='index', how='inner')
        final_df = final_df.set_index('index')
        final_df.index.name = None
        
        del(df_cohens_d)
        del(combined_mean_std)
        del(cat1_data)
        del(cat2_data)
        del(df_mean)
        del(df_std)
        
        gc.collect()

        return final_df
        
        

    def compare_datasets(self, real_data, synthetic_data, columns):
        real_stats = pd.DataFrame(self.calculate_statistics(real_data[columns]))
        synthetic_stats = pd.DataFrame(self.calculate_statistics(synthetic_data[columns]))

        cd = {}

        for column in columns: 
            cohens_d = pg.compute_effsize(
                real_data[column], synthetic_data[column], paired=False, eftype='cohen')
            cd[column] = round(cohens_d, 3)

        synthetic_stats["Cohen's d"] = cd

        synthetic_stats.index = synthetic_stats.index.map(lambda x: x + "_synth")
        real_stats.index = real_stats.index.map(lambda x: x + "_real")

        stats = pd.concat([real_stats, synthetic_stats]
                              ).sort_index()

        return stats, real_stats, synthetic_stats

    
    def compare_demographics(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, column: str, categories: List):
        
            
        df_real = self.calculate_demographic_statistics(real_data, column, categories)
        df_real.columns = df_real.columns.map(lambda x: x + "_real")

        df_synth = self.calculate_demographic_statistics(synthetic_data, column, categories)
        df_synth.columns = df_synth.columns.map(lambda x: x + "_synth")

        df = pd.merge(df_real, df_synth, left_index=True, right_index=True, how='inner')
        return df
        