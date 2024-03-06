import os
import json
import pickle
from typing import Dict
import pandas as pd
from io import StringIO
from src.core.utilities import globals
from src.core.utilities.settings import settings

from azure.storage.blob import (BlobServiceClient, ContainerClient)


class LoadBlob(object):

    def __init__(self, blob_name: str):
        if not blob_name:
            raise ValueError("Data name is required.")
        self.blob_name = blob_name
        self.pkl_file: str = self.blob_name.split(".")[0] + ".pkl"
        self._local_filename: str = globals.DATA_DIR / blob_name

        # Create a BlobServiceClient
        self.blob_service_client: BlobServiceClient = BlobServiceClient(
            account_url=settings.AZURE_ACCOUNT_URL,
            credential=settings.AZURE_BLOB_ACCESS_KEY)

        # Get a reference to the container
        self.container_client: ContainerClient = self.blob_service_client.get_container_client(
            settings.AZURE_CONTAINER_NAME)

    @property
    def data_name(self):
        return self.blob_name

    def __repr__(self) -> str:
        """
        Provides a string representation of the LoadBlob instance, including its class type and blob name.
        """
        return f"{type(self.__class__.__name__)} {self.blob_name}"

    def _load_csv_from_azure_storage(self) -> pd.DataFrame:

        print(f"Downloading {self.blob_name}")

        # Get a reference to the blob
        blob_client = self.container_client.get_blob_client(self.blob_name)

        # Download the blob content as a string
        blob_data = blob_client.download_blob()

        with open(self._local_filename, "wb") as my_blob:
            blob_data.download_to_stream(my_blob)

        # Create a Pandas DataFrame from the CSV content
        df: pd.DataFrame = pd.read_csv(
            self._local_filename, encoding='iso-8859-1',  low_memory=False)

        self._cache_df(df)

        print(f"Download complete")

        return df

    def _cache_df(self, df: pd.DataFrame) -> None:
        os.makedirs(globals.DATA_DIR, exist_ok=True)
        filepath: str = str(globals.DATA_DIR / self.pkl_file)

        with open(filepath, 'wb') as pkl:
            pickle.dump(df, pkl, protocol=pickle.HIGHEST_PROTOCOL)

    def load_data(self) -> pd.DataFrame:

        try:

            if not os.path.exists(self._local_filename):

                data = self._load_csv_from_azure_storage()
                return data

            else:
                pkl_file = globals.DATA_DIR / self.pkl_file

                if not os.path.exists(pkl_file):
                    data = self._load_csv_from_azure_storage()
                    
                with open(pkl_file, 'rb') as pkl:
                    data = pickle.load(pkl)
                    return data

        except EOFError:
            print(
                "Error: End of file reached unexpectedly. Check for file corruption or empty file.")
        except FileNotFoundError:
            print("Error: File not found. Verify the file path.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
