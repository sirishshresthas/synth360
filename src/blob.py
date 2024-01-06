import os
import pandas as pd
from io import StringIO
from dotenv import load_dotenv

from azure.storage.blob import (BlobServiceClient, ContainerClient)
from col_dtype import load_col_types

load_dotenv()

CSV_FOLDER_PATH: str = ''

# Create a BlobServiceClient
blob_service_client: BlobServiceClient = BlobServiceClient(
    account_url=os.environ["AZURE_ACCOUNT_URL"], 
    credential=os.environ["AZURE_BLOB_ACCESS_KEY"])

# Get a reference to the container
container_client: ContainerClient = blob_service_client.get_container_client(
    os.environ["AZURE_CONTAINER_NAME"])


def load_csv_from_azure_storage(blob_name):
    

    # Get a reference to the blob
    blob_client = container_client.get_blob_client(blob_name)

    # Download the blob content as a string
    blob_data = blob_client.download_blob()
    csv_content = blob_data.readall().decode('utf-8')

    # Create a Pandas DataFrame from the CSV content
    df = pd.read_csv(StringIO(csv_content), 
                    dtype=load_col_types)

    return df

