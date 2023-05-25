from zipfile import ZipFile
from src.logger import logging
from src.exception import CustomException
import os
import paramiko
import sys

def download_file_sftp(hostname:str, username:str, password:str, remote_file_path:str, local_file_path:str) -> None:
    """
    Downloads a file from an SFTP server to the local file system.

    Args:
        hostname : The hostname or IP address of the SFTP server.
        username : The username for authentication.
        password : The password for authentication.
        remote_file_path : The path of the file on the SFTP server.
        local_file_path : The path where the file will be downloaded locally.
    Returns:
        None
    Raises:
        CustomException: If any error occurs during the download process.

    """
    try:
        # Create an SSH client
        client = paramiko.SSHClient()

        # Automatically add the server's host key
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        # Connect to the SFTP server
        client.connect(hostname, username=username, password=password)

        # Create an SFTP client from the SSH client
        sftp = client.open_sftp()

        # Download the file
        sftp.get(remote_file_path, local_file_path)

        logging.info(f"File '{remote_file_path}' downloaded successfully.")

    except Exception as e:
        error_message = str(e)
        raise CustomException(error_message, sys)
    finally:
        # Close the SFTP session and the SSH connection
        sftp.close()
        client.close()

def unzip_file(file_path: str, destination_path:str ) -> None:
    """
    Unzips a ZIP file to the specified destination path.

    Args:
        file_path (str): The path of the ZIP file to be extracted.
        destination_path (str): The path where the contents of the ZIP file will be extracted.
    Returns:
        None

    """
    with ZipFile(file_path, 'r') as zip_file:
        zip_file.extractall(destination_path)
        os.remove(file_path)
    logging.info(f"File '{file_path}' unzipped successfully.")

def get_latest_best_model(model_path: str) -> str:
    """
    Retrieves the path of the latest best model file in the specified directory.

    Args:
        model_path: The path to the directory containing the model files.

    Returns:
        The path of the latest best model file.
    """
    model_files = [file for file in os.listdir(model_path) if not file.endswith('.txt')]
    model_files.sort(reverse=True)
    if model_files:
        model_path = os.path.join(model_path, model_files[0])
        return model_path
    else:
        print("No model files found in the directory.")
        return None