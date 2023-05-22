from zipfile import ZipFile
from src.logger import logging
from src.exception import CustomException
import os
import paramiko
import sys

def download_file_sftp(hostname, username, password, remote_file_path, local_file_path):
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

def unzip_file(file_path, destination_path):
    with ZipFile(file_path, 'r') as zip_file:
        zip_file.extractall(destination_path)
        os.remove(file_path)
    logging.info(f"File '{file_path}' unzipped successfully.")