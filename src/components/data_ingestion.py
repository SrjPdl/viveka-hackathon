from dataclasses import dataclass
import os
import sys
from src.utils import download_file_sftp, unzip_file
from src.logger import logging
from src.exception import CustomException

@dataclass
class DataIngestionConfig:
    """
    Configuration class for data ingestion.

    Attributes:
        TRAIN_LOCAL_DATA_DIR (str): The local directory path for training data.
        TEST_LOCAL_DATA_DIR (str): The local directory path for test data.
        TRAIN_REMOTE_DATA_DIR (str): The remote directory path for training data on the server.
        TEST_REMOTE_DATA_DIR (str): The remote directory path for test data on the server.
        HOSTNAME (str): The hostname of the FTP/SFTP server.
        USER_NAME (str): The username for accessing the FTP/SFTP server.
        PASSWORD (str): The password for accessing the FTP/SFTP server.
    Note:
        The `TRAIN_LOCAL_DATA_DIR` and `TEST_LOCAL_DATA_DIR` attributes are generated using the current working directory
        and appending the respective directory names. The remote data directories (`TRAIN_REMOTE_DATA_DIR` and
        `TEST_REMOTE_DATA_DIR`) should be relative to the FTP/SFTP server's root directory.
    """

    TRAIN_LOCAL_DATA_DIR: str = os.path.join(os.getcwd(), "artifacts", "train_data")
    TEST_LOCAL_DATA_DIR: str = os.path.join(os.getcwd(), "artifacts", "test_data")
    TRAIN_REMOTE_DATA_DIR: str = "findit/FindIt-Dataset-Train.zip"
    TEST_REMOTE_DATA_DIR: str = "findit/FindIt-Dataset-Test.zip"
    HOSTNAME: str = "L3i-Share.univ-lr.fr"
    USER_NAME: str = "findit-participant"
    PASSWORD: str = "69cQek4N"


class DataIngestion:
    """
    Class for initiating data ingestion.

    Attributes:
        config (DataIngestionConfig): An instance of the DataIngestionConfig class that holds the configuration settings.
    Methods:
        initiate_data_ingestion(): Initiates the data ingestion process by creating local directories, downloading and unzipping the data from the remote server.

    Note:
        The initiate_data_ingestion method performs the following steps:
        1. Creates the local data directories specified in the DataIngestionConfig.
        2. Downloads the train and test data from the remote server using SFTP.
        3. Unzips the downloaded data files.
        
        This class assumes that the download_file_sftp and unzip_file functions are implemented separately.
    """

    def __init__(self) -> None:
        self.config = DataIngestionConfig()
    
    def initiate_data_ingestion(self) -> None:
        """
        Initiates the data ingestion process.

        This method creates the local data directories, downloads the train and test data from the remote server using SFTP,
        and unzips the downloaded data files.

        Raises:
            CustomException: If an error occurs during the data ingestion process.

        Example:
            >>> data_ingestion = DataIngestion()
            >>> data_ingestion.initiate_data_ingestion()
        """

        logging.info("Initiating data ingestion...")
        try:
            logging.info("Creating local data directories...")
            os.makedirs(self.config.TRAIN_LOCAL_DATA_DIR, exist_ok=True)
            os.makedirs(self.config.TEST_LOCAL_DATA_DIR, exist_ok=True)

            logging.info("Downloading train data...")
            download_file_sftp(self.config.HOSTNAME, self.config.USER_NAME, self.config.PASSWORD, self.config.TRAIN_REMOTE_DATA_DIR, os.path.join(self.config.TRAIN_LOCAL_DATA_DIR, "FindIt-Dataset-Train.zip"))

            logging.info("Downloading test data...")
            download_file_sftp(self.config.HOSTNAME, self.config.USER_NAME, self.config.PASSWORD, self.config.TEST_REMOTE_DATA_DIR, os.path.join(self.config.TEST_LOCAL_DATA_DIR, "FindIt-Dataset-Test.zip"))

            logging.info("Unzipping train data...")
            unzip_file(os.path.join(self.config.TRAIN_LOCAL_DATA_DIR, "FindIt-Dataset-Train.zip"), self.config.TRAIN_LOCAL_DATA_DIR)

            logging.info("Unzipping test data...")
            unzip_file(os.path.join(self.config.TEST_LOCAL_DATA_DIR, "FindIt-Dataset-Test.zip"), self.config.TEST_LOCAL_DATA_DIR)

            logging.info("Data ingestion completed successfully.")
        except Exception as e:
            error_message = str(e)
            raise CustomException(error_message, sys)
            
