import os
from src.components.data_ingestion import DataIngestion
from src.components.data_loader import DataLoadTransform
from src.logger import logging
from src.components.models_architecture import InceptBaseModel
from src.components.model_trainer import ModelTrainer
import torch
import torch.nn as nn
from src.components.data_loader import DataConfig

class ModelConfig:
    def __init__(self, model) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = DataConfig().BATCH_SIZE
        self.epochs = 5
        self.model = model
        self.save_model_path = os.path.join(os.getcwd(), "artifacts", "model")
        self.loss_fn = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)



if __name__ == "__main__":
    # data_ingestion = DataIngestion()
    # data_ingestion.initiate_data_ingestion()
    data_loader = DataLoadTransform()
    train_loader = data_loader.get_train_loader()
    test_loader = data_loader.get_test_loader()
    logging.info("Get data loader completed successfully.")
    model = InceptBaseModel()
    model_config = ModelConfig(model)
    logging.info("Model configuration completed successfully.")
    model_trainer = ModelTrainer(model, train_loader, test_loader, model_config.loss_fn, model_config.optimizer, model_config.device, model_config.batch_size)
    logging.info("Starting model training...")
    model_trainer.train_model(model_config.epochs, model_config.save_model_path)

    