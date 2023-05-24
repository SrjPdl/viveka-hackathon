from datetime import datetime
from src.logger import logging
import torch
import os
class ModelTrainer:
    def __init__(self, model, train_loader, test_loader,  loss_fn, optimizer, device, batch_size):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.batch_size = batch_size
    
    def __train_one_epoch(self, train_data_len):
        running_loss = 0.
        last_loss = 0.
        total_batch = train_data_len// self.batch_size
        correct = 0
        for i, data in enumerate(self.train_loader):
            inputs, labels = data['image'].to(self.device), data['label'].to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs.logits, labels)
            loss.backward()
            self.optimizer.step()
            outputs = (outputs.logits>0.5).float()
            correct += (outputs == labels).float().sum()
            running_loss += loss.item()
            if i % 1 == 0:
                last_loss = running_loss 
                logging.info(f'batch {i+1}/{total_batch + 1} loss: {last_loss}')
                running_loss = 0.
        accuracy = 100 * correct / train_data_len
        return last_loss, accuracy
    
    def train_model(self, epochs, save_model_path):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        best_test_loss = 1_000_000.

        for epoch in range(epochs):
            logging.info(f'EPOCH {epoch + 1}/{epochs}: ')
            self.model.train(True)
            avg_loss, accuracy = self.__train_one_epoch(len(self.train_loader.dataset))

            self.model.train(False)
            running_test_loss = 0.0
            correct_test = 0
            for i, t_data in enumerate(self.test_loader):
                t_inputs, t_labels = t_data['image'].to(self.device), t_data['label'].to(self.device)
                t_outputs = self.model(t_inputs)
                t_loss = self.loss_fn(t_outputs, t_labels)
                t_outputs = (t_outputs>0.5).float()
                correct_test += (t_outputs == t_labels).float().sum()
                running_test_loss += t_loss

            avg_test_loss = running_test_loss / (i + 1)
            test_accuracy = 100 * correct_test / len(self.test_loader.dataset)
            logging.info('LOSS train {} Test {}'.format(avg_loss, avg_test_loss))
            logging.info('Accuracy train {} Test {}'.format(accuracy, test_accuracy))

            if avg_test_loss < best_test_loss:
                best_test_loss = avg_test_loss
                os.makedirs(save_model_path, exist_ok=True)
                model_path = os.path.join(save_model_path, 'model_{}_{}'.format(timestamp, epoch + 1))
                report_file = os.path.join(save_model_path, 'report_{}_{}.txt'.format(timestamp, epoch + 1))
                with open(report_file, 'w') as f:
                    f.write('LOSS train {} Test {}\n'.format(avg_loss, avg_test_loss))
                    f.write('Accuracy train {} Test {}'.format(accuracy, test_accuracy))
                torch.save(self.model.state_dict(), model_path)