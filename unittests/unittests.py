import unittest
from unittest.mock import patch, mock_open, MagicMock
from training.train import DataProcessor, SimpleNN, train_and_evaluate_model, evaluate_model, main
import pandas as pd
import torch

class TestTrainingScript(unittest.TestCase):
    @patch('training.train.get_project_dir')
    @patch('training.train.configure_logging')
    def test_data_processor_prepare_data(self, mock_configure_logging, mock_get_project_dir):
        mock_get_project_dir.return_value = 'dummy_project_dir'

        data_proc = DataProcessor()
        path = 'dummy_path'

        with patch.object(data_proc, 'data_extraction', return_value=pd.DataFrame()), \
                patch.object(data_proc, 'drop_duplicates', return_value=pd.DataFrame()), \
                patch.object(data_proc, 'data_split', return_value=(pd.DataFrame(), pd.DataFrame())), \
                patch.object(data_proc, 'data_scaling', return_value=(pd.DataFrame(), pd.DataFrame())):
            train_loader, test_loader = data_proc.prepare_data(path)

            self.assertIsNotNone(train_loader)
            self.assertIsNotNone(test_loader)

    def test_simple_nn_model(self):
        model = SimpleNN()
        input_tensor = torch.randn((1, 4))
        output_tensor = model(input_tensor)

        self.assertEqual(output_tensor.shape, torch.Size([1, 3]))

    @patch('training.train.configure_logging')
    @patch('training.train.DataProcessor.prepare_data', return_value=(MagicMock(), MagicMock()))
    @patch('training.train.train_and_evaluate_model')
    @patch('training.train.evaluate_model')
    @patch('training.train.save_model')
    def test_main_function(self, mock_save_model, mock_evaluate_model, mock_train_and_evaluate_model, mock_prepare_data, mock_configure_logging):
        with patch('builtins.open', mock_open(
                read_data='{"general": {"data_dir": "dummy_data_dir", "models_dir": "dummy_models_dir", "target_variable": "dummy_target_var", "random_state": 42, "datetime_format": "%Y-%m-%d %H:%M:%S"}, "train": {"table_name": "dummy_table_name", "batch_size": 32, "test_size": 0.2, "epochs": 5, "learning_rate": 0.001, "verbose": 1, "plot": true}}}')):
            main()

        mock_configure_logging.assert_called_once()
        mock_prepare_data.assert_called_once()
        mock_train_and_evaluate_model.assert_called_once()
        mock_evaluate_model.assert_called_once()
        mock_save_model.assert_called_once()


if __name__ == '__main__':
    unittest.main()
