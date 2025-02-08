import unittest
from unet_model import Unet
from unet_model import Unet
from dataset import UnetData
from torch.optim import SGD
from trainer import Trainer
from torch.utils.data import DataLoader, TensorDataset

class TestUnetModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = Unet()
        print("unet network created")

    def test_convergence(self):

        input_size = 572
        output_size = 388
        batch_size = 1

        dataGenerator = UnetData(input_size, output_size, 0.25, int(input_size/4), int(input_size/3), 
                                min_center = 100, max_center=200)
        data, labels = dataGenerator.generateBatch(batch_size)
        train_dataloader = DataLoader(TensorDataset(data, labels), batch_size=batch_size)

        optimizer = SGD(self.model.parameters(), lr=0.1, momentum = 0.5)

        trainer = Trainer(self.model)

        loss = trainer.train(20, optimizer, train_dataloader)
        self.assertLess(loss.item(), 0.1)
