import unittest
from trainer import Trainer
import torch


class TestTrainer(unittest.TestCase):

    def setUp(self):
        print("chapeau")

    def test_foo(self):
        self.assertEqual(1,1)

    def test_dice(self):
        pred = torch.rand(size=(1,5,5))
        label = torch.randint(high=2, size = (1, 5,5))
        ypred =  (pred > 0.5).to(torch.int32)
        self.assertAlmostEqual(Trainer.dice(label, label),1, delta=0.001)
        self.assertAlmostEqual(Trainer.dice(pred, ypred),1, delta=0.001)
        self.assertAlmostEqual(Trainer.dice(torch.zeros(1,5,5), label), 0)