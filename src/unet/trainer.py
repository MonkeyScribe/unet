from torch.nn import Module
from torch.nn.functional import binary_cross_entropy
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
import logging
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import torch
import numpy as np
from typing import Optional

logger = logging.getLogger("Trainer")
logging.basicConfig(level=logging.INFO)

class Trainer():

    def __init__(self, model:Module, writer : Optional[SummaryWriter]=None):
        self.model = model
        self.writer = writer
    
    def train(self, n_epochs : int, optimizer : Optimizer, train_dataloader : DataLoader):
        self.model.train()
        prev_loss = 1
        for n in range(n_epochs):
            for batch, (data, label) in enumerate(train_dataloader):
                output = self.model(data)
                y_pred = output[:,1,:,:]
                loss = binary_cross_entropy(y_pred.flatten(), label.flatten())
                loss.backward()
                optimizer.step()
                if self.writer is not None : 
                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            self.writer.add_histogram(f"{name}_grad", param.grad, n*len(train_dataloader.dataset)+batch)
                optimizer.zero_grad()
            if loss.item() < 0.1 * prev_loss:
                prev_loss = loss.item()
                for g in optimizer.param_groups:
                    g['lr'] = g['lr']/10
            
            if self.writer is not None : 
                self.writer.add_scalar("train loss", loss.item(), n_epochs)
            
            logger.info("nepochs is : "+str(n))
            logger.info(loss.item())
        return loss

    @staticmethod
    def dice(pred: Tensor, label : Tensor)-> np.float32:
        ypred = (pred > 0.5).to(torch.int32)
        common = torch.sum((ypred == 1)*(label==1))
        a = torch.sum(ypred)
        b = torch.sum(label)
        dice = 2*common/(a+b)
        return dice.item()

    def test(self, test_dataloader : DataLoader):
        self.model.eval()
        for batch, (data, label) in enumerate(test_dataloader):
            output = self.model(data)
            y_pred = output[:,1,:,:]
            loss = binary_cross_entropy(y_pred.flatten(), label.flatten())
            self.writer.add_scalar("test loss", loss.item())
            tensor_list = torch.unbind(data, dim=0)
            mean = np.mean([Trainer.dice(el) for el in tensor_list])



