from torch.utils.data import DataLoader, TensorDataset
from unet_model import Unet
from dataset import UnetData
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter
from trainer import Trainer

def main():
    writer = SummaryWriter("runs/exp1")

    model = Unet()

    input_size = 572
    output_size = 388
    batch_size = 2

    dataGenerator = UnetData(input_size, output_size, 0.25, int(input_size/4), int(input_size/3), 
                            min_center = 100, max_center=200)
    data, labels = dataGenerator.generateBatch(batch_size)
    dataset = TensorDataset(data, labels)
    train_dataloader = DataLoader(dataset, batch_size=batch_size)

    data, labels = dataGenerator.generateBatch(batch_size*1, seed = 75)
    test_dataloader = DataLoader(TensorDataset(data, labels), batch_size=batch_size)

    optimizer = SGD(model.parameters(), lr=0.1, momentum = 0.5)

    trainer = Trainer(model, writer)

    loss = trainer.train(30, optimizer, train_dataloader)

    print(loss.item())

if __name__ == "__main__":
    main()


