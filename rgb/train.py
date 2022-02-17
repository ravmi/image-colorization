import torch
import os
from torch.utils.data import DataLoader
from basic_model import Net
import torchvision.transforms as T
from skimage import io, color
import numpy as np
from PIL import Image
import neptune.new as neptune


class Trainer:
    def __init__(self, data,
                 lr=1e-6,
                 epochs=300,
                 validation_split=0.1,
                 batch_size=32,
                 shuffle=True,
                 device='cuda',
                 alpha=0.,
                 loss_type='bright1'):
        self.lr = lr
        self.epochs = epochs
        self.data = data
        self.validation_split = validation_split
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device
        self.alpha = alpha
        self.loss_type = loss_type

    def train(self):
        run = neptune.init(
            project="rm360179/image-coloring",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2MmVjN2EzYS04Y2FmLTRkYjItOTkyMi1mNmEwYWQzM2I3Y2UifQ==",
        ) 
        run['lr'] = self.lr
        run['epochs'] = self.epochs 
        run['alpha'] = self.alpha
        run['loss_type'] = self.loss_type
        val_size = int(self.validation_split * len(self.data))

        train_dataset = self.data[val_size:]
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)

        val_dataset = self.data[:val_size]
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=self.shuffle)

        # Model
        model = Net()
        run['model'] = model
        model.to(self.device)
        # Loss function to use
        criterion = torch.nn.MSELoss()
        def criterion_bright(y_pred, y):
            return (-y_pred**2).mean()

        def criterion_bright2(y_pred, y):
            return ((abs(y_pred) - abs(y))**2).mean()

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        for epoch in range(1, self.epochs + 1):
            # TRAIN #
            model.train()
            train_losses = list()
            for batch in train_dataloader:
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)
                y_pred = model(x)

                loss = criterion(y_pred, y)
                if self.loss_type=='bright1':
                    loss += criterion_bright(y_pred, y) * self.alpha
                elif self.loss_type=='bright2':
                    loss += criterion_bright2(y_pred, y) * self.alpha

                run['loss'].log(loss.item())
                train_losses.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            run['loss_train'].log(sum(train_losses)/len(train_losses))

            # EVAL #
            model.eval()
            val_losses = list()
            with torch.no_grad():
                for batch in val_dataloader:
                    x, y = batch
                    x = x.to(self.device)
                    y = y.to(self.device)
                    y_pred = model(x)
                    loss = criterion(y_pred, y)
                    val_losses.append(loss.item())

            run['loss_eval'].log(sum(val_losses)/len(val_losses))
            if epoch % 10 == 0:
                if not os.path.exists('models'):
                    os.makedirs('models')
                torch.save(model, f'models/model_{epoch}.pt')
                with torch.no_grad():
                    for batch in val_dataloader:
                        x, y = batch
                        x = x.to(self.device)
                        y = y.to(self.device)
                        y_pred = model(x)
                        # We will generate some images, but only on the first batch
                        break

                    for xs, ys, yps in zip(x, y, y_pred[:10]):
                        def to_image(data):
                            return T.ToPILImage()(data * 0.5 + 0.5)

                        run['gray_image'].log(to_image(xs))
                        run['original_image'].log(to_image(ys))
                        run['generated_image'].log(to_image(yps))

        run.stop()


    def validate(self):
        pass
        # Validation loop begin
        # ------
        # Validation loop end
        # ------
        # Determine your evaluation metrics on the validation dataset.
