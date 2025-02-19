### Pytorch lightning

#### Model development

Instead of nn.Module -> pl.LightningModule

training_step: get batch, compute gradients, step, compute loss, return loss
common_step: shared by training and validation steps

#### Data module

def prepare_data(self):
    download data
def setup(stage):
    customize dataset (transform)
def train_dataloader()
def val_dataloader()
def test_dataloader()
def predict_dataloader()

![alt text](image.png)
![alt text](image-1.png)
