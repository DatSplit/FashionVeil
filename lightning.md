### Pytorch lightning

Instead of nn.Module -> pl.LightningModule

training_step: get batch, compute gradients, step, compute loss, return loss
common_step: shared by training and validation steps