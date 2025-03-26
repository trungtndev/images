import pytorch_lightning as pl
import torch
from torchmetrics import Accuracy
from .model.basecnn import CNNClassifier

class LitModel(pl.LightningModule):
    def __init__(self,
                 num_classes: int,
                 d_model: int,

                 requires_grad: bool,
                 drop_rate: float,
                 proj_drop_rate: float,
                 attn_drop_rate: float,
                 drop_path_rate: float,

                 growth_rate: int,
                 num_layers: int,

                 # training
                 learning_rate: float,
                 weight_decay: float,
                 patience: int,
                 ):
        super().__init__()
        self.model = CNNClassifier(num_classes=num_classes)

        self.train_accuracy = Accuracy(task='multiclass', num_classes=2)
        self.val_accuracy = Accuracy(task='multiclass', num_classes=2)
        self.test_accuracy = Accuracy(task='multiclass', num_classes=2)

        self.save_hyperparameters()

    def forward(self, image):
        return self.model(image)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            weight_decay=self.hparams.weight_decay
        )
        step_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            verbose=True,
            step_size=self.hparams.patience,
            gamma=0.25
        )

        scheduler = {
            "scheduler": step_scheduler,
            "interval": "epoch",
            "frequency": 1,
            "strict": True,
        }

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        image, labels = batch
        outputs = self(image)
        self.train_accuracy(outputs.softmax(dim=-1), labels)

        loss = self.compute_loss(outputs, labels)

        self.log('train_loss', loss,
                 prog_bar=True,
                 on_step=True,
                 on_epoch=True,
                 sync_dist=True
                 )
        self.log('train_acc', self.train_accuracy,
                 prog_bar=True,
                 on_step=False,
                 on_epoch=True,
                 )

        return loss

    def validation_step(self, batch, batch_idx):
        image, labels = batch
        outputs = self(image)

        loss = self.compute_loss(outputs, labels)
        self.val_accuracy(outputs.softmax(dim=-1), labels)

        self.log('val_loss', loss,
                 prog_bar=True,
                 on_step=True,
                 on_epoch=True,
                 sync_dist=True
                 )
        self.log('val_acc', self.val_accuracy,
                 prog_bar=True,
                 on_step=False,
                 on_epoch=True,
                 )

    def test_step(self, batch, batch_idx):
        image, labels = batch
        outputs = self(image)

        loss = self.compute_loss(outputs, labels)
        self.test_accuracy(outputs, labels)

        self.log('test_loss', loss,
                 on_step=False,
                 on_epoch=True,
                 sync_dist=True
                 )
        self.log('test_acc', self.test_accuracy,
                 on_step=False,
                 on_epoch=True,
                 )

    def compute_loss(self, outputs, labels):
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        return loss


if __name__ == '__main__':
    # model = LitModel(2,
    #                  512,
    #                  True,
    #                  0.1, 0.1, 0.1, 0.1,
    #                  32, 4,
    #                  1e-3, 1e-4, 3)
    # spa = torch.randn(1, 3, 224, 224)
    # fre = torch.randn(1, 3, 224, 224)
    # out = model(fre)
    pass
