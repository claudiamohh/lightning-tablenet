import pytorch_lightning as pl
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision.models import vgg19, vgg19_bn
from .metrics import DiceLoss, binary_mean_iou

EPSILON = 1e-15

class TableNet(nn.Module):
    """
    Create a TableNet model with pre-trained layers of VGG-19.
    Using an encoder-decoder network architecture, conv1 to pool5 layers are used as common encoder layers,
    two decoder branches (table and column) are emerged after the encoder layers.
    """

    def __init__(self, num_class: int = 1, encoder="vgg"):
        """
        Args:
            num_class(int): Number of classes per point (output channel)
            encoder (str): Select choice of encoder (eg. vgg, vgg_bn)

        Usage example:
            TableNet(num_class=1)
                >> vgg19 encoder is used
            TableNet(num_class=1, encoder='vgg_bn')
                >> vgg19_bn encoder is used
        """
        super().__init__()
        self.encoder = (
            vgg19_bn(pretrained=True).features
            if encoder == "vgg_bn"
            else vgg19(pretrained=True).features
        )
        self.pooling_layers = [26, 39] if encoder == "vgg_bn" else [18, 27]
        self.model = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.8),
            nn.Conv2d(512, 512, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.8),
        )
        self.table_decoder = TableDecoder(num_class)
        self.column_decoder = ColumnDecoder(num_class)

    def forward(self, x):
        """Forward pass that maps an input tensor to a prediction output tensor.

        Args:
            x (tensor): Batch of images to perform forward-pass.

        Return (Tuple[tensor, tensor]): Table, Column prediction.

        Usage example:
            model = TableNet()
            output = model(torch.rand(2, 3, 864, 864)
                >>([tensor], [tensor])
        """
        results = []
        for idx, layer in enumerate(self.encoder):
            x = layer(x)
            if idx in self.pooling_layers:
                results.append(x)

        x_table = self.table_decoder(x, results)
        x_column = self.column_decoder(x, results)
        return torch.sigmoid(x_table), torch.sigmoid(x_column)


class ColumnDecoder(nn.Module):
    """
    Column Decoder responsible for segmenting out the columns from the image and construct a Column Mask.

    Function creates two convolution layers for inputs to pass through, upscaled by the given scale_factor,
    pool3 and pool 4 to meet the original image dimension and returns the transposed output.
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.8),
            nn.Conv2d(512, 512, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.transpose_layer = nn.ConvTranspose2d(
            1280, num_classes, kernel_size=2, stride=2, dilation=1
        )

    def forward(self, x, pools):
        pool_3, pool_4 = pools
        x = self.decoder(x)
        x = F.interpolate(x, scale_factor=2)
        x = torch.cat([x, pool_4], dim=1)
        x = F.interpolate(x, scale_factor=2)
        x = torch.cat([x, pool_3], dim=1)
        x = F.interpolate(x, scale_factor=2)
        x = F.interpolate(x, scale_factor=2)
        return self.transpose_layer(x)


class TableDecoder(ColumnDecoder):
    """
    Table Decoder responsible for segmenting out the tables from the image and construct a Table Mask,
    inheriting from the ColumnDecoder class.

    Function creates one convolution layer, pass through forward function and returns transposed output.
    """

    def __init__(self, num_classes):
        """
        Args:
            num_classes (int): Number of classes per point.
        """

        super().__init__(num_classes)
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1),
            nn.ReLU(inplace=True),
        )


class Lightning_TableNet(pl.LightningModule):
    """Create TableNet Pytorch Lightning model using the TableNet model."""

    def __init__(self, num_class: int = 1, encoder="vgg"):
        """
        Args:
            num_class (int) : Number of classes per point
            encoder (str): Select choice of encoder (vgg, vgg_bn)
        """
        super().__init__()
        self.model = TableNet(num_class, encoder="vgg")
        self.num_class = num_class
        self.encoder = encoder
        self.dice_loss = DiceLoss()
        self.example_input_array = torch.rand(2, 3, 1024, 1024)

    def forward(self, batch):
        """
        Forward pass that maps an input tensor to a prediction output tensor.

        Args:
            batch (tensor): Batch of images to perform forward-pass.

        Returns (Tuple[tensor, tensor]): Table, Column prediction.

        Usage example:
            model = Lightning_TableNet()
            output = model(torch.rand(2, 3, 864, 864)
                >>([tensor], [tensor])
        """
        return self.model(batch)

    def evaluate(self, batch, batch_idx, stage=None):
        samples, labels_table, labels_column = batch
        output_table, output_column = self.forward(samples)

        loss_table = self.dice_loss(output_table, labels_table)
        loss_column = self.dice_loss(output_column, labels_column)

        if stage:
            if batch_idx == 0:
                self._log_images(
                    f"{stage}",
                    samples,
                    labels_table,
                    labels_column,
                    output_table,
                    output_column,
                )
            self.log(f"{stage}_loss_table", loss_table, on_epoch=True)
            self.log(f"{stage}_loss_column", loss_column, on_epoch=True)
            self.log(f"{stage}_loss", loss_column + loss_table, on_epoch=True)
            self.log(
                f"{stage}_binary_mean_iou_table",
                binary_mean_iou(output_table, labels_table),
                on_epoch=True,
            )
            self.log(
                f"{stage}_binary_mean_iou_column",
                binary_mean_iou(output_column, labels_column),
                on_epoch=True,
            )
            self.log(
                f"{stage}_accuracy_column",
                accuracy(output_column, labels_column.int()),
                on_epoch=True,
            )
            self.log(
                f"{stage}_accuracy_table",
                accuracy(output_table, labels_table.int()),
                on_epoch=True,
            )

        return loss_table + loss_column

    def training_step(self, batch, batch_idx):
        """
        Training TableNet model with Dataset

        Args:
            batch (List[Tensor]): Data for training.
            batch_idx (int): batch index

        Returns: Tensor
        """
        samples, labels_table, labels_column = batch
        output_table, output_column = self.forward(
            samples
        )  # Pass through forward function to train with model

        # Calculate loss of table and column
        loss_table = self.dice_loss(output_table, labels_table)  # Table Loss
        loss_column = self.dice_loss(output_column, labels_column)  # Column Loss

        self.log("train_loss_table", loss_table)
        self.log("train_loss_column", loss_column)
        self.log("train_loss", loss_column + loss_table)
        return loss_table + loss_column

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, batch_idx, "test")

    def configure_optimizers(self):
        """
        Using Adam Optimizer with learning rate=0.0001

        Returns: optimizer for pytorch lightning
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)

        return optimizer

    def _log_images(
        self, mode, samples, labels_table, labels_column, output_table, output_column
    ):
        """Log images to logger"""

        self.logger.experiment.add_images(
            f"{mode}_generated_images", samples[0:4], self.current_epoch
        )
        self.logger.experiment.add_images(
            f"{mode}_labels_table", labels_table[0:4], self.current_epoch
        )
        self.logger.experiment.add_images(
            f"{mode}_labels_column", labels_column[0:4], self.current_epoch
        )
        self.logger.experiment.add_images(
            f"{mode}_output_table", output_table[0:4], self.current_epoch
        )
        self.logger.experiment.add_images(
            f"{mode}_output_column", output_column[0:4], self.current_epoch
        )
