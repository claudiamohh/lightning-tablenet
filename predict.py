from collections import OrderedDict
from typing import List
import numpy as np
import pandas as pd
from albumentations import Compose
from PIL import Image
from pytesseract import image_to_string
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square, convex_hull_image
from skimage.transform import resize
from skimage.util import invert
from models.tablenet import LightningTableNet


class PredictImage:
    def __init__(
        self,
        checkpoint_path: str,
        transforms: Compose,
        threshold: float = 0.5,
        per: float = 0.005,
    ):
        """
        PredictImage class containing helper functions to predict and recognize
        content of the columns using OCR tesseract. It loads the TableNet model
        and uses its output, table and column masks, to predict the content.

        Args:
            checkpoint_path (str): model weights path
            transforms (Optional[Compose]): Compose object from albumentations
            used for training
            per (float): Minimum area for tables and columns to be
            considered
        """

        self.transforms = transforms
        self.threshold = threshold
        self.per = per

        self.model = LightningTableNet.load_from_checkpoint(checkpoint_path)
        self.model.eval()
        self.model.requires_grad_(False)

    def predict(self, image: Image) -> List[pd.DataFrame]:
        processed_image = self.transforms(image=np.array(image))["image"]

        table_mask, column_mask = self.model.forward(processed_image.unsqueeze(0))

        table_mask = self.add_threshold(table_mask)
        column_mask = self.add_threshold(column_mask)

        segmented_tables = self.process_tables(self.segment_image(table_mask))

        tables = []
        for table in segmented_tables:
            segmented_columns = self.process_columns(
                self.segment_image(column_mask * table)
            )
            if segmented_columns:
                cols = []
                for column in segmented_columns.values():
                    cols.append(self.column_to_dataframe(column, image))
                tables.append(pd.concat(cols, ignore_index=True, axis=1))
        return tables

    def add_threshold(self, mask):
        mask = mask.squeeze(0).squeeze(0).numpy() > self.threshold
        return mask.astype(int)

    def process_tables(self, segmented_tables):
        width, height = segmented_tables.shape
        tables = []
        for i in np.unique(segmented_tables)[1:]:
            table = np.where(segmented_tables == i, 1, 0)
            if table.sum() > height * width * self.per:
                tables.append(convex_hull_image(table))
        return tables

    def process_columns(self, segmented_columns):
        width, height = segmented_columns.shape
        cols = {}
        for j in np.unique(segmented_columns)[1:]:
            column = np.where(segmented_columns == j, 1, 0)
            column = column.astype(int)

            if column.sum() > width * height * self.per:
                position = regionprops(column)[0].centroid[1]
                cols[position] = column
        return OrderedDict(sorted(cols.items()))

    @staticmethod
    def segment_image(image):
        threshold = threshold_otsu(image)
        bw = closing(image > threshold, square(2))
        removed = clear_border(bw)
        label_image = label(removed)
        return label_image

    @staticmethod
    def column_to_dataframe(column, image):
        width, height = image.size
        column = (
            resize(np.expand_dims(column, axis=2), (height, width), preserve_range=True)
            > 0.01
        )

        crop = column * image
        white = np.ones(column.shape) * invert(column) * 255
        crop = crop + white
        ocr = image_to_string(Image.fromarray(crop.astype(np.uint8)))
        return pd.DataFrame(
            {"col": [value for value in ocr.split("\n") if len(value) > 0]}
        )
