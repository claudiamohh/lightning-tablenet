# Pytorch Lightning Tablenet

This repository consists of a Pytorch Lightning implementation of Tablenet https://arxiv.org/pdf/2001.01469.pdf. The end product of this repository is a gradio application where it returns the contents of the tables using OCR tesseract. 

# Introduction 

The task here is to extract tables from the dataset using the trained model, and return the predicted content of each table and column in a dataframe. 

The model takes in image pages, predicts and creates table and column masks called output_table and output_column respectively. Binary mean IOU evaluates the model by comparing output_table and output_column with ground truth masks. With the best trained model chosen, it then predicts the content of these created masks using OCR tesseract. Lastly, an appplication is created using Gradio where users are able to input an image and retrieve a table of the content in a dataframe format. 

# Marmot Dataset 

This dataset contains annotated English pages for table recognition. Image data is in .bmp (bitmap image file) format and there are 509 different images in the dataset whereas there are 510 column and table masks files. While studying the dataset in `Marmot_EDA.ipynb`, there is an error such that the table_mask displays more than one table when the image has only one table. 

The dataset includes images, column masks and table masks, which can be found in the `data` directory in `data.zip`.  

This is an example of how the dataset looks like after unzipping (refer to `Execution` on how to unzip file): 

```
└── lightning-tablenet/
    └── data/
        ├── Marmot_data/
        │   ├── 10.1.1.1.2006_3.bmp
        │   ├── 10.1.1.1.2006_3.xml
        │   ├── 10.1.1.1.2044_7.bmp
        │   └── 10.1.1.1.2044_7.xml
        ├── table_mask/
        │   ├── 10.1.1.1.2006_3.bmp
        │   └── 10.1.1.1.2044_7.bmp
        └── column_mask/
            ├── 10.1.1.1.2006_3.bmp
            └── 10.1.1.1.2044_7.bmp
```
For more details of the dataset, it can be explored in `Marmot_EDA.ipynb`. 

# Model Weights 

The model is trained with a pretrained vgg19 model as the enocder and creates table and column decoders respectively. The model weights is also available [here](https://drive.google.com/file/d/1aJfBOwOk6F2wRS0wRevZFGB9cZkDv_Sy/view?usp=sharing) where it was trained for 56 epochs with gradient clipping. 

| Model | No. of epochs | Validation Loss | Binary mean IOU for table | Binary mean IOU for column |
|-------|---------------|-----------------|---------------------------|----------------------------|
|tablenet_baseline_adam_gradclipping| 56 | 0.212 | 0.753 | 0.689 | 

For a better understanding of the pretrained VGG19 model, refer to `vgg19_understanding.ipynb` in `notebooks' directory. In this notebook, the last classifier layer of VGG19 is replaced with 10 output classes to suit the training of MNIST dataset. 

Users are to create a new directory `pretrained_models` and save the model weights inside (refer to `Getting Started` Section). If you are training your own model, do rename it to `tablenet_baseline_adam_gradclipping.ckpt` to be able to run gradio_demo.py. 

# Model Results 

Each model with different paramenters is trained 3 times. Following are the results of the best checkpoint: 

| Model | No. of epochs | Validation Loss | Binary mean IOU for table | Binary mean IOU for column | 
|-------|---------------|-----------------|---------------------------|----------------------------|
|1. tablenet_baseline_adam_1| 1 | 0.384 | 0.740 | 0.652 |
|2. tablenet_baseline_adam-2| 4 | 0.361 | 0.750 | 0.650 | 
|3. tablenet_baseline_adam_3| 28 | 0.298 | **0.807** | 0.697|
|4. tablenet_baseline_adam_gradclipping_1| 56 | **0.212** | 0.753 | 0.689|
|5. tablenet_baseline_adam_gradclipping_2| 25 | 0.225 | 0.682 | 0.709|
|6. tablenet_baseline_adam_gradclipping_3| 24 | 0.220 | 0.756 | **0.717**|
|7. tablenet_baseline_adam_gradclipping_weightdecay_1| 9 | 0.321 | 0.136 | 0.112|
|8. tablenet_baseline_adam_gradclipping_weightdecay_2| 11 | 0.299 | 0.136 | 0.111|
|9. tablenet_baseline_adam_gradclipping_weightdecay_3| 6 | 0.342 | 0.136 | 0.113|
|10. tablenet_baseline_adam_lr_5e-5_1|19|0.286| 0.812| 0.707
|11. tablenet_baseline_adam_lr_5e-5_2|20|0.266| 0.572| 0.673
|12. tablenet_baseline_adam_lr_5e-5_3|26|0.281|0.796| 0.715
|13. tablenet_baseline_adam_transposed_1|0|0.456| 0.649| 0.651|
|14. tablenet_baseline_adam_vggbn_lowlr_1|19|0.285|0.136|0.111
|15. tablenet_baseline_adam_vggbn_lowlr_2|39|0.285|0.136|0.112
|16. tablenet_baseline_adam_vggbn_lowlr_3|30|0.288|0.136|0.112
|17. tablenet_baseline_sgd_onecycelr_1|44|0.488|0.226| 0.279|
|18. tablenet_baseline_sgd_onecycelr_2|45|0.583|0.592|0.497|
|19. tablenet_baseline_sgd_onecycelr_3|52|0.511|0.585|0.302
|20. tablenet_baseline_sgd_onecycelr_lowlr_1|55|0.459|0.252| 0.242
|21. tablenet_baseline_sgd_onecycelr_lowlr_2|91|0.804|0.161|0.233 |
|22. tablenet_baseline_sgd_onecycelr_lowlr_3|37|0.648|0.177| 0.137| 

Hence, `tablenet_baseline_adam_gradclipping_1` is used as the model weights as it has the lowest validation loss and decent values of binary mean IOU. 

# Getting Started 

To clone this repository: 
```
$ git clone https://github.com/claudiamohh/lightning-tablenet.git
$ cd lightning-tablenet
```

To install and activate virtual environment:
```
$ python -m venv env

# Linux
source env/bin/activate

# Windows
source env/Scripts/activate
```

To install requirements:
```
$ pip install -r requirements.txt
```

Steps to move model weights inside `pretrained_models` directory:
1. Click on the link in `Model Weights` Section to download the file in Google Drive
2. Create `pretrained_models` directory and move weights inside 
```
$ mkdir pretrained_models
$ mv tablenet_baseline_adam_gradclipping.ckpt pretrained_models/
```

# Execution 

To unzip the dataset, simply type the following command: 
```
$ cd data/ && unzip -q data.zip
```

To train the model, type the following command: 
```
$ python train.py
```

To create a gradio application as shown below, ensure that `examples` and `model weights` have already been downloaded and type the following command: 
```
$ python gradio_demo.py
```

After running this command, users are able to visit `http://localhost:7861` to try out the demo. It is a user friendly demo where users are able to click to upload or drag images in the input component and returns a dataframe. 

![image](https://user-images.githubusercontent.com/107597583/186386498-567dd549-441c-4da5-8d85-5948c37f91b2.png)

![Screenshot 2022-08-24 173202](https://user-images.githubusercontent.com/107597583/186386131-392d7866-91fe-4bb9-9655-ac260f1f4d29.png)

# References 
1. [OCR_tablenet](https://github.com/tomassosorio/OCR_tablenet)
2. [TableNet-pytorch](https://github.com/tomassosorio/OCR_tablenet)

