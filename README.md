# Pytorch Lightning Tablenet

This repository consists of a Pytorch Lightning implementation of Tablenet https://arxiv.org/pdf/2001.01469.pdf. The end product of this repository is a gradio application where it returns the contents of the tables using OCR tesseract. 

This repository includes 3 notebooks (for a better understanding of the dataset and pretrained model used) and 6 scripts. 

The notebooks are: 
1. Marmot_EDA.ipynb
2. albumentation.ipynb
3. vgg19_understanding.ipynb

# Introduction 

The task here is to extract tables from the dataset using the trained model, and return the predicted content of each table and column in a dataframe. 

The model takes in image pages, predicts and creates table and column masks called labels_table and label_column respectively. Binary mean IOU evaluates the model by comparing labels_table and labels_column with ground truth masks. With the best trained model chosen, it then predicts the content of these created masks using OCR tesseract. Lastly, an appplication is created using Gradio where users are able to input an image and retrieve a table of the content in a dataframe format. 

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

The model is trained with a pretrained vgg19 model as the enocder and creates table and column decoders respectively. The model weights is also available [here]('https://drive.google.com/file/d/1aJfBOwOk6F2wRS0wRevZFGB9cZkDv_Sy/view?usp=sharing') where it was trained for 56 epochs with gradient clipping. 

| Model | No. of epochs | Validation Loss | Binary mean IOU for table | Binary mean IOU for column |
|-------|---------------|-----------------|---------------------------|----------------------------|
|tablenet_baseline_adam_gradclipping| 56 | 0.212 | 0.753 | 0.689 | 

For a better understanding of the pretrained VGG19 model, refer to `vgg19_understanding.ipynb` in `notebooks' directory. In this notebook, the last classifier layer of VGG19 is replaced with 10 output classes to suit the training of MNIST dataset. 

Users are to create a new directory `pretrained_model` and save the model weights inside (refer to `Getting Started` Section). If you are training your own model, do rename it to `tablenet_baseline_adam_gradclipping.ckpt` to be able to run gradio_demo.py. 

# Getting Started 

To clone this repository: 
```
$ git clone https://github.com/claudiamohh/lightning-tablenet.git
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

Steps to move model weights inside `pretrained_model` directory:
1. Click on the link in `Model Weights` Section to download the file in Google Drive
2. Create `pretrained_model` directory and move weights inside 
```
$ mkdir pretrained_model
$ mv tablenet_baseline_adam_gradclipping.ckpt pretrained_model/
```

# Execution 

To unzip the dataset, simply type the following command: 
```
$ unzip data/data.zip
```

To train the model, type the following command: 
```
$ python train.py
```

To create a gradio application, ensure that examples and model path has already been downloaded and type the following command: 
```
$ python gradio_demo.py
```


