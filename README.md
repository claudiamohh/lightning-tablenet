# Pytorch Lightning Tablenet

This repository consists of a Pytorch Lightning implementation of Tablenet https://arxiv.org/pdf/2001.01469.pdf. The end product of this repository is a gradio application where it returns the contents of the tables using OCR tesseract. 

This repository includes 3 notebooks (for a better understading of the dataset and pretrained model used) and 6 scripts. 
The notebooks are: 
1. Marmot_EDA.ipynb
2. albumentation.ipynb
3. vgg19_understanding.ipynb

# Introduction 

The task here is to extract tables from annotated pages where it will be able to return the content in the tables. Firstly, training of model is needed which detects the table and column regions in each image and create masks. Secondly, we use OCR tesseract to predict the content in the detected table and column masks. Lastly, with the predictions, we create an appplication using Gradio where users are able to input an image and retrieve an output of the tables in a dataframe format. 

# Marmot Dataset 

The original Marmot dataset contains both English and Chinese annotated pages, but for this model, only the English pages are used. Image data is in .bmp (bitmap image file) format and there are 509 different images in the dataset whereas there are 510 column and table masks files. However, cleaning up of data is not required as it does not affect the training of model. This dataset containing images, column masks and table masks, can be found in the `data` directory in `data.zip`. For more details of the dataset, it can be explored in `Marmot_EDA.ipynb`. 

In `dataset.py`, Lightning_MarmotDataset() is used to train the dataset with pytorch lightning. 

# Model Weights 

The model is trained with a pretrained vgg19 model as the enocder and creates table and column decoders respectively. The model weights is also available [here]('https://drive.google.com/file/d/1aJfBOwOk6F2wRS0wRevZFGB9cZkDv_Sy/view?usp=sharing') where it was trained for 56 epochs with gradient clipping. 

| Model | No. of epochs | Validation Loss | Binary mean IOU for table | Binary mean IOU for column |
|-------|---------------|-----------------|---------------------------|----------------------------|
|tablenet_baseline_adam_gradclipping| 56 | 0.212 | 0.753 | 0.689 | 

For a better understanding of the pretrained VGG19 model, refer to `vgg19_understanding.ipynb` in `notebooks' directory. 

`metrics.py` contains the loss function, Dice Loss and an evaluation metric, Binary Mean IOU. 

Users are to create a new directory `pretrained_model` and save the model weights inside. If you are training your own model, do rename it to `tablenet_baseline_adam_gradclipping.ckpt` to be able to run gradio_demo.py. 

In `model.py`, Lightning_TableNet() creates a pytorch lightning model for training. 

# Getting Started 

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

# Execution 

To train the model, type the following command: 
```
$ python train.py
```

To create a gradio application, ensure that examples and model path has already been downloaded and type the following command: 
```
$ python gradio_demo.py
```


