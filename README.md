# Cat vs Dog Classifier

---

## Introduction
This project implements a deep learning classifier to distinguish between cat and dog images using **PyTorch**.  
It demonstrates data preprocessing, model training, evaluation, and prediction workflows.  

The [Cat and Dog Dataset on Kaggle](https://www.kaggle.com/datasets/tongpython/cat-and-dog) is organized into two classes (`cats/` and `dogs/`), stored in separate subdirectories.

---

## Project Structure
classification_cat_dog/ <br>
│── assets/  <br>
│── inference/ <br>
│── models/ <br>
│── train/ <br>

---

## Content

The training pipeline for image classification is located under `train/`.  
The backbone of the model is **EfficientNet-B0**, followed by custom CNN layers.  
It can either be trained from scratch or initialized with pretrained EfficientNet-B0 weights.  

**Training Results:** <br>

Accuracy without pretrained weights: **0.7829** <br>
<div align="center">
<img src="assets/training_model.png" alt="Training without pretrained weights" height="300"/> 
</div>
<br>

Accuracy with pretrained weights: **0.9683** <br>
<div align="center">
<img src="assets/training_pretrained_model.png" alt="Training with pretrained weights" height="300"/>
</div>

<br>
The best models from each training run are saved under `models/` and can later be used for inference.  

Inference scripts are located under `inference/`, where predictions can be run on the test dataset or on custom images.  

<p>
A demonstration of inference on custom images is provided in the notebook  
<a href="example_notebook.ipynb">example_notebook.ipynb</a>.
</p>

<br>
<div align="center">
<img src="assets/output_cat.png" alt="Inference example - Cat" height="300"/> 
</div>

<div align="center">
<img src="assets/output_dog.png" alt="Inference example - Dog" height="300"/>
</div>

---

## Installation 

Creat python env 
```bash 
conda create -n classification_cat_dog python==3.10 
conda activate classification_cat_dog 
``` 
Clone the repository and install dependencies: 
```bash 
git clone https://github.com/WJannik/classification_cat_dog.git 
cd classification_cat_dog 
pip install -r requirements.txt
```
