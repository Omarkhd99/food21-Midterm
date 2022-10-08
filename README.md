# Food Regconition Application
#### What's your favorite food?
---

Steven  L Truong

---

![](https://img.shields.io/badge/PYTHON-blue?style=for-the-badge)![](https://img.shields.io/badge/keras-green?style=for-the-badge)![](https://img.shields.io/badge/heroku-blueviolet?style=for-the-badge)![](https://img.shields.io/badge/STREAMLIT-red?style=for-the-badge)![](https://img.shields.io/badge/TENSORFLOW-pink?style=for-the-badge)

## Abstract
---
In this project, I build a web application to regconize over **130** kinds of food using Computer Vision and Transfer Learning. This is an extended implementation of this original paper [Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) (which only regconize 101 kinds of food). 

---

The app can be accessed via [this link](https://share.streamlit.io/luongtruong77/food_130_recognition/main/app.py). 


## Data
---
In this project, I will use 2 datasets to train my model:
#### The Food-101 Data Set
![](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/static/img/food-101.jpg)

This challenging data set has 101 food categories, with 101'000 images. For each class, 250 manually reviewed test images are provided as well as 750 training images. On purpose, the training images were not cleaned, and thus still contain some amount of noise. This comes mostly in the form of intense colors and sometimes wrong labels. All images were rescaled to have a maximum side length of 512 pixels.

#### The 30VNFoods Data Set
![](https://github.com/luongtruong77/food_130_recognition/blob/main/figures/banh_mi.jpg?raw=true)

This dataset has 30 Vietnamese food categories, and collected by [Quan Dang](https://github.com/18520339). It has the most authentic Vietnamese food categories, and is good for this purpose of building this app.
Since `pho` is in both datasets, the combined dataset has 130 categories (instead of 131).

## Algorithms
---
##### Transfer Learning (feature extraction)
To build my base model for feature extraction, I use [EfficientNet Architectures](https://tfhub.dev/s?module-type=image-classification,image-feature-vector&tf-version=tf2&q=efficientnet).

##### Fine Tuning:
Fine tune in the entire unfrozen layers and the final model has the accuracy of **73%**.

#### Deployment
Streamlit web app is built and deployed to Heroku.


## Tools
---
- Python
- Tensorflow
- Keras
- Pandas
- Numpy
- Streamlit
- Heroku


















