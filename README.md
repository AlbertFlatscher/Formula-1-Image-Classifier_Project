# Formula-1-Image-Classifier
Building an Image Classifier Capable of Recognizing Different F1 Teams as Classes

### Table of Contents
 
1. [Project Motivation](#motivation)
2. [Data](#data)
3. [Files](#files)
4. [Libraries](#libraries)
5. [Results](#results)
6. [Licensing](#licensing)
7. [Code Execution](#execution)

## Project Motivation <a name="motivation"></a>

This is the Capstone project of my Udacity Data Scientist Nanodegree Program. We try to create an algorithm which is capable of classifying all the Formula 1 cars in an image into their respective manufacturers. This could be the first step of building a specialised image search engine.

We use transfer learning on a pre trained CNN and train the last layers to our problem at hand. In addition we tackle the problem of identifying more then one class per image. This is done using a pretrained Cascade Classifier for object detection on the image. Each detected object will provide a section of the image to be used on our single-label classifier.

With this approach it should be possible to detect each F1 car in an image one by one. The single-label classification results can be evaluated and if fullfilling certain criteria, added up to give the multi-label result.

## Data <a name="data"></a>
The content of this project consist of the following files:
<ul>
  <li>Image_Classifier_using_Object_Detection.ipynb - Jupyter Notebook containg all the code and detailed documentation about the project
  <li>categories.csv - maps the messages into one of 36 categories
</ul>

```
Formula-1-Image-Classifier/
│
├── Image_Classifier_using_Object_Detection.ipynb
├── README.md
|
├── saved_models/
│   ├── cars.xml
│   ├── cas1.xml
│   ├── cas1.xml
│   └── F1_2019_Image_Classifier.h5
│
├── data/
│   ├── multi_car/ -> training data for single-label classifier
│       ├── alfa_romeo/
│       ├── bwt/
│       ├── ferrari/
│       ├── haas/
│       ├── mclaren/
│       ├── mercedes/
│       ├── redbull/
│       ├── renault/
│       ├── toro_rosso/
│       └── williams/
│   └── multi_car -> test data for final algorithm
│       ├── bwt_ferrari_mclaren/
│       ├── entire_field/
│       ├── ferrari_mercedes/
│       ├── ferrari_redbull/
│       ├── mclaren_renault/
│       ├── mercedes_mclaren/
│       └── mercedes_redbull/
```

## Libraries <a name="libraries"></a>

I used a Jupyter notebook (Python) for the analysis. The ipynb file should run on any Python 3 version (3.0.* or higher) without any problems.</br>

Here are the additional Python libraries used within this project:

<ul>
  <li>os</li>
  <li>pathlib from Path</li>
  <li>numpy as np</li>
  <li>matplotlib.pyplot as plt</li>
  <li>json</li>
  <li>PIL from Image</li>
  <li>cv2</li>
  <li>tensorflow_hub</li>
  <li>tensorflow as tf</li>
  <li>tf_keras</li>
  <li>tf_keras.preprocessing.image from ImageDataGenerator</li>
  <li>tf_keras.optimizers from Adam</li>
  <li>scipy.ndimage from rotate</li>
  <li>sklearn.metrics from accuracy_score, precision_score, recall_score, f1_score</li>
  <li>sklearn.preprocessing from MultiLabelBinarizer</li>
</ul>

## Results <a name="results"></a>

The best performing ML model conained a CountVectorizer, StartingVerbExtractor and a MultiOutputClassifier using AdaBoost as an estimator. The best parameters for the Estimator were: n_estimators=100, learning_rate=0.01.

This enabled us to get the following metrics:
<li>Accuracy: 0.944</li>
<li>Precision: 0.932</li>
<li>Recall: 0.944</li>
<li>F1-score: 0.9301</li>

We were able to to use this pipeline for creating a Web Application which can be used for disaster response coordination.

## Licensing <a name="licensing"></a>

Thanks to [Appen](https://www.appen.com/) for providing the data.

<div class="container">
    <h2>Code Execution <a name="execution"></a></h2>

    # To create a processed sqlite db
    python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db

    # To train and save a pkl model
    python train_classifier.py ../data/DisasterResponse.db classifier.pkl

    # To deploy the application locally
    python run.py
</div>
