# Bird_Song_App
#### Author: Santiago Miro 2024

This repository contains all the files used for building a bird song classification model and web app.

The purpose of this application is to develop and train a CNN model for classification of bird songs between different type of species. At this moment there are only five birds in the training catalogue:

* Bewick's Wren
* Northern Cardinal
* American Robin
* Song Sparrow
* Northern Mockingbird

The data was obtained from https://www.kaggle.com/datasets/vinayshanbhag/bird-song-data-set and sourced from https://www.xeno-canto.org/.

## Table of Contents:

* __Data Processing and Training__: This folder includes both notebooks for data processing and the model training.
* __static__: This folder includes the css file for style and the images displayed on the website.
* __templates__: This folder contains the .html files for the wev development.
* __app.py__: The Flask web app script.
* __cnn_model.keras__: The trained CNN model.
* __Procfile__: Heroku file for web app cloud deployment.
* __requirements.txt__: Dependencies needed for the app to run in the VM.

## Data Processing

Data set includes only "songs" from 5 species.

For simplicity, data set excludes other types of calls (alarm calls, scolding calls etc) made by these birds. Additionally, only recordings graded as highest quality on xeno-canto API are included.

Original recordings from xeno-canto were in mp3. All files were converted to wav format. Further, using onset detection, original recordings of varying lengths were clipped to exactly 3sec such that some portion of the target bird's song is included in the clip.
Original mp3 files from the source have varying sampling rates and channels. As part of the conversion process each wav file was sampled at exactly 22050 samples/sec and is single channel.

CSV file includes recording metadata, such as genera, species, location, datetime, source url, recordist and license information.
The filename column in CSV corresponds to the wav files under wavfiles folder.
