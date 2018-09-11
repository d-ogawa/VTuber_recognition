# VTuber_recognition
This repository contains python codes for classifing Virtual YouTuber images.

## Environment

- Ubuntu 16.04
- Python 3.5.2
- Tensorflow 1.4.0
- Opencv 3.4.1

## Making dataset

```
python download_videos.py
```
for downloading VTuber videos.

```
python movie2face.py
```
for extracting face images and then exclude outlier images manually.

```
python make_tfrecords.py
```
for making TFRecords files.

## Train

Using Supervisor class, 
```
python train.py
```

Using MonitoredTrainingSession class, 
```
python train_byMT.py
```

## Test

Using Supervisor class, 
```
python test.py
```

Using MonitoredTrainingSession class, 
```
python test_byMT.py
```
