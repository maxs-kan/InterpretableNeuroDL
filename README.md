# "Brain Differences Between Men and Women: Evidence From Deep Learning"

### 5 Nearest_Neighbors_team

We analyze data provided by The Human Connectome Project (HCP).  Using full-size MRI data we solve the problem of binary classification of finding the sex of an object.

## Setup and Dependencies

- Python

- numpy

- pytorch 

- nilearn



## DATABASE 

We used part of the available data (517 objects), including 307 women and 210 men. Each object is represented by a 1 GB ZIP archive with a name corresponding to a unique object ID. Each archive contains a lot of different information. For automatic access to the target MRI file, the power shell script was written that can be found in DATE_ACCESS.md. The script allows you to extract the necessary file from the internal ZIP archive(inside the main archive), without unzipping the main one to save time. Also, a unique ID corresponding to each object is assigned as a name for each file.



## CNN Model

![](D:\Brain_Differences_project_ML2020\Capture.PNG)

## Training

bla bla