
# Facial Keypoint Detection

## Project Overview

In this project, I've designed a LeNet based Deep CNN to predict the locations of facial keypoints on each face.
The model is inspired by the architecture described in [NaimishNet](https://arxiv.org/abs/1710.00977) and the dataset 
used for training is the [Youtube Faces dataset](https://www.cs.tau.ac.il/~wolf/ytfaces/).
I completed this project as part of Udacity's course on Computer Vision. 

<img src="images/key_pts_example.png" width="80%">

Facial keypoints (also called facial landmarks) are the small magenta dots shown on each of the faces in the image above. In each training and test image, there is a single face and 68 keypoints, with coordinates (x, y), for that face. These keypoints mark important areas of the face: the eyes, corners of the mouth, the nose, etc. These keypoints are relevant for a variety of tasks, such as face filters, emotion recognition, pose recognition, and so on. Here they are, numbered, and you can see that specific ranges of points match different portions of the face.

<img src="images/landmarks_numbered.jpg" width="150px" >

# Setup & Usage

1. Create and configure the Python environment

```bash
conda env create --file facial_keypoint.yml
conda activate facial_keypoint
python -m ipykernel install --user --name=facial_keypoint
```
2. Run the Training Notebook : [Training Notebook](1_Train.ipynb)

3. Inference. Run the model on unseen images : [Inference Notebook](2_Inference.ipynb)





LICENSE: This project is licensed under the terms of the MIT license.
