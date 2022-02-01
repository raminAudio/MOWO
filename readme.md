# MOWO
## Mouse Only Walk Oooh

This repo contains training and inference scripts for tracking a mouse on a transparent treadmill (https://mousespecifics.com/digigait/).

As shown in some of the snapshots below, the camera quality is not great. The treadmill is transparent, however, overtime it gets dirty and scratches makes the video quality even lower. As a result, the program that comes with this treadmill makes inaccurate predictions about the walking patterns of a mouse/rat. Normally, researchers are interested on the number of steps per its deviation from a regular pattern to see if a particular operation or disease has improved or detriment the animal's walking.

As with most machine learning tools data, "good data", is required. To overcome the lack of labeled data, I used image processing tools to create several heuristics on some of the videos (and from those videos only a selected set of frames) that would accurately extract the mouse paws for me.

You can find the notebook here: notebooks/mouse-moving.ipynb  

See /pictures and videos/video_1.mov
![alt text](pictures%20and%20videos/shot1.png)
![alt text](pictures%20and%20videos/shot2.png)
![alt text](pictures%20and%20videos/shot3.png)

The subplot in the right bottom represents where the mouse activities in the frame are (look at the maximum of the curve).  

The subplot on bottom right depicts a walking signal. If the walking signal keeps going up, it means the mouse is not walking during those frames. For more information see the mouse-moving notebook.


I used the boundary boxes generated from such heuristics to create a training dataset that resembles an object localization dataset. The amount of generated data wasn't that much to train a model from scratch. Data augmentation didn't make much sense here as the camera angel and colors are fairly consistent for DigiGait. Though, I did try to augment my data by rotating the mouse to capture the paw from different angels, I later found out that my dataset contains several of those examples without me having to synthetically rotate the image and its boundary boxes.



Hence, I decided to fine-tuned an existing model which is where I came across YOLO (You Only Look Once , https://arxiv.org/abs/1506.02640).

You could find some results from running YOLO on different datasets and how fast it can detect objects (https://pjreddie.com/darknet/yolo/). There are several files and scripts that are needed to train a YOLO model, most of which you will find in the link I mentioned earlier.

As a heavy Tensorflow uses, and a lazy engineer, I started to look for a repo where someone has taken the weights and re-implemented YOLO with Tensorflow which is how I came across (https://github.com/qqwweee/keras-yolo3). This repo contains most of the fine-tuning scripts as well as a sample code for running inference.

I grabbed and modified some of their codes to match what I have for my training dataset.

You can find a copy of my training notebook here: notebooks/MOVO.ipynb  

This notebooks starts by generating features from YOLO which are then passed through the pre-trained YOLO model (YoloV3)  in two different round. First, with keeping only layer weights fixed except the last two, second by training the weights for all layers (as was suggested and implemented in https://github.com/qqwweee/keras-yolo3).

Image below are the boundary boxes detected by the image processing tool. Mouse center was used to distinguish between front and back paws.
![alt text](pictures%20and%20videos/shot4.png)

Image below are the boundary boxes predicted by the fine-tuned YOLOV3 model (MOVO as I call it, mouse only walk oooh, original I know).

![alt text](pictures%20and%20videos/shot5.png)

![](pictures%20and%20videos/video_2.mov)


Once the model is trained, inspired by the mentioned repo, I wrote a notebook that used this fine-tuned YOLO model on the mouse videos that,

  1- Detect the frames where the mouse is moving. This step was also done during data pre-processing for training. The reason for this is because we're only interested in how the mouse is walking (not how the mouse is not walking). It also made it easier for the image processing tool to detect paws, but that's just a bonus.

  2- Once we have a shorter version of the mouse video only walking, it is passed to some pre-processing steps before it is passed to MOVO . This step output boundary boxes for each detected paw. Normally two or three paws depending on how the mouse is walking.

  3- Outputs from step 2 is then passed to few analytics which calculate number of steps and deviation for each paw.

![alt text](pictures%20and%20videos/shot6.png)

See /pictures and videos/video_2.mov

That's pretty much it.
