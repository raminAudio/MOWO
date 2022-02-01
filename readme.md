This repo contains my work for tracking a mouse on a treadmill (https://mousespecifics.com/digigait/)

As shown in some of the snapshots below, the camera quality is not great. The treadmill is transparent, however, overtime it gets dirty and scratches makes the video quality even lower. As a result, the program that comes with this treadmill makes inaccurate predictions about the walking patterns of a mouse/rat. Normally, researchers are interested on the number of steps per its deviation from a regular pattern to see if a particular operation or disease has improved or detriment the animal's walking.

As a machine learning engineer I wanted to see if I can do better, even if it is not objective, than what I heard from researchers using this tool. As with most machine learning tools data, "good data", is required. I neither have the time to draw a boundary box around each of the mouse paw nor couldn't use any of the data predicted from DigiGait as ground truth.

To overcome the lack of labeled data, I used image processing tools to create several heuristics on some of the videos (and from those videos only a selected set of frames) that would accurately extract the mouse paws for me.

You can find the notebook here: notebooks/mouse-moving.ipynb  

You can see the videos under : picture and videos/

![alt text](pictures%20and%20videos/shot1.png)
![alt text](pictures%20and%20videos/shot2.png)
![alt text](pictures%20and%20videos/shot3.png)

The "curvey" plots under the mouse represent where the mouse activities in the frame are (look at the maximum of the curve).  


I used the boundary boxes generated from such heuristics to create a training dataset that resembles an object localization dataset. The amount of generated data wasn't that much to train a model from scratch. Data augmentation didn't make much sense here as the camera angel and colors are fairly consistent for DigiGait. Though, I did try to augment my data by rotating the mouse to capture the paw from different angels, I later found out that my dataset contains several of those examples without me having to synthetically rotate the image and its boundary boxes.



Hence, I decided to fine-tuned an existing model which is where I came across YOLO (You Only Look Once)

You could find some results from running YOLO on different datasets and how fast it can detect objects (https://pjreddie.com/darknet/yolo/). There are several files and scripts that are needed to train a YOLO model, most of which you will find in the link I mentioned earlier.

As a heavy Tensorflow uses, and a lazy engineer, I started to look for a repo where someone has taken the weights and re-implemented YOLO with Tensorflow which is how I came across (https://github.com/qqwweee/keras-yolo3). This repo contains most of the fine-tuning scripts as well as a sample code for running inference.

I grabbed and modified some of their codes to match what I have for my training dataset.

You can find a copy of my training notebook here: notebooks/train-bottleneck.ipynb  

This notebooks starts by generating features from YOLO which are then passed through the pre-trained YOLO model (YoloV3)  in two different round. First, with keeping only layer weights fixed except the last two, second by training all layers unfreezed (as was suggested and implemented in https://github.com/qqwweee/keras-yolo3).

![alt text](pictures%20and%20videos/shot4.png)
![alt text](pictures%20and%20videos/shot5.png)


Once the model is trained, inspired by the mentioned repo, I wrote a notebook that used this fine-tunned YOLO model on the mouse videos that,
  1- Detect the frames where the mouse is moving. This step was also done during data pre-processing for training. The reason for this is because we're only interested in how the mouse is walking (not how the mouse is not walking). It also made it easier for the image processing tool to detect paws, but that's just a bonus.
  2- Once we have a shorter version of the mouse video only walking, it is passed to some pre-processing steps before it is passed to MOVO (mouse only walk oooh, original I know.). This step output boundary boxes for each detected paw. Normally two or three paws depending on how the mouse is walking.
  3- Outputs from step 2 is then passed to few analytics which calculate number of steps and deviation for each paw.

![alt text](pictures%20and%20videos/shot6.png)



That's pretty much it.
