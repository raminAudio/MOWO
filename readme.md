# MOWO
## Mouse Only Walk Oooh

This repo contains training and inference scripts for tracking a mouse on a transparent treadmill (https://mousespecifics.com/digigait/).

As shown in some of the snapshots below, the camera quality is not great. The treadmill is transparent, however, overtime it gets dirty and scratches makes the video quality even lower. As a result, the program that comes with this treadmill makes inaccurate predictions about the walking patterns of a mouse/rat. Normally, researchers are interested on the number of steps per its deviation from a regular pattern to see if a particular operation or disease has improved or detriment the animal's walking.

As with most machine learning tools data, "good data", is required. To overcome the lack of labeled data, I used image processing tools to create several heuristics on some of the videos (and from those videos only a selected set of frames) that would accurately extract the mouse paws for me.

You can find the notebook here: notebooks/mouse-moving.ipynb  

See Youtube Video Below:
[![IMAGE ALT TEXT HERE](pictures%20and%20videos/walk_snapshot.png)](https://www.youtube.com/watch?v=yv3E0Pz0a1A_)


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


Once the model is trained, inspired by the mentioned repo, I wrote a notebook that used this fine-tuned YOLO model on the mouse videos.

See Youtube video of MOWO in action below.
[![IMAGE ALT TEXT HERE](pictures%20and%20videos/mowo_snapshot.png)](https://www.youtube.com/watch?v=D9q5ykkn8Og_)

This notebook does the following:

  1- Detect the frames where the mouse is moving. This step was also done during data pre-processing for training. The reason for this is because we're only interested in how the mouse is walking (not how the mouse is not walking). It also made it easier for the image processing tool to detect paws, but that's just a bonus.

  2- Once we have a shorter version of the mouse video only walking, it is passed to some pre-processing steps before it is passed to MOVO . This step output boundary boxes for each detected paw. Normally two or three paws depending on how the mouse is walking.

  3- Outputs from step 2 is then passed to few analytics which calculate number of steps and deviation for each paw.

![alt text](pictures%20and%20videos/step.png)

Figure below shows a "walking scores" for each paw (FR is front right and BL is back left).
                                [[FR BR],
                                 [FL,BL]]

Please keep in mind for all the figures below, the window size (i.e., number of frames) affects how each plot look like. It's important to keep that constant when comparing different cases. Also, a reminder that only walking frames are being analyzed, so a sudden shift in the distribution doesn't mean the mouse jumped to the back of the treadmill, it simply means the mouse wasn't walking during those frames, so when the mouse started to walk again it was at the back of the treadmill.                               

Each point on these subplot represent the average number of steps the mouse has taken on the corresponding paw over 400 frames. Frames were overlapped 50% and frames where a mouse paw was not detected by MOWO were not accounted when calculating these scores. Note that MOWO's inaccuracies when detecting a paw is not a big deal for this metric, as it's an average over several frames. So, what does these four subplots tell you about the mouse walking?

Take front left paw on top left subplot with the score 0.897. The mouse seems to have been walking an average of 9-10 steps per 400 frames for most part of the video. You see a similar patten for every other paw. Around index 25-30 there is not as many steps on the left paws. This maybe be a mistake or perhaps the mouse is having some issues with the paws on left, TBD.

Now let's look at the four subplots below.

![alt text](pictures%20and%20videos/xy.png)


Each point in the subplot shows the mouse paw location on the X and Y axis of the image (horizontal and vertical) averaged over 400 frames. One thing to note is that in the case where a model was not able to detect a paw the x and y for the paw is replaced by the average of the x's and y's of that paw. This mainly useful to compare how each paw is utilized by the mouse. There are some sudden peaks in all four paws on the X axis that seem to match with the low left paws walking scores from earlier. Y axis seems to be more or less constant for most frames.


Next subplots shown below are polar plots of each frame where a paw was detected. You can think of these plots to represent the direction and the range for each paw. Take caution when considering outliers though (e.g, the markers at 1000 radius shown for the BR paw).

![alt text](pictures%20and%20videos/polar.png)

Last but not least, the two figures below show the angles and radius for each frame. Ignore the values, look at them relative to each other to find out whether the mouse was using one paw more than another, etc.  

Angles
![alt text](pictures%20and%20videos/angles.png)

Radius
![alt text](pictures%20and%20videos/radius.png)

That's pretty much it. A lot of these analytics is just something I tried to derive to see how a mouse is doing on a treadmill, but you may very well wants to use the location of the paws to come up with your own statistic. If you have any questions or want to discuss other features feel free to drop me a line at ramin.audio@gmail.com
