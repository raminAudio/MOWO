def extract_walking_frames(meanxs, offset):
    '''
    meanxs : the maximum index from the filtered means of subtracted frames 
    offset : number of frames between each step 
    local min and local max of meanxs are used to detect if a mouse is walking. 
    Outputs : indexes where the mouse is walking 
    '''
    
    b, a = signal.butter(5, 0.3)
    y = signal.filtfilt(b, a, np.array(meanxs), padlen=10) # more filtering 

    local_max_idx = argrelextrema(y, np.greater) 
    local_min_idx = argrelextrema(y, np.less)

    walking_frames = []
    for imin,imax in zip(local_min_idx[0],local_max_idx[0]):
        if abs(imax-imin)<10:
            walking_frames.append((min(imax+offset,imin+offset),max(imax+offset,imin+offset)))
    return walking_frames

def merge_frame_indicies(walking_frames,thr_merging,thr_save):
    '''
    walking_frames : output from extract_walking_frames contains walking frame indicies 
    thr_merging  : Merges groups of walking frames together to compensate for mistakes from the heuristics
    thr_save : Threshol for saving group of frames 
    '''
    results = []
    count = 0 
    tuples = np.concatenate((walking_frames))

    for tuple_min_max in tuples:
        x = tuple_min_max[0]
        y = tuple_min_max[1]
        if count == 0:
            so_far = [x,y]
            count += 1
        else:
            if x - so_far[1] < thr_merging: 
                so_far[1] = y
            else:
                if so_far[1] - so_far[0] > thr_save:
                    results.append(so_far)
                count = 0
    return results

def split_video_to_walking_only(files_abs_path, disp = 0):
    
    # Figuring out the frames where the mouse is walking 
    offset = 400

    batch = []
    images = []
    dict_video = {}

    # Looping over frames for each video to extract walking frames 

    walking_frames = []

    count = 0

    for frame_path in files_abs_path:

        try:
            # Read each image and averaging over all channels 
            image_data = plt.imread(frame_path)
            image_data = np.int32(np.mean(image_data,axis = -1 ))
            batch.append(image_data)
        except:
            x = 1 

        print(len(batch),end='\r')

        if len(batch) >= offset:  

            o1 = offset * count # keeping track of where we are in the frames 
            count += 1 
            meanxs = [] 
            kk = 1
            count_index = 0
            batch_arr = np.array(batch).squeeze() # batch of frames 
            diff_batch = batch_arr[0:-1,:,:] - batch_arr[1:,:,:] # subtracting the frames 

            for xx in diff_batch:
                count_index += 1 
                mean_x = np.mean(xx[:,0:600]>0,axis=0) # avergaing over pixels (removing the treadmill)
                b, a = signal.butter(8, 0.025) # filter for smoothing mean
                y = signal.filtfilt(b, a, np.array(mean_x), padlen=150)
                meanxs.append(np.argmax(y)) # finding the max index of filtered version of mean signals 
                if disp:
                    # Display 
                    figure, ax = plt.subplots(2,2,figsize=(20,10))
                    
                    ax[0][0].imshow(xx>0)
                    ax[0][0].set_title(" mouse activity ")
                    ax[0][1].imshow(batch_arr[count_index])
                    ax[0][1].set_title(" frame ")

                    ax[1][0].plot(np.arange(kk),np.array(meanxs))
                    ax[1][0].set_title(" walking signal over time ")
                    ax[1][1].plot(y)
                    ax[1][1].set_title(" frame walking signal smoothed ")

                    kk += 1 
                    plt.show()
                    time.sleep(0.001) 
                    clear_output(wait=True)

            # extracting Walking frames 
            walking_frames.append(extract_walking_frames(meanxs,o1))
            batch = []

    # collect all frames, concatenate, merge...
    thr_merging = 30
    thr_save = 50
    final_walking_frames = []
    walking_frame_intervals = merge_frame_indicies(walking_frames,thr_merging,thr_save)
    for frame in walking_frame_intervals:
        for i in range(frame[0],frame[1]):
            final_walking_frames.append(str(i) + '.jpg')

    pickle.dump(final_walking_frames , open('walking_frames_all_videos' +video_name + '.pickle','wb'))        

    print("Ratio of walking frames over not walking " + str(len(final_walking_frames)/len(files_abs_path)))
    return final_walking_frames

def paw_detection_video(files_abs_path, disp):
    # Performing inference and plotting it on top each frame 
    count = 0 
    dict_paws = {}

    for f in files_abs_path:
        image_name = f.split('/')[-1]
        if image_name in final_walking_frames: # performing inference only when the mice is walking
            image = Image.open(f)
            image_data = np.array(image)
            image_ = Image.fromarray(image_data.astype('uint8'), 'RGB')
            image,locations, labels = yoloObj.detect_image(image_)
            if disp:
                figure, ax = plt.subplots(1,1,figsize=(40,20))
                ax.imshow(image)
                plt.show()
                time.sleep(0.1)
                clear_output(wait=True)
            count += 1
            if not disp:
                print(round(count/len(final_walking_frames),3), end='\r')
            for location,label_ in zip(locations,labels):
                ymin = location[0]
                xmin = location[1]
                ymax = location[2]
                xmax = location[3]
                centery = (ymin+ymax)/2
                centerx = (xmin+xmax)/2
                label = ' '.join(label_.split()[0:2])
                array_ = np.zeros((2,count+1))
                array_[0][count] =  centerx 
                array_[1][count] =  centery
                if label not in dict_paws.keys():
                    dict_paws[label] = array_
                else:
                    existing_array = dict_paws[label]
                    array_[0][:len(existing_array.T)] +=  existing_array[0]
                    array_[1][:len(existing_array.T)] +=  existing_array[1]
                    dict_paws[label] = array_

    pickle.dump(dict_paws,open(video_name + 'dict_paws.pickle','wb'))
    return dict_paws

def cal_num_steps(X,Y, window = 100):
    '''
    finding the number of steps
    '''
    w = [0,window]
    b, a = signal.butter(5, 0.1)
    while w[1] < len(X):
        XY   = (X+Y)[w[0]:w[1]]
        XY = signal.filtfilt(b, a, np.array(XY), padlen=10) 
        fft_br = np.abs(fft(XY))
        num_steps = np.argmax(fft_br[1:int(len(fft_br)/2)])
        w[0] += int(window/2)
        w[1] += int(window/2)
        yield num_steps
        
def step_analytics_video(dict_paws, wn = 200):
    x   = dict_paws['back right']
    Xbr = x[0]
    Ybr = x[1]
    x   = dict_paws['back left']
    Xbl = x[0]
    Ybl = x[1]
    x   = dict_paws['front right']
    Xfr = x[0]
    Yfr = x[1]
    x   = dict_paws['front left']
    Xfl = x[0]
    Yfl = x[1]

    genfr = cal_num_steps(Xfr,Yfr, window = wn)
    genbr = cal_num_steps(Xbr,Ybr, window = wn)
    genfl = cal_num_steps(Xfl,Yfl, window = wn)
    genbl = cal_num_steps(Xbl,Ybl, window = wn)

    num_steps_fr = []
    num_steps_br = []
    num_steps_bl = []
    num_steps_fl = []
    # Front Right
    while True: 
        try:
            num_steps_fr.append(next(genfr))
        except StopIteration:
            break
    # Back Right
    while True:
        try:
            num_steps_br.append(next(genbr))
        except StopIteration:
            break
    # Back Left
    while True:
        try:
            num_steps_bl.append(next(genbl))
        except StopIteration:
            break
    # Front Left
    while True:
        try:
            num_steps_fl.append(next(genfl))
        except StopIteration:
            break


    figure, ax = plt.subplots(2,2,figsize=(20,10))

    thr = 2 # minimum number of steps to accept (ideally this is 1, but the model makes mistakes)
    nfr = np.array(num_steps_fr) >= thr 
    scoreFR = np.sum(nfr)/len(nfr)
    print("Avg Number of steps over " + str(wn) + " with 50% overlap")
    ax[0][0].plot(num_steps_fr)
    ax[0][0].set_title('FR score ' + str(scoreFR))

    nbr = np.array(num_steps_br) >= thr 
    scoreBR = np.sum(nbr)/len(nfr)
    ax[0][1].plot(num_steps_br)
    ax[0][1].set_title("BR score " + str(scoreBR))

    nbl = np.array(num_steps_bl) >= thr 
    scoreBL = np.sum(nbl)/len(nbl)
    ax[1][0].plot(num_steps_bl)
    ax[1][0].set_title("BL score " +  str(scoreBL))

    nfl = np.array(num_steps_fl) >= thr 
    scoreFL = np.sum(nfl)/len(nfl)
    ax[1][1].plot(num_steps_fl)
    ax[1][1].set_title("FL score " +  str(scoreFL))
    
    print("Total Average Score : " + str(np.sum(scoreFL+scoreFR+scoreBL+scoreBR)/4))
    plt.show()

    figure, ax = plt.subplots(2,2,figsize=(20,10))
    paw_names = ['front right', 'front left', 'back left','back right']
    indexes = [(0,0),(0,1),(1,0),(1,1)]
    print(' Average change in axis over a ' + str(wn) + ' frames with 50% overlap')
    for paw_name,index in zip(paw_names,indexes):
        x = dict_paws[paw_name][0]
        x[x == 0] += 100#np.mean(x)
        y = dict_paws[paw_name][1]
        y[y == 0] += 100#np.mean(y)
        b = [0, wn]
        xmu_list = []
        ymu_list = []

        while b[1] < len(x):
            x_mean = np.mean(x[b[0]:b[1]])
            y_mean = np.mean(y[b[0]:b[1]])
            xmu_list.append(x_mean)
            ymu_list.append(y_mean)
            b[0] += int(wn/2)
            b[1] += int(wn/2)
        ax[index[0]][index[1]].plot(xmu_list)
        ax[index[0]][index[1]].plot(ymu_list)
        ax[index[0]][index[1]].legend(['x','y'])
        ax[index[0]][index[1]].set_title(paw_name + " std x " +  str(round(np.std(xmu_list),2)) + " std y " + str(round(np.std(ymu_list),2)))

    plt.show()
    
    paw_names = ['front right', 'front left', 'back left','back right']
    indexes = [(0,0),(0,1),(1,0),(1,1)]
    print(' Polar plots per each frame where a paw was detected')
    figure, ax = plt.subplots(2,2,figsize=(15,15),subplot_kw={'projection': 'polar'})
    for paw_name,index in zip(paw_names,indexes):
        x = dict_paws[paw_name][0]
        y = dict_paws[paw_name][1]
        x = x[x != 0]
        y = y[y != 0]

        r_mean = np.sqrt(np.power(x,2)+np.power(y,2))
        theta_mean = np.arctan(y/x)

        ax[index[0]][index[1]].plot(theta_mean,r_mean,'x')
        ax[index[0]][index[1]].grid(True)
        ax[index[0]][index[1]].set_title(paw_name)
    plt.show()
    
    
    
    
    x = dict_paws['front right'][0]
    y = dict_paws['front right'][1]
    r_meanFR = np.sqrt(np.power(x,2)+np.power(y,2))
    theta_meanFR = np.arctan(y/(x+0.000001))
    
    x = dict_paws['front left'][0]
    y = dict_paws['front left'][1]
    r_meanFL = np.sqrt(np.power(x,2)+np.power(y,2))
    theta_meanFL = np.arctan(y/(x+0.000001))
    
    x = dict_paws['back right'][0]
    y = dict_paws['back right'][1]
    r_meanBR = np.sqrt(np.power(x,2)+np.power(y,2))
    theta_meanBR = np.arctan(y/(x+0.000001))
    
    x = dict_paws['back left'][0]
    y = dict_paws['back left'][1]
    r_meanBL = np.sqrt(np.power(x,2)+np.power(y,2))
    theta_meanBL = np.arctan(y/(x+0.000001))
    
    figure, ax = plt.subplots(1,1,figsize=(15,15))
    ax.plot(theta_meanFR+4)
    ax.plot(theta_meanFL+2)
    ax.plot(theta_meanBR)
    ax.plot(theta_meanBL-2)
    ax.legend(['FR','FL','BR','BL'])
    ax.set_title("Paw angles between -Pi to Pi for each frame - 0 if not detected")
    plt.show()
    figure, ax = plt.subplots(1,1,figsize=(15,15))
    ax.plot(r_meanFR+600)
    ax.plot(r_meanFL+300)
    ax.plot(r_meanBR)
    ax.plot(r_meanBL-300)
    ax.legend(['FR','FL','BR','BL'])
    ax.set_title("Radius")
    plt.show()
