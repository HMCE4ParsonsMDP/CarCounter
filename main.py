import numpy as np
import cv2

# our array of spots in the parking lot
# add more spots in the following format:
# [(xcoord, ycoord), (xsize, ysize), 0, 0]

spots = [
    [(373, 410), (485-373, 525-410), 0, 0],
    [(240, 410), (345-240, 525-410), 0, 0],
    #[(1100, 1344), (323, 300), 0, 0],
    #[(130, 150), (300, 300), 0, 0],
    #[(480, 150), (300, 300), 0, 0]
]

def main():
    # input from video file
    cap = cv2.VideoCapture('C0005-720.MP4')

    # or input direclty from camera if it is plugged in
    #cap = cv2.VideoCapture(0)

    # define capture parameters
    yres = 720
    xres = 1280

    # set up parameters if we are using a real camera
    cap.set(cv2.CAP_PROP_EXPOSURE,-10)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,yres)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,xres)

    # create a kernel for smoothing out the background subraction result
    # a size of 3x3 seems to work pretty well
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

    # create the actual background subractor using the MOG2 algorithm
    fgbg = cv2.createBackgroundSubtractorMOG2()
    

    # define variables for use in the loop later
    framecount = -1
    ret, frame = cap.read()
    prevgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    while(True):
        # fetch a new picture from the camera
        ret, frame = cap.read()
        # increment our frame count
        framecount += 1


        # debug information so we can see that the code is running properly
        print("frame:", framecount)

        # give the background subtractor some time to get trained initially because it
        # produces unreliable results while it is still getting an idea of the background
        if framecount < 400:
            fgmask = fgbg.apply(frame)
            print("training")
            # skip everything else in the loop and just run this again
            continue

        ###################
        # main processing #
        ###################

        # run the background subtraction and train it on the same frame
        # a learning rate of 0.001 seems to work well. 0.008 is too slow
        # it also cannot be too much faster or it would not work on vehicles parking slowly
        fgmask = fgbg.apply(frame, learningRate=0.001)

        # use our kernel to clean up the mask a little bit
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

        # create a greyscale version of the image for processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # use the Farenback algorithm to find the movement in the frame
        # these parameters seem to work well and were used from the OpenCV example for this algorithm
        flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # update the previous greyscale image now that we have used it for comparison
        prevgray = gray

        ######################
        # spot counting loop #
        ######################

        for spot in spots:
            # just run our check_spot code on each spot's coordinates
            diff, flowval = check_spot(frame, flow, fgmask, spot[0], spot[1])
            # create a finite state machine with 4 states
            # spot[3] is whether or not the spot is occupied
            # spot[2] is whether or not the spot is in a transition state (car entering or exiting)
            if diff > 50 and flowval > 0.1 and spot[2] == 0:
                spot[3] = 1 - spot[3]
                spot[2] = 1
            elif flowval < 0.01:
                spot[2] = 0

            # display those green or red squares depending on if the spot is empty or full
            color = (0,255,0)
            if spot[3] == 1:
                color = (0,0,255)
            pos = spot[0]
            size = spot[1]
            # actually draw the square
            cv2.rectangle(frame,pos,(pos[0]+size[0],pos[1]+size[1]),color,3)
            # some debug so we can verify that the code is showing the 
            print("Spot:", spot, diff, flowval)

        # print the number of occupied spots in the parking lot
        print("Full spots:", sum([spot[3] for spot in spots]))

        # save images of the different parts of the processing for use in the videos
        # this would be removed in the final version and is here so we can actually see what the code does
        black = np.zeros((512,512,3), np.uint8)
        fl = draw_flow(frame, flow)
        outputdir = "frames/"
        numStr = '{0:05d}'.format(framecount)
        cv2.imwrite(outputdir+'fl' + numStr + ".jpg", fl)
        cv2.imwrite(outputdir+'fgmask' + numStr + ".jpg", fgmask)

        # uncomment to display debug windows (slows down code a lot)
        #cv2.imshow('flow', fl)
        #cv2.imshow('frame',fgmask)

        # some stuff to make it so that opencv can properly display windows and not crash
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


def check_spot(frame, flow, mask, pos, size):
    (x, y) = pos
    (h, w) = size
    # crop the whole image to be just the parking spot we care about
    crop_img = mask[y:y+h, x:x+w]
    # do the same with the computed flow vectors
    crop_flow = np.array(flow)[y:y+h, x:x+w]
    # find the average flow
    flowval = np.square(crop_flow).mean()

    # find the average non-backgroundness
    mean = np.mean(crop_img)

    # return our computed values
    return (mean, flowval)


# modified from the opencv optical flow example
def draw_flow(vis, flow, step=16, scale=1):
    # define the x and y of the center of each vector
    h, w = vis.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)

    # finally get the actual useful flow data
    fx, fy = flow[y,x].T
    # create an array of the ends of the fectors
    lines = np.vstack([x, y, x+scale*fx, y+scale*fy]).T.reshape(-1, 2, 2)

    # draw all of the green lines and circles
    lines = np.int32(lines + 0.5)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)

    return vis

# make python happy and only run the code if you are running the file directly
if __name__ == "__main__":
    main()


