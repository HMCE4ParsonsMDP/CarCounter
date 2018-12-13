import numpy as np
import cv2

spots = [#[(1100, 1344), (323, 300), 0, 0],
[(373, 410), (485-373, 525-410), 0, 0],
[(240, 410), (345-240, 525-410), 0, 0],
#[(130, 150), (300, 300), 0, 0],
    #[(480, 150), (300, 300), 0, 0]
]

def main():
    cap = cv2.VideoCapture('C0005-720.MP4')
    #cap = cv2.VideoCapture(0)


    cap.set(cv2.CAP_PROP_EXPOSURE,-10)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    fgbg = cv2.createBackgroundSubtractorMOG2()
    
    framecount = -1
    ret, frame = cap.read()
    prevgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    while(True):
        ret, frame = cap.read()
        framecount += 1
        print("frame:", framecount)
        if framecount < 400:
            fgmask = fgbg.apply(frame)
            print("training")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


        fgmask = fgbg.apply(frame, learningRate=0.001)
        flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        prevgray = gray

        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

        for spot in spots:
            diff, flowval = check_spot(frame, flow, fgmask, spot[0], spot[1])
            if diff > 50 and flowval > 0.1 and spot[2] == 0:
                spot[3] = 1 - spot[3]
                spot[2] = 1
            elif flowval < 0.01:
                spot[2] = 0

            color = (0,255,0)
            if spot[3] == 1:
                color = (0,0,255)
            pos = spot[0]
            size = spot[1]
            cv2.rectangle(frame,pos,(pos[0]+size[0],pos[1]+size[1]),color,3)
            print("Spot:", spot, diff, flowval)
        print("Full spots:", sum([spot[3] for spot in spots]))

        fl = draw_flow(frame, flow)
        #cv2.imshow('flow', fl)
        #cv2.imshow('frame',fgmask)
        outputdir = "frames/"
        numStr = '{0:05d}'.format(framecount)
        cv2.imwrite(outputdir+'fl' + numStr + ".jpg", fl)
        cv2.imwrite(outputdir+'fgmask' + numStr + ".jpg", fgmask)
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

def draw_flow(vis, flow, step=16):
    h, w = vis.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis


def check_spot(frame, flow, mask, pos, size):
    (x, y) = pos
    (h, w) = size
    crop_img = mask[y:y+h, x:x+w]
    crop_flow = np.array(flow)[y:y+h, x:x+w]
    flowval = np.square(crop_flow).mean()

    #cv2.imshow('cropped', crop_img)
    mean = np.mean(crop_img)
    return (mean, flowval)

if __name__ == "__main__":
    main()


