import cv2
import time
import sys
import numpy as np
import pickle as pl
import matplotlib.pyplot as plt
import skimage
from ParamsSetting import generalparams
from Multilook import multilookProcess
from PreProcessing import preProcess
from Levelset import levelsetProcess
from PostProcessing import postProcess


def saveText(FirstTimeSaveText, phiInit, ReInitPhi, shape, outDir, num):
    phiMin, phiMax = phiInit.min(), phiInit.max()
    if(FirstTimeSaveText):
        inform = ["|FRAME " + str(num) + "| ", "phi min:" + str(phiMin), ',',
                  "phi max:" + str(phiMax), ',',
                  "reinitial phi:" + str(ReInitPhi), ',',
                  "shape:" + shape]
        FirstTimeSaveText = False
        with open(outDir + "\\information_bag.txt", "w") as f:
            f.writelines(inform)
            f.close
    else:
        inform = ["\n|FRAME " + str(num) + "| ", "phi min:" + str(phiMin), ',',
                  "phi max:" + str(phiMax), ',',
                  "reinitial phi:" + str(ReInitPhi), ',',
                  "shape:" + shape]
        with open(outDir + "\\information_bag.txt", "a+") as f:
            f.writelines(inform)
            f.close
    return FirstTimeSaveText


def readText(outDir, num):
    with open(outDir + "\\information_bag.txt", "r") as f:
        cnt = 1
        for informs in f.readlines():
            if(cnt == num):
                data = informs.split('\n')
                groups = data[0].split(',')
                phiMin = float(groups[0].split(':')[1])
                phiMax = float(groups[1].split(':')[1])
                ReInitPhi = eval(groups[2].split(':')[1])
                shape = groups[3].split(':')[1]
                break
            else:
                cnt += 1
                continue
        return phiMin, phiMax, ReInitPhi, shape


def RetrievePhi(phiSave, phiMinValue, phiMaxValue):
    phiSaveMinValue = 0
    phiSaveMaxValue = 255
    phiMin = phiMinValue * np.ones(phiSave.shape)
    phiMax = phiMaxValue * np.ones(phiSave.shape)
    phiSaveMin = phiSaveMinValue * np.ones(phiSave.shape)
    phiSaveMax = phiSaveMaxValue * np.ones(phiSave.shape)
    phiRetrieve = ((phiSave * (phiMax - phiMin)) / (phiSaveMaxValue - phiSaveMinValue)) + phiMin
    return phiRetrieve


def process():
    startTime = time.time()
    firstFrame = generalparams.first_frame
    lastFrame = generalparams.last_frame
    currentFrame = generalparams.current_frame
    frameDir = generalparams.frame_dir
    outDir = generalparams.output_dir
    cropFrameLimit = generalparams.crop_frame_limit
    framePath = frameDir + "\\Blur"
    ReInitPhi = generalparams.reinit_phi
    FirstTimeSaveText = False
    """Check this is the first frame of the dataset?
        - Yes: multilook and initial phi with CACFAR in levelset process
        - No: Read the previous phi which used as the initial phi of the current frame"""
    if (currentFrame == firstFrame):
        multilook = True
        FirstTimeSaveText = True
        phiInit = None
        shapePrev = None
    else:
        multilook = False
        print("Import the initial phi(#%d) of the current frame#%d" %(currentFrame - 1, currentFrame))
        phiInit = cv2.imread(outDir + "\\phi_" + str(currentFrame - 1) + ".png", 0)
        """read phi min, max"""
        phiMinValue, phiMaxValue, ReInitPhi, shapePrev = readText(outDir, currentFrame - firstFrame)
        print("phiInit read from .txt (min, max): (%f, %f)" % (phiMinValue, phiMaxValue))
        phiInit = RetrievePhi(phiInit, phiMinValue, phiMaxValue)
        phiInit = phiInit.astype(np.float32)
        print("phiInit retrieve from phi.png (min, max): (%f, %f)" %(phiInit.min(), phiInit.max()))

    for i in range(currentFrame, lastFrame + 1):
        """-------1. Multilook Process-------"""
        if (multilook):
            frame = multilookProcess(framePath, cropFrameLimit, i)
            # plt.title("multilook result")
            # plt.imshow(frame, 'gray')
            # plt.show()
            multilook = False
        else:
            frame = cv2.imread(framePath + str(i) + ".png", 0)
            frame = frame[cropFrameLimit[0]:cropFrameLimit[1], cropFrameLimit[2]: cropFrameLimit[3]]
        """-------2. Pre-processing Process-------"""
        resPreprocess = preProcess(frame, cropFrameLimit, i)
        resPreprocess = resPreprocess.astype(dtype=np.uint8)
        # plt.title("pre-processing result")
        # plt.imshow(resPreprocess, 'gray')
        # plt.show()
        """-------3. Level-Set Process-------"""
        phi, shape = levelsetProcess(resPreprocess,
                                     phiInit,
                                     ReInitPhi,
                                     shapePrev,
                                     i)
        shapePrev = shape
        fig = plt.figure()
        plt.title("levelset result" + str(i))
        plt.imshow(resPreprocess, cmap='gray')
        CS = plt.contour(phi, 0, colors='r', linewidths=2)
        plt.draw()
        # plt.show()
        plt.savefig(outDir + "\\result_" + str(i) + ".png")
        """-------4. Post-processing Process-------"""
        phiDirection, ReInitPhi = postProcess(phi, shape, i)
        # phiSave = (0 * (phiInit < 0)) + (255 * (phiInit >= 0))
        # cv2.imwrite(outDir + "\\phi_binary_" + str(i) + ".png", phiSave)
        """save phi details"""
        FirstTimeSaveText = saveText(FirstTimeSaveText, phiDirection, ReInitPhi, shape, outDir, i)
        phiMinValue, phiMaxValue = phiDirection.min(), phiDirection.max()
        phiSaveMinValue = 0
        phiSaveMaxValue = 255
        phiMin = phiMinValue * np.ones(phiDirection.shape)
        phiMax = phiMaxValue * np.ones(phiDirection.shape)
        phiSave = ((phiDirection - phiMin) * (phiSaveMaxValue - phiSaveMinValue))/(phiMax - phiMin)
        phiSave = np.round(phiSave).astype(np.uint8)
        cv2.imwrite(outDir + "\\phi_" + str(i) + ".png", phiSave)
        phiInit = phiDirection
        # a = cv2.imread(outDir + "\\phi_" + str(i) + ".png", 0)


if __name__ == "__main__":
    process()
