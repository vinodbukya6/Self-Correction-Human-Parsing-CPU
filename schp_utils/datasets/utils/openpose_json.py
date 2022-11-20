import cv2
import time
import numpy as np
from random import randint
import argparse
import json
import pandas as pd 

#image1 = cv2.imread("000010_0.jpg")
protoFile = "checkpoints/openpose_pose_coco.prototxt"
weightsFile = "checkpoints/pose_iter_440000.caffemodel"
nPoints = 18

# COCO Output Format
keypointsMapping = ['Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho', 'L-Elb', 'L-Wr', 'R-Hip', 'R-Knee', 'R-Ank', 'L-Hip', 'L-Knee', 'L-Ank', 'R-Eye', 'L-Eye', 'R-Ear', 'L-Ear']

def getKeypoints(probMap, threshold=0.1):

    mapSmooth = cv2.GaussianBlur(probMap,(3,3),0,0)

    mapMask = np.uint8(mapSmooth>threshold)
    keypoints = []

    #find the blobs
    contours, _ = cv2.findContours(mapMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #for each blob find the maxima
    for cnt in contours:
        blobMask = np.zeros(mapMask.shape)
        blobMask = cv2.fillConvexPoly(blobMask, cnt, 1)
        maskedProbMap = mapSmooth * blobMask
        _, maxVal, _, maxLoc = cv2.minMaxLoc(maskedProbMap)
        keypoints.append(maxLoc + (probMap[maxLoc[1], maxLoc[0]],))

    return keypoints
    
# generate target image pose key points    
def generate_pose_keypoints(img_file):
    '''
    Generates pose keypoints
    Input: Person Image
    Output: Writes json file with keypoints(shape: 18*3 = 54)
    '''
    
    image1 = cv2.imread("dataroot/test/"+ img_file)
    frameWidth = image1.shape[1]
    frameHeight = image1.shape[0]
    
    t = time.time()
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
    #if args.device == "cpu":
    net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
    print("Using CPU device")
    #elif args.device == "gpu":
    #     net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    #     net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    #     print("Using GPU device")

    # Fix the input Height and get the width according to the Aspect Ratio
    inHeight = 368
    inWidth = int((inHeight/frameHeight)*frameWidth)

    inpBlob = cv2.dnn.blobFromImage(image1, 1.0 / 255, (inWidth, inHeight),
                                    (0, 0, 0), swapRB=False, crop=False)

    net.setInput(inpBlob)
    output = net.forward()
    print("Time Taken in forward pass = {}".format(time.time() - t))

    detected_keypoints = []
    keypoints_list = np.zeros((0,3))
    keypoint_id = 0
    threshold = 0.1

    for part in range(nPoints):
        probMap = output[0,part,:,:]
        probMap = cv2.resize(probMap, (image1.shape[1], image1.shape[0]))
        keypoints = getKeypoints(probMap, threshold)
        print("Keypoints - {} : {}".format(keypointsMapping[part], keypoints))
        keypoints_with_id = []
        for i in range(len(keypoints)):
            keypoints_with_id.append(keypoints[i] + (keypoint_id,))
            keypoints_list = np.vstack([keypoints_list, keypoints[i]])
            keypoint_id += 1

        detected_keypoints.append(keypoints_with_id)


    frameClone = image1.copy()
    pose_keypoints = []
    for i in range(nPoints):
        if detected_keypoints[i] ==[]:
            pose_keypoints.append(0)
            pose_keypoints.append(0)
            pose_keypoints.append(0)
            
        for j in range(len(detected_keypoints[i])):
            pose_keypoints.append(detected_keypoints[i][j][0])
            pose_keypoints.append(detected_keypoints[i][j][1])
            pose_keypoints.append(detected_keypoints[i][j][2].astype(float))

    json_data = {"version": 1.0, "people": [{
        "face_keypoints": [],
        "pose_keypoints":pose_keypoints,
        "hand_right_keypoints": [],
        "hand_left_keypoints": []
        }]}

    anno = list(json_data['people'][0]['pose_keypoints'])
    x = np.array(anno[1::3])
    y = np.array(anno[::3])
    
    x = np.array(x)
    y = np.array(y)
    x[x==0] = -1
    y[y==0] = -1
    x = x[:18]
    y = y[:18]
    x1 = list(map(int, x))
    y1 = list(map(int, y))

    annotation_file = pd.read_csv("dataroot/fasion-annotation-test.csv", sep=':')
    annotation_file.loc[len(annotation_file.index)] = [img_file, x1, y1] 
    #data_anno = [{'name': img_file , 'keypoints_x': x1, 'keypoints_y': y1}]
    # saving the dataframe
    annotation_file.to_csv('dataroot/fasion-annotation-test.csv',index=False, sep=':')
    
    # write json file
    #json_file = img_file.split('.jpg')[0] + "_keypoints.json" 
    #with open("dataroot/pose_json/"+ json_file, 'w') as outfile:
        #json.dump(json_data, outfile)
    return "written keypoints json file"




    
    
    
    
    
    
    
 


        
