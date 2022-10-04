

import os

import cv2

import mediapipe as mp
import dollarpy
from dollarpy import Recognizer, Template, Point

import numpy as np






# returns a list of the landmarks in a video,
# takes target video path and a list of target joints to extract
def GetLandMarksFromVideo(videoPath,target_lms):
    mp_drawing = mp.solutions.drawing_utils

    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(videoPath)

    landmarks = []
    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
        while cap.isOpened():

            ret, frame = cap.read()
            if not ret:
                break
            # Recolor image to RGZB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)
            if results.pose_landmarks == None:
                continue
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                      )
            lm = results.pose_landmarks.landmark
            # landmarks.append(Point(lm[11].x, lm[11].y))
            # landmarks.append(Point(lm[12].x, lm[12].y))
            # landmarks.append(Point(lm[13].x, lm[13].y))
            # landmarks.append(Point(lm[14].x, lm[14].y))
            # landmarks.append(Point(lm[24].x, lm[24].y))
            # landmarks.append(Point(lm[23].x, lm[23].y))
            # landmarks.append(Point(lm[25].x, lm[25].y))
            # landmarks.append(Point(lm[26].x, lm[26].y))
            # landmarks.append(Point(lm[27].x, lm[27].y))
            # landmarks.append(Point(lm[28].x, lm[28].y))

            index = 0

            for target in target_lms:

                landmarks.append(Point(lm[target].x,lm[target].y))
                print (landmarks[-1])
            cv2.imshow('Mediapipe Feed', image)


    return landmarks


# returns a dollarpy template from video & selected joints
def CreateTemplateFromVideo(videoPath, label,target_lms):
    lms = GetLandMarksFromVideo(videoPath,target_lms)
    return Template(label, lms)


def getactivityduration(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)      # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count/fps
    seconds = int(duration%60)
    print(seconds)
    return seconds
    cap.release()

def GetFramesOfVideo(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    return frames

def GetLandMarksFromFrames(frames,target_lms):
    mp_drawing = mp.solutions.drawing_utils

    mp_pose = mp.solutions.pose
    font = cv2.FONT_HERSHEY_SIMPLEX

    # org

    # fontScale
    fontScale = 1

    # Blue color in BGR
    color = (0, 0, 0)

    # Line thickness of 2 px
    thickness = 2
    landmarks = []
    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
        for image in frames:

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            if results.pose_landmarks == None:
                continue

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                      )
            lm = results.pose_landmarks.landmark

            for target in target_lms:
                landmarks.append(Point(lm[target].x, lm[target].y))



    return landmarks




# C:\Users\GAMING\PycharmProjects\Football_Drill_Evaluation\src\actions\trims\high_knees

actions = ["high_knees" ,"jumping_jacks","lateral_left","lateral_right"]



actions = os.listdir("src/actions/trims")

target_joints = [24,25,26,27,28,29,30,31,32,12,11,14,13]
target_joints = range(1,33)
print (target_joints)



trims_directory="./src/actions/trims"
tests_directory="./src/actions/tests"

ROOT_DIR = os.path.abspath(os.curdir)

templates = []




for action in actions:

    action_path = os.path.join(trims_directory,action)
    print (action)
    for video in os.listdir(action_path):
        video_path = os.path.join(action_path,video)

        duration = getactivityduration(video_path)
        framesOfVideo = GetFramesOfVideo(video_path)
        print("frames  count :", str(len(framesOfVideo)))
        if (duration <=0):
            print (video_path)
            continue
        fps = int(len(framesOfVideo) / duration)
        print("FPS IS: " + str(fps))
        step = fps * 1
        start = 0
        while start < len(framesOfVideo) - step:
            print("ENTERED")
            end = start + step
            ct = 0
            templates.append(Template(action, GetLandMarksFromFrames(framesOfVideo[start:end], target_joints)))

            start += step




print ("testing for two feet juggle")



recognizer = Recognizer(templates)
print("TEMPLATES" +str (len(templates)))



# testing on an already seen vidoe (100% confidence)
# print (recognizer.recognize(GetLandMarksFromVideo(os.path.join(trims_directory,actions[0],"1.mp4")
#                                            ,target_joints)))
#


#
# print (recognizer.recognize
#        (GetLandMarksFromVideo(os.path.join(tests_directory,"two_feet_juggle","5.mp4"),target_joints)))
#
#
#
#
#
#

#
#
# print (recognizer.recognize
#        (GetLandMarksFromVideo(os.path.join(tests_directory,actions[3],"2.mp4"),target_joints)))
#
#



# print (recognizer.recognize
#        (GetLandMarksFromVideo(os.path.join(tests_directory,actions[0],"2.mp4"),target_joints)))
#
#
#
#

#
# print (recognizer.recognize
#        (GetLandMarksFromVideo(os.path.join(tests_directory,"two_feet_juggle","4.mp4"),target_joints)))
#


test_path = os.path.join(tests_directory,"sequence","1.mp4")


duration = getactivityduration(test_path)
framesOfVideo = GetFramesOfVideo(test_path)
print ("frames  count :",str (len(framesOfVideo)))
fps = int (len(framesOfVideo)/duration)
print ("FPS IS: " + str(fps))

step = fps * 1
start = 0


while start < len(framesOfVideo)-step:
    print ("ENTERED")
    end = start + step
    dict = {}
    for action in actions:
        dict[action] =0

    dict[None] =0
    for frame in framesOfVideo[start:end]:
        try:

            result = recognizer.recognize(GetLandMarksFromFrames([frame], target_joints))
            dict[result[0]] += 1

        except:
            print("Contibue")
            continue
    print (dict)
    start += step
#
#
# start =0
# end = start+step
# step = fps * 1
# shift =1
# while end < len(framesOfVideo):
#     window = framesOfVideo[start:end]
#     dict = {}
#     for action in actions:
#         dict[action] =0
#
#     dict[None] = 0
#
#     for frame in window:
#         result = recognizer.recognize(GetLandMarksFromFrames([frame], target_joints))
#         dict[result[0]]+=1
#
#     print (dict)
#     start+=shift
#     end+=shift







