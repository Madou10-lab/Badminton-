import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
import numpy as np
import pandas as pd 
import os
import queue
import csv
import enum
import tqdm
import cv2
from Court_detection import CourtDetector
from MultiPose_estimation import MultiPoseEstimation
from clean_strokes import *

class BodyPart(enum.Enum):
    
  """Enum representing human body keypoints detected by pose estimation models."""
  NOSE = 0
  LEFT_EYE = 1
  RIGHT_EYE = 2
  LEFT_EAR = 3
  RIGHT_EAR = 4
  LEFT_SHOULDER = 5
  RIGHT_SHOULDER = 6
  LEFT_ELBOW = 7
  RIGHT_ELBOW = 8
  LEFT_WRIST = 9
  RIGHT_WRIST = 10
  LEFT_HIP = 11
  RIGHT_HIP = 12
  LEFT_KNEE = 13
  RIGHT_KNEE = 14
  LEFT_ANKLE = 15
  RIGHT_ANKLE = 16

class ActionDetector():

    def __init__(self,weights_path):

        self.model = keras.models.load_model(weights_path)
        self.class_names =['BH_Serve', 'Drop-Clear-Drive-Smash', 'FH_serve', 'Lob-Net','No_Pose']
        #strokes recognition csv files
        self.strokesA = pd.DataFrame(columns=['Player_A'])
        self.strokesB = pd.DataFrame(columns=['Player_B'])

    def get_center_point(self,landmarks, left_bodypart, right_bodypart):
        """Calculates the center point of the two given landmarks."""
        left = tf.gather(landmarks, left_bodypart.value, axis=1)
        right = tf.gather(landmarks, right_bodypart.value, axis=1)
        center = left * 0.5 + right * 0.5
        return center

    def get_pose_size(self,landmarks, torso_size_multiplier=2.5):
        """Calculates pose size.
        It is the maximum of two values:
        * Torso size multiplied by `torso_size_multiplier`
        * Maximum distance from pose center to any pose landmark
        """
        # Hips center
        hips_center = self.get_center_point(landmarks, BodyPart.LEFT_HIP, 
                                 BodyPart.RIGHT_HIP)

        # Shoulders center
        shoulders_center = self.get_center_point(landmarks, BodyPart.LEFT_SHOULDER,
                                      BodyPart.RIGHT_SHOULDER)

        # Torso size as the minimum body size
        torso_size = tf.linalg.norm(shoulders_center - hips_center)
        # Pose center
        pose_center_new = self.get_center_point(landmarks, BodyPart.LEFT_HIP, 
                                     BodyPart.RIGHT_HIP)
        pose_center_new = tf.expand_dims(pose_center_new, axis=1)
        # Broadcast the pose center to the same size as the landmark vector to
        # perform substraction
        pose_center_new = tf.broadcast_to(pose_center_new,
                                    [tf.size(landmarks) // (17*2), 17, 2])

        # Dist to pose center
        d = tf.gather(landmarks - pose_center_new, 0, axis=0,
                name="dist_to_pose_center")
        # Max dist to pose center
        max_dist = tf.reduce_max(tf.linalg.norm(d, axis=0))

        # Normalize scale
        pose_size = tf.maximum(torso_size * torso_size_multiplier, max_dist)
        return pose_size

    def normalize_pose_landmarks(self,landmarks):
        """Normalizes the landmarks translation by moving the pose center to (0,0) and
        scaling it to a constant pose size.
        """
        # Move landmarks so that the pose center becomes (0,0)
        pose_center = self.get_center_point(landmarks, BodyPart.LEFT_HIP, 
                                 BodyPart.RIGHT_HIP)

        pose_center = tf.expand_dims(pose_center, axis=1)
        # Broadcast the pose center to the same size as the landmark vector to perform
        # substraction
        pose_center = tf.broadcast_to(pose_center, 
                                [tf.size(landmarks) // (17*2), 17, 2])
        landmarks = landmarks - pose_center

        # Scale the landmarks to a constant pose size
        pose_size = self.get_pose_size(landmarks)
        landmarks /= pose_size
        return landmarks

    def landmarks_to_embedding(self,landmarks_and_scores):
        """Converts the input landmarks into a pose embedding."""
        # Reshape the flat input into a matrix with shape=(17, 3)
        reshaped_inputs = keras.layers.Reshape((17, 3))(landmarks_and_scores)

        # Normalize landmarks 2D
        landmarks = self.normalize_pose_landmarks(reshaped_inputs[:, :, :2])
        # Flatten the normalized landmark coordinates into a vector
        embedding = keras.layers.Flatten()(landmarks)
        return embedding


    def get_person(persons,frame):

        scores = {}

        for i in range(len(persons)):

            y, x, c = frame.shape
            shaped = np.squeeze(np.multiply(persons[i], [y,x,1]))
            person_score = np.average(list(shaped[:,2]))
            scores[i]=person_score

        return max(scores, key=scores.get)

    def detect_stroke(self,multipose,frame):

        embeddingA =[]  
        embeddingB =[]
        
        playerA = min(multipose.positions,key=lambda tup: tup[1])
        playerB = max(multipose.positions,key=lambda tup: tup[1])
        skeletonA = multipose.skeletons[multipose.positions.index(playerA)]
        skeletonB = multipose.skeletons[multipose.positions.index(playerB)]

        embeddingA.append(tf.reshape(self.landmarks_to_embedding(tf.reshape(tf.convert_to_tensor(skeletonA), (1, 51))), (34)))
        embeddingA = tf.convert_to_tensor(embeddingA)
        predictionA = self.model.predict(embeddingA)

        embeddingB.append(tf.reshape(self.landmarks_to_embedding(tf.reshape(tf.convert_to_tensor(skeletonB), (1, 51))), (34)))
        embeddingB = tf.convert_to_tensor(embeddingB)
        predictionB = self.model.predict(embeddingB)

        #88 % : validation accuracy
    
        if(self.class_names[np.argmax(predictionA)] != 'No_Pose' and predictionA[0][np.argmax(predictionA)]*100 >= 88):
            cv2.putText(frame,"Action Detected", playerA , cv2.FONT_HERSHEY_TRIPLEX, 1.0, (0 , 0, 255))
            self.strokesA = self.strokesA.append({'Player_A':self.class_names[np.argmax(predictionA)]}, ignore_index=True)
        else:
            self.strokesA = self.strokesA.append({'Player_A':'___'}, ignore_index=True)

        if(self.class_names[np.argmax(predictionB)] != 'No_Pose' and predictionB[0][np.argmax(predictionB)]*100 >= 88):
            cv2.putText(frame,"Action Detected", playerB , cv2.FONT_HERSHEY_TRIPLEX, 1.0, (0 , 0, 255))
            self.strokesB = self.strokesB.append({'Player_B':self.class_names[np.argmax(predictionB)]}, ignore_index=True)
        else:
            self.strokesB = self.strokesB.append({'Player_B':'___'}, ignore_index=True)


    def save_actions(self,pathA,pathB,input_csv,actionA_path,actionB_path):

        self.strokesA.to_csv(pathA,index=False)
        self.strokesB.to_csv(pathB,index=False)

       # Clean Strokes of Top Player 
        preprocess(pathA,actionA_path,"Player_A",get_ballY(input_csv))
        testA = pd.read_csv(actionA_path)
        recognize(testA,'Lob-Net',get_ballY(input_csv),'Player_A',actionA_path)
        recognize(testA,'Drop-Clear-Drive-Smash',get_ballY(input_csv),'Player_A',actionA_path)

        # Clean Strokes of Bottom Player 
        preprocess(pathB,actionB_path,"Player_B",get_ballY(input_csv))
        testB = pd.read_csv(actionB_path)
        recognize(testB,'Lob-Net',get_ballY(input_csv),'Player_B',actionB_path)
        recognize(testB,'Drop-Clear-Drive-Smash',get_ballY(input_csv),'Player_B',actionB_path)

       #get stats
        get_stats(testA,testB) 


