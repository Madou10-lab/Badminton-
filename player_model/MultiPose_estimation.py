import numpy as np
import cv2
from numba import jit



class MultiPoseEstimation:

    """
    Detecting and tracking player's position and skeleton

    """
    def __init__(self,nb,type,movenet):

        self.edges = {
            (0, 1): 'm',
            (0, 2): 'c',
            (1, 3): 'm',
            (2, 4): 'c',
            (0, 5): 'm',
            (0, 6): 'c',
            (5, 7): 'm',
            (7, 9): 'm',
            (6, 8): 'c',
            (8, 10): 'c',
            (5, 6): 'y',
            (5, 11): 'm',
            (6, 12): 'c',
            (11, 12): 'y',
            (11, 13): 'm',
            (13, 15): 'm',
            (12, 14): 'c',
            (14, 16): 'c'
                    }
        self.movenet= movenet
        self.confidence_threshold = 0.1
        #number of players must between 2-4
        self.number_players = nb
        #type of tracking : skeleton / bounding_box
        if(type not in ['skeleton','bounding_box']):
            raise Exception('please enter a valid type !!')
        self.type = type
        self.skeletons =[] 
        self.positions =[]
        self.center = 0 

    @jit 
    def draw_keypoints(self,frame, keypoints):
        green = False
        y, x, c = frame.shape
        shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
        for kp in shaped:
            ky, kx, kp_conf = kp
            if kp_conf > self.confidence_threshold:
                cv2.circle(frame, (int(kx), int(ky)), 3, (0,255,0), -1)
            else:
                break

        self.skeletons.append(shaped)
        if (kp_conf > self.confidence_threshold):
            green = True
            return green

    @jit 
    def draw_connections(self,frame, keypoints):
    
        y, x, c = frame.shape
        shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
        for edge, color in self.edges.items():
            p1, p2 = edge
            y1, x1, c1 = shaped[p1]
            y2, x2, c2 = shaped[p2]
        
            if (c1 > self.confidence_threshold) & (c2 > self.confidence_threshold):      
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 3)

    @jit 
    def find_player(self,center,centers,players):

        A=np.array(list(centers.values()))
        true_centers={}
        true_players={}
        for i in range(self.number_players):
            distances = np.linalg.norm(A-np.array(center), axis=1)
            key = list(centers.keys())[list(centers.values()).index(tuple(A[np.argmin(distances),:]))]
            true_centers[key]=(int(A[np.argmin(distances),0]),int(A[np.argmin(distances),1]))
            true_players[key]=players[key]
            A = np.delete(A,np.argmin(distances),0)
            players.pop(key)

        return true_centers,true_players

    @jit 
    def detect_player(self,input_img,frame,court):

        self.center = court[1] 
        results = self.movenet(input_img)

        centers={}
        players={}

        for i in range(6):

            ymin,xmin,ymax,xmax = np.squeeze(results['output_0'][:,:,51:])[i][:4]

            start_point=tuple(np.multiply(np.array([xmin,ymin]), [frame.shape[1],frame.shape[0]]).astype(int))
            end_point=tuple(np.multiply(np.array([xmax,ymax]), [frame.shape[1],frame.shape[0]]).astype(int))

            #position circle on zones to be checked after if here is player or not
            center = (int((start_point[0]+end_point[0])/2), int(end_point[1])) #center of player (x,y)
            centers[i]=center
            players[i]=(start_point,end_point)
        

        if court is not None:
        
            #range is same passed to find_player
            centers,players = self.find_player(court[1],centers,players)
            
            for key,value in centers.items():

                if cv2.pointPolygonTest(court[0],value,False) == 1.0: #1 inside contour / 0 on the edge


                    if(self.type=='bounding_box'):
                        #bounding box
                        cv2.rectangle(frame,players[key][0],players[key][1],( 255 , 0 , 0 ),3)
                        cv2.putText(frame,"Player", (players[key][0][0],players[key][0][1]), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (0, 255, 0))

                    if(self.type=='skeleton'):
                        #Render keypoints
                        person = results['output_0'].numpy()[:,:,:51].reshape((6,17,3))[key]
                        self.draw_connections(frame, person)
                        green = self.draw_keypoints(frame, person)
                        self.positions.append(value)
                        if(green):
                            cv2.circle(frame,value,8,( 0 , 255 , 0 ),-1)
                            cv2.putText(frame,"Player", (players[key][0][0],players[key][0][1]), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (0 , 255, 0))