import cv2
import pandas as pd
import numpy as np
import csv
import math
import matplotlib.pyplot as plt

def get_ballY(input_csv_path):

    with open(input_csv_path) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        y = []
        list1 = []
        for row in readCSV:
            list1.append(row)
        for i in range(1 , len(list1)):
            y += [int(float(list1[i][3]))]

    return y

def clean(strokes,name,player):

    d = strokes[strokes[player] == name]
    i=0
    while (i < d.shape[0]-1):

        index1 = list(d.index)[i]
        index2 = list(d.index)[i+1]
        if(d.loc[index2]['Frame'] - d.loc[index1]['Frame'] <= 15):
            strokes=strokes.drop(index2)

        i+=1

    return strokes

def clean2(strokes,name,player,y):

    #filtering serve actions 
    strokes = strokes[strokes[player] != 'FH_serve']
    strokes = strokes[strokes[player] != 'BH_Serve']

    d = strokes[strokes[player] == name]
    i=0
    while (i < d.shape[0]):

        index = list(d.index)[i]

        if(name == 'Lob-Net'):
            ball= list(filter(lambda a: a !=0, y[d.loc[index]['Frame']+10:d.loc[index]['Frame']+20]))
            diff_avg = np.average(abs(np.diff(np.array(ball))))
            if(math.isnan(diff_avg)):
                strokes=strokes.drop(index)
        
        else:
            ball= list(filter(lambda a: a !=0, y[d.loc[index]['Frame']+5:d.loc[index]['Frame']+15]))
            diff_avg = np.average(abs(np.diff(np.array(ball))))
            if(math.isnan(diff_avg)):
                strokes=strokes.drop(index)

        i+=1

    return strokes


def recognize(strokes,name,y,player,result_path):

    for i, row in strokes.iterrows():
        if (row[player] == name):

            if(name == 'Lob-Net'):

                ball= list(filter(lambda a: a !=0, y[row['Frame']+10:row['Frame']+20]))
                diff_avg = np.average(abs(np.diff(np.array(ball))))
                if(diff_avg  < 10):
                    strokes.at[i, player]='Net'
                else:
                    strokes.at[i, player]='Lob'

            else:
                
                ball= list(filter(lambda a: a !=0, y[row['Frame']+5:row['Frame']+15]))
                diff_avg = np.average(abs(np.diff(np.array(ball))))
                if(diff_avg  < 10):
                    strokes.at[i, player]='Smash'
                elif(10 <= diff_avg  < 30):
                    strokes.at[i, player]='Drop'
                else:
                    strokes.at[i, player]='Clear'
    
    strokes.to_csv(result_path,index=False)



def preprocess(csv_path,result_path,player,y):

    data = pd.read_csv(csv_path)
    data = data.loc[data[player].shift() != data[player]]
    data.to_csv(result_path,index=True)

    test = pd.read_csv(result_path)
    test.rename( columns={'Unnamed: 0':'Frame'}, inplace=True )
    test = clean(test,'Lob-Net',player)
    test = clean(test,'Drop-Clear-Drive-Smash',player)
    test = clean2(test,'Lob-Net',player,y)
    test = clean2(test,'Drop-Clear-Drive-Smash',player,y)
    test.to_csv(result_path,index=False)


def get_stats(csvA,csvB):

    strokes = ['Lob','Net','Smash','Drop','Clear']
    resA = []
    resB = []
    OffA,DefA = 0,0
    OffB,DefB = 0,0

    actionsA = csvA[csvA['Player_A'] != '___']['Player_A'].tolist()
    actionsB = csvB[csvB['Player_B'] != '___']['Player_B'].tolist()
 
    for s in actionsA:

        if(s == 'Smash'):
            OffA+=2
        if(s in ['Lob','Clear']):
            DefA+=1
        if(s in ['Net','Drop']):
            OffA+=1

    for s in actionsB:

        if(s == 'Smash'):
            OffB+=2
        if(s in ['Lob','Clear']):
            DefB+=1
        if(s in ['Net','Drop']):
            OffB+=1
        
    for i in strokes:
        resA.append((actionsA.count(i)/len(actionsA))*100)
        resB.append((actionsB.count(i)/len(actionsB))*100)
        
    fig2, (axA, axB) = plt.subplots(1, 2)
    axA.pie(resA, labels=strokes,autopct='%1.1f%%',textprops=dict(color="w"),shadow=True, startangle=90)
    axA.set_title("Top Player")
    wedges, texts, autotexts = axB.pie(resB, labels=strokes,autopct='%1.1f%%',textprops=dict(color="w"),shadow=True, startangle=90)
    fig2.legend(wedges, strokes,
        title="Strokes",
        loc="center left",
        bbox_to_anchor=(1.0, 0.5))

    plt.setp(autotexts, size=8, weight="bold")
    axB.set_title("Bottom Player")
    fig1, (ax1, ax2) = plt.subplots(1, 2)
    ax1.pie([OffA/(DefA+OffA),DefA/(DefA+OffA)], labels=['Off','Def'],autopct='%1.1f%%',textprops=dict(color="w"),shadow=True, startangle=90)
    wedges, texts, autotexts = ax2.pie([OffB/(DefB+OffB),DefB/(DefB+OffB)], labels=['Off','Def'],autopct='%1.1f%%',textprops=dict(color="w"),shadow=True, startangle=90)
    fig1.legend(wedges, ['Offensive','Defensive'],
        title="Style of play",
        loc = 'center left', bbox_to_anchor = (1.0, 0.5))
    fig1.savefig("/home/JouiniAhmad/Desktop/douma/Badminton_Project/classifier_inputs/stats2.png",bbox_inches="tight")
    fig2.savefig("/home/JouiniAhmad/Desktop/douma/Badminton_Project/classifier_inputs/stats1.png",bbox_inches="tight")
    
    img1 = cv2.imread("/home/JouiniAhmad/Desktop/douma/Badminton_Project/classifier_inputs/stats1.png")
    img2 = cv2.imread("/home/JouiniAhmad/Desktop/douma/Badminton_Project/classifier_inputs/stats2.png")
    img2 = cv2.resize(img2,(img1.shape[1], img1.shape[0]))
    im_v = cv2.vconcat([img1, img2])
    cv2.imwrite("/home/JouiniAhmad/Desktop/douma/Badminton_Project/classifier_inputs/stats.png",im_v)