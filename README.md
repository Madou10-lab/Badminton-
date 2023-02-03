# Badminton-
A computer vision project to analyse player's strategy in the game of badminton via detection and tracking algorithms

notes : 
!!! VIDEO MUST BE A RALLY not A MATCH

- WHEN HAVING FULL MATCH - SPLIT IT INTO RALLIES
- INPUT VIDEO MUST CONTAIN ONLY THE GAME PLAYED : AN EXAMPLE IS PASSED EXAMPLE.MP4

main.ipynb : final notebook to test all models on insep_dataset

	-> this notbeook outputs automatically the final statistics and results video with visual tracks of all parts along with strokes csv file of each player

ball_predicted folder : output folder to store shuttlecock trajectory coordinates
classifier_inputs : output folder for the final system outputs 

to get final results : just run all cels of main.ipynb  and add paths from all models folders : court_model / player_model / shuttlecock_model / strokes_classification_model

!!! DO NOT CHANGE OUTPUT FOLDER NAMES
