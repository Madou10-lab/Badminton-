action_Data_augm.ipynb : augment the data before preprocessing

action_Data_Preprocess.ipynb : prepare the data from strokes_dataset

Action_rec_model.ipynb : train / validate the data preprocessed

action_rec_weights : fichier du modèle entrainé sur strokes_train_data

detect_actions.py : to detect actions on a rally video and outputs csv files of each player strokes 

clean_strokes.py : clean the log of action and classify each stroke based on shuttlecock trajectory and velocity of the ball

test.ipynb : calculate confusion matrix of stroke classification using true / predicted csv files, and accuracy of prediction

