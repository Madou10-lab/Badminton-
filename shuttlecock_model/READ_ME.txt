TrackNet3.py : deep network architecture

tracknet_weights : fichier du modèle entrainé sur shuttlecock_dataset

Ball_Tracker : experimentation notebook to generate csv ball coordinates of a passed input video

TrackNet : folder to prepare data / train the network :
	- dataset : Shuttlecock_dataset
	* Prepare training data : apply python3 gen_data_rally.py
	* Step 1 : Generate the frame of video
	python3 Frame_Generator.py <videoPath> <outputFolder>
	<videoPath> is the video path you want to train, and <outputFolder> is the output directory you want to store the frames under.
	* Step 2 : Preprocess the labeling csv file
		labeling software details : 	https://hackmd.io/CQmL6OKKSGKY9xUvU8n0iQ	 
	* Step 3 : Generate training data
	python3 gen_data.py --batch=<batchSize> --label=<csvFile> --frameDir=<frameDirectory> --dataDir=<npyDataDirectory>
	<csvFile> is the .csv file after apply Rearrange_Label.py at second step, 	<frameDirectory> is the frame folder of the video at first 	step, and 	<npyDataDirectory> is the output directory you want to store the training 	data under.
	* Step 4 : Start training TrackNetV2 python3 train_TrackNet.py --	save_weights=<weightPath> --dataDir=<npyDataDirectory> --		epochs=<trainingEpochs> --tol=<toleranceValue> <weightPath> is 	TrackNetV2 weight after this training, <npyDataDirectory> is the 		directory of the .npy training data at third step, and <toleranceValue> 	means tolerance value of true positive.
	* Step 5 : Retrain TrackNetV2 model If you want to retrain the model, 		please add load_weights argument. python3 		train_TrackNet.py --	load_weights=<previousWeightPath> --	save_weights=<newWeightPath> --dataDir=<npyDataDirectory> --			epochs=<trainingEpochs> --tol=<toleranceValue><previousWeightPath> is the model weights you had trained before.
	* Step 6 : 5. Provide the performance information
python3 accuracy.py --load_weights=<weightPath> --dataDir=<npyDataDirectory> --tol=<toleranceValue>
accuracy.py provide following version:
Number of true positive
Number of true negative
Number of false positive
Number of false negative
Accuracy
Precision
Recall