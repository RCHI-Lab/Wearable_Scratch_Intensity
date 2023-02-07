# A Multimodal Sensing Ring for Quantification of Scratch Intensity

This code accompanies the submission:  
A Multimodal Sensing Ring for Quantification of Scratch Intensity, 
Akhil Padmanabha, Sonal Choudhary, Carmel Majidi, and Zackory Erickson

All data can be downloaded from [here](https://drive.google.com/drive/folders/1fyGAndOEDU-AejCiPgw0x9vvYic74ehT?usp=sharing). The data folder should be placed locally under the "final_code" directory. 

The "final_code" directory consists of 6 scripts for training and testing all 6 models using LOSO-CV. It also contains the final intensity regression model, normalizer, and script  "intensity_model_training.py" for training this model using all 20 participants' data from the first human study. It also has various functions in the "processing_functions.py" script. "save_data.py" uses the processing functions and data to save the datasets into pickle files (located under "data" directory)

The "final_paper" directory consists of 4 jupyter notebook files used to generate results and create figures for the paper. 

The "teensy_code" folder has the code for the teensy to sample the contact microphone and accelerometer. 

The "study_scripts" directory consists of the python data collection scripts for both study 1 and study 2. 
