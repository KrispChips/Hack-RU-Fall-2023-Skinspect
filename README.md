# Hack-RU-Fall-2023-Skinspect
We created a Skin Tumor Classification model that can detect whether your tumor is malignant or benign! We did this in the HackRU 2023 Fall Hackathon and ended up getting Runer Up in the Health Category. There are issues in the code of course and not everything is perfect, but we wish to improve upon this idea in the future and continue this project!

The dataset was made by CLAUDIO FANCONI on Kaggle and was named Skin Cancer: Malignant vs. Benign. We would like to give credit to him and access to his dataset can be found [here](https://www.kaggle.com/datasets/fanconic/skin-cancer-malignant-vs-benign)! The dataset consists of skin moles that are either benign or malignant and are already split into test and train subset folders.

Our web-app implements TensorFlow and Keras for our Convolution Neural Network (CNN) model and trains these images using gens and random split! These images are first split into generators with hyper-parameters and split-rate and split into train, test and validation gens. After, the model is created using Sequential with Conv2D and MaxPooling. At the end, the fully connected layer is a Dense Softmax layer with the probability of where the image belongs in the class. Of course, the model was quickly trained with basic hyper-parameters so we wish to further our accuracy using transfer learning with advanced model such as vgg16 or resent50.

Our front-end for the web app was made with python API Streamlit! Our site has a side-bar which has basic information on what a benign and malignant tumor is and can help the reader understand the condition better. In the center, there is a file input where the user can either drag-and-drop their file or upload the file from their computer. Once the file is taken, our model will classify whether the skin mole is either benign or malignant and report underneath with more information on further treatment and action to take. It is also reported on the sidebar with accuracy percentage of our model. In the future, we wish to improve on the UI of our website as it is simple but it gets the job done!

![website][Screenshot Hack RU 2023 Fall SkinSpect Site 2023-10-10 134914.png]


We would like to thank Hack RU for inspiring and pushing us to do this project and we ended up getting Runner Up in the Health Track of the competition! We would love to further this project and improve upon the simple idea we currently have. 
