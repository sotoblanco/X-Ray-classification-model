
# X-ray image classification (Pneumonia detection in pediatric patients) with TensorFlow

## Problem description

Pneumonia is an infection that inflames the air sacs in one or both lungs, causing cough with phlegm or pus, fever, chills, and difficulty breathing. A variety of organisms, including bacteria, viruses and fungi, can cause pneumonia. 

Key facts provided by the WHO

-   Pneumonia accounts for 14% of all deaths of children under 5 years old, killing 740 180 children in 2019.
-   Pneumonia can be caused by viruses, bacteria or fungi.
-   Pneumonia can be prevented by immunization, adequate nutrition, and by addressing environmental factors.
-   Pneumonia caused by bacteria can be treated with antibiotics, but only one third of children with pneumonia receive the antibiotics they need.

Being able to accurately detect pneumonia in pediatric patients is a live and death procedure, in which being able to act fast can increase the survival chances.

In this project, we evaluate pneumonia in pediatric patients by using deep-learning techniques with **TensorFlow** that allows to create a classification model.

## Data

The data were obtained from Kaggle datasets under the name: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia). The weight of the folder is 2GB which cannot be uploaded to GitHub, but it can be downloaded using the Download button in the top right corner, or by using the step-by-step guide provided here to download the data using kaggle keys provided in this link. 

The following is the same description provided in the kaggle dataset about details of the data

> The data contains three folders (Train, test, val) containing subfolders for each image category (Pneumonia/Normal). There are 5,863 X-Ray images (JPEG) and 2 categories (Pneumonia/Normal). 

> Chest X-ray images (anterior-posterior) were selected from retrospective cohorts of pediatric patients of one to five years old from Guangzhou Women and Children’s Medical Center, Guangzhou. All chest X-ray imaging was performed as part of patients’ routine clinical care.

> For the analysis of chest x-ray images, all chest radiographs were initially screened for quality control by removing all low quality or unreadable scans. The diagnoses for the images were then graded by two expert physicians before being cleared for training the AI system. In order to account for any grading errors, the evaluation set was also checked by a third expert.

## Structure of the repository

``Dockerfile``: For deployment of the model in AWS as lambda function.

``Notebook.ipynb``: Notebook for exploratory data analysis, creating and exporting the model. 

``Pipfile`` and ``Pipfile.loc``: contains the dependencies to run the project.

``pneumonia-class.tflite``: Model with TensorFlow lite

``process_data.py``: Python script to process an url with the image and return a prediction

``test.py``: Python script to test the prediction service using AWS. 

## How to run

- Clone the repo

- Download the data from kaggle

- Install the dependencies
```
pipenv install
```
- Activate the virtual enviroment

```
pipenv shell
```

### Building the prediction model and service

Run the ``train.py`` file to obtain the best model for the training parameters as a ``.h5`` file and convert to tflite file.

#####add the kaggle notebook
> To make easier for you to run the training file you can go to this kaggle notebook that replicates the ``train.py`` file.

Run the docker file:

First build the model:

```
docker build -t pneumonia-model .
```
Run the docker image
```
docker run -it --rm -p 8080:8080 pneumonia-model:latest
```
Run the prediction service: Open a new command line (make sure you are running the docker file)
```
python test.py
```
The ``test.py`` already have an x-ray image link to return a prediction. 

> you can change the link to make a different prediction (some times do not work to take the link directly, you can just take a screenshot and upload to github

## Deployment

### Cloud deployment

AWS

**pre-requisets**  needs to have AWS CLI installed which is command line to interact with AWS ( I have a windows and working with WSL, so I download the cli using the linux command)

### Elastic Container Registry:

Place to store your container

Create repo View push command

Go to security credentials and find the access key to configure your AWS

run in your command line:  `aws configure`  and type your credentials from the above step

run:

Create the repo to store the image
```
aws ecr create-repository --repository-name pneumonia-class-images
```
Obtain the URI of the 
```
xxxxxx2.dkr.ecr.us-west-2.amazonaws.com/pneumonia-class-images
```
Set at the command line
```
$(aws ecr get-login --no-include)

ACCOUNT=xxxxxxx

REGION=us-west-2

REGISTRY=pneumonia-class-images

PREFIX=${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${REGISTRY}

TAG=pneumonia-class-model-v1-001

REMOTE_URI=${PREFIX}:${TAG}
```

Push the docker image to AWS
```
docker tag penumonia-model:latest ${REMOTE_URI}
docker push ${REMOTE_URI}
```

Create the lambda function

![image](https://user-images.githubusercontent.com/46135649/207652437-dfd995f8-6135-4229-b6a2-38183d273afa.png)

Browse the image

![image](https://user-images.githubusercontent.com/46135649/207652976-6470c49a-27f7-409d-80d5-402dbf83f298.png)

For deep learning task we need to increase the time of the response and the memory allocated to perform the function. 

We need to go configuration -> General configuration and change the timeout to 30 seconds and the memory to 1024

#### Create the method and post

- Use API Gateaway
- Select the POST method
- Integration type: lambda
- Select the lambda function

#### Deploy the endpoint

Go to actions and click on deploy

![image](https://user-images.githubusercontent.com/46135649/207659795-fddbf3a3-1dc3-4ca8-9680-02fa8b5a3574.png)


![image](https://user-images.githubusercontent.com/46135649/207660014-9baef1b4-fdb6-4637-a044-0fad8a86e8d3.png)

 Now we just need to obtain the URL and add predict at the end:
 ![image](https://user-images.githubusercontent.com/46135649/207660282-f9c17a53-aa2b-4c04-8c17-74efcb1b88ba.png)

## Demostration


