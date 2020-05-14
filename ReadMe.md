# Basic Entity Recognition Engine 
This is a solution for VectorAI's coding challenge. The task is described [Here](https://github.com/LalitPradhan/VectorAI-Entity-Recognition-Engine-Challenge/blob/master/Vector%20Machine%20Learning%20Engineer%20Project%202.pdf)
This repository recieves a string and categorizes it into 5 classes {company_name, address, location, product, serial_number} and then clusters the individual strings in their respective categories based on cosine similarity matching. 

## Requirements
Code runs on Python3. 
- **To install dependencies**: `sudo pip3 install -r requirements.txt`
- **Download pretrained spacy mode**: `python -m spacy download en_core_web_lg`

## Description of Approach
The first task is to classify the images. To this end basic Rule based matching approaches were used. A pretrained Model as mentioned above was used for vectors. Each category has basic naive rules. For example to check address check for 'CARDINAL' in spacy's ent properties of a doc.
For the second task a naive spacy's similarity function is used and default threshold set to 0.7 for clustering together.

## Code Description
- The following arguments can be passed:
 	- a) demo_flag : set this to true to run a demo on pre written strings
  - a) similarity_threshold : set threshold between 0.0 to 1.0 for similarity matching. currently set to 0.7.
  - a) database_file_path : path to pickle file for historical data. default is output.pkl in current directory.
  - a) use_database_flag : set this to true to use historical data processed so far with the correct database_file_path
  - a) overwrite_flag : set this to true to update the historical database with the new entries 
  
- **To run demo**: `python3 -W ignore code.py --demo_flag True --overwrite_flag True`
- **To enter string manually**: `python3 -W ignore code.py --use_database_flag True --overwrite_flag True`

- The output to demo looks like the following:
![alt text](https://github.com/LalitPradhan/VectorAI-Entity-Recognition-Engine-Challenge/blob/master/demo_output.png)

-The output to manually entered strings looks like the following:
![alt text](https://github.com/LalitPradhan/VectorAI-Entity-Recognition-Engine-Challenge/blob/master/ManualEntry.png)

## Scopes for Improvement
- The entities can better classified by training from a data corpus
- The similarity distance can be improved for clustering by using tf-idf or custom distance for individual categories.
- This repo is to understand the approch for completing the task. The basic approach in nutshell described is to make a classifier and then cluster for individual categories. 
