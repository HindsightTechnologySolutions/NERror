# Hindsight NER Error Suite

## NOTE:
* The main functionality of this project along with the project history have been removed and replaced with a bare-bones skeleton to protect proprietary code. I can however provide a presentation of the codes use if needed. 

## Team Manager:
* Onur Kara

## Contributors:
* Kaelan Nettleship - Lead
* Sebastian Gonzales 
* Alexis Soulias
* Sophia Pentakalos
* Ivy Peng

## Purpose:
Suite designed to identify errors within the output of an NER system and act as a tool in finetuning said NER model. Unlike other analysis tools, the Hindsight NER Error Suite provides highly specific error classification and powerful visualizations.

### Inputs:
A json consisting of:
* article url
* article title
* raw text of scraped article
* entites found by NER model
* ground truth (annotated) entities

## The Code:

identitifer.py - Contains the Error_Identifier class that is instantiated for each article and that finds and classifies all errors in the entities found by the NER model for the given article 

data_cleaning.py - Used to clean an articles text and convert it to a spaCy doc object 

connect_bucket.py - Contains the Connect_bucket class which allows access to an s3 bucket to save the performance statistics for each article processed and then access again for later use

run.py - Contains all of the streamlit code neceessary for generating the UI and runs each article through the analysis processing pipeline (data_cleaning -> identifier -> run)






