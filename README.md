# Music Genre Classifier

# Overview: 
This program aanalyzes the numerical attributes of 23,000 Spotfiy songs to predict their genre. It leverages numpy and pandas for data wrangling, Matplotlib for visualization ouput, and Scikit-Learn to build a KNN prediction model based on feature proximity.

# Data Analysis & Trends: 
The visualizations are split into two figures for ease of readability. Throughout these visualizations, there are many notable trends with the most significant being:

* A significant decline in overall **acousticness**
* Consistently high levels of **speechiness** within the Rap genre
* Notable spikes in **instrumentalness** within EDM
* A general downward trend observed amongst 'happiness' attribute (**valence**)

# Interactive User Prediction: 
The section containing code for user input is tests the genre prediction model by asking the user to enter the numerical feautures for a song, with some flexibility. The average value of a feature is used if the user is unsure. A website is recommend just for ease of accessing these features quickly for any song. 

# Imporant Note:
To properly generate and view Matplotlib data visualizations, the user must quit the interactive prediction prompt. The graphs will render immediately after the program loop ends.
