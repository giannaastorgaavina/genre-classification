# genre-classification

# Overview: 
This program uses numpy and pandas to filter the numerical features of 23,000 songs on Spotify; matplotlib to help visualize genres and their numerical feature averages, showcasing patterns within each genre/subgenre; and Scikit-Learn for the KNN model to predict genre labels by analyzing the proximity of numerical features.

# Data Analysis: 
The visualizations are split into two figures for ease of readability. Throughout these visualizations, there are many notable trends with the most significant being: decline in acousticness over the observed time period; rap genre has maintained a high level of 'speechiness'; edm reflecting spikes in instrumentalness; general downward trend observed amongst 'happiness' attribute (valence).

# User Input Feature: 
the section containing code for user input is tests the genre prediction model by asking the user to enter the numerical feautures for a song, with some flexibility. The average value of a feature is used if the user is unsure. A website is recommend just for ease of accessing these features quickly for any song. 
