# Music Genre Classifier

# Overview: 
This program analyzes the numerical features of 23,000 Spotify songs across **six musical genres**: EDM, rock, latin, pop, r&b, and rap. The data is observed over a time period of seven decades, 1950 through 2020. The following **eight** features are plotted: valence, acousticness, instrumentalness, tempo, danceability, speechiness, loudness, and energy. 

It leverages numpy and pandas for data wrangling, Matplotlib for visualization output, and Scikit-Learn to build a KNN prediction model based on feature proximity.

# Data Analysis & Trends: 
To ensure readability, the data visualizations are divided into two distinct figures, displaying **four** features per figure. Key trends identified over the observed time period include:

* A significant decline in overall **acousticness**
* Consistently high levels of **speechiness** within the Rap genre
* Notable spikes in **instrumentalness** within EDM
* A general downward trend observed amongst 'happiness' feature (**valence**)

# Interactive User Prediction: 
This application allows users to test the K-NN model by inputting the numerical features of any preferred song. To ensure a seamless user experience, the program utilizes flexible fallback logic: if a user leaves a feature blank, the model automatically substitutes the dataset's average for that features. 

**Note:** 
* I recommend using the website tunebat.com at your own discretion to look up features for a desired song. If unsure of a value, hit enter to use features average.
* 'Loudness' is measured in decibels (dB). This value is usually negative.
* The values entered for features **MUST be recorded in decimals less than 1**, with the exception of loudness, as tunebat.com uses **integers** (e.g., if danceability is 60, input this as 0.6)

# Imporant Note:
To properly generate and view Matplotlib data visualizations, the user must quit the interactive prediction prompt. The graphs will render immediately after the program loop ends.
