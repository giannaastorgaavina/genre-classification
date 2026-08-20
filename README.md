# Music Genre Classifier

# Overview: 
This program analyzes the numerical attributes of 23,000 Spotify songs to across **six musical genres**: EDM, rock, latin, pop, r&b, and rap. The data is observed over a time period of seven decades, 1950 through 2020. The following **eight** attributes are plotted: valence, acoutsicness, instrumentalness, tempo, danceability, speechiness, loudnesss, and energy. 

It leverages numpy and pandas for data wrangling, Matplotlib for visualization output, and Scikit-Learn to build a KNN prediction model based on feature proximity.

# Data Analysis & Trends: 
To ensure readability, the data visualizations are divided into two distinct figures, displaying **four** features per figure. Key trends identified over the observed time period include:

* A significant decline in overall **acousticness**
* Consistently high levels of **speechiness** within the Rap genre
* Notable spikes in **instrumentalness** within EDM
* A general downward trend observed amongst 'happiness' attribute (**valence**)

# Interactive User Prediction: 
This application allows users to test the K-NN model by inputting the numerical features of any perferred song. To ensure a seamless user experience, the program utilizes flexible fallback logic: if a user leaves a feature blank, the model automatically substitutes the dataset's average for that attribute. 

**Note:** 
* I recommend using the the website tunebat.com at your own discretion to look up attributes for a desired song. If unsure of a value, hit enter to use attribute average.
* 'Loudness' is measured in decibels (dB). This value is usually negative.
* The values entered for attributes **MUST be recorded in decimals less than 1**, with the exception of loudness, as tunebat.com uses **integers** (e.g., if danceability is 60, input this as 0.6)

# Imporant Note:
To properly generate and view Matplotlib data visualizations, the user must quit the interactive prediction prompt. The graphs will render immediately after the program loop ends.
