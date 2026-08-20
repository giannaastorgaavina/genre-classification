import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report 

def main(): 
    # ----------------------------------------------------------------------------------------
    # Scientific Computation (Data Wrangling Section)
    spotify_file = 'spotify_songs.csv'
    spotify_df = pd.read_csv(spotify_file) 
    spotify_df.drop_duplicates(inplace=True)

    # for song era use (ex: 70s, 80s, 90s)
    # did not use dt because some years are not formatted as yyyy-mm-dd
    num_years = []
    for year in spotify_df['track_album_release_date']:
        num_year = int(str(year)[:4]) # extract first 4 digits
        num_years.append(num_year) # append as a year

    spotify_df['Year'] = num_years

    # create decade column for visualizing how decade may be correlated with 
    # patterns in numerical averages
    # ex: songs in 1980s would likely have high values for danceability
    spotify_df['Decade'] = (spotify_df['Year'] // 10 ) * 10 # arithmetic to extract decade

    # downselect necessary columns ()
    df_columns = spotify_df[[
    'track_name',
    'track_artist',
    'playlist_genre',
    'playlist_subgenre',
    'Year',
    'Decade',
    'danceability', # numeric features from this point below
    'energy',
    'loudness',
    'speechiness',
    'acousticness',
    'instrumentalness',
    'valence',
    'tempo']]

    # drop any corrupted data
    df_columns.dropna(inplace=True)

    # grouping data by genre
    genre_grouped = df_columns.groupby('playlist_genre')

    # select all rows and target columns only 
    # with numerical features for aggregation
    numerical_columns = df_columns.iloc[:, 4:13]

    # for each genre and each year, aggregate the numerical features
    numerical_col_means = numerical_columns.groupby([spotify_df['Year'], spotify_df['playlist_genre']]).mean()
    # ----------------------------------------------------------------------------------------
    # Data Visualization: Part One -- First Four Feautures
    
    # Tick Mark Labels and Line Formatting
    decade_ticks = [1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020]
    genres = ["pop", "rap", "rock", "latin", "r&b", "edm"]
    genre_colors = ['#071b8f','#4E8566','#803043','#c975b4','#b86018','#bf000a']

    # Figure 1 (First Four Line Plots)
    features_group_1 = ['danceability','energy','loudness','speechiness']
    fig1, axes1 = plt.subplots(2,2,figsize=(20, 10))
    fig1.suptitle("Visualization 1: Musical Feature Aggregations Over 63 Years")

    # turn axes into list for looping
    axes1_list = [axes1[0,0], axes1[0,1], axes1[1,0],axes1[1,1]]

    # plot different feature for each ax
    for feature1, ax in zip(features_group_1, axes1_list):
        
        # plot the lines and line colors for each genre 
        for genre, color in zip(genres, genre_colors):

            # calculate numerical aggregations, use for axes 
            genre_subset = df_columns[df_columns['playlist_genre'] == genre] # filter
            genre_yearly = genre_subset.groupby('Year')[feature1].mean() 
            smoothed1= genre_yearly.rolling(5, center=True).mean() # plot smooth trend lines

            ax.plot(
                genre_yearly.index,
                genre_yearly,
                linewidth=1.5,
                color=color,
                linestyle="--",
                alpha=0.5,
                )
            # plot smoothed trend lines

            ax.plot(
                genre_yearly.index,
                smoothed1,
                linewidth=1.7,
                color=color,
                alpha=0.8,
                label=genre
                )

        # titles, labels etc
        ax.set_title(feature1)
        ax.set_xticks(decade_ticks)
        ax.set_xlabel("Years")
        ax.set_ylabel("Numerical Feature") 
        ax.legend()

    plt.tight_layout()
    fig1.savefig("figure1.png")
    # ----------------------------------------------------------------------------------------
    # Figure 2 -- Next Four Features
    features_group_2 = ['acousticness','instrumentalness','valence','tempo']
    fig2, axes2 = plt.subplots(2,2,figsize=(20, 10))
    fig2.suptitle("Visualization 2: Musical Feature Aggregations Over 63 Years")

    # turn axes into list for iteration
    axes2_list = [axes2[0,0], axes2[0,1], axes2[1,0],axes2[1,1]]

    # plot feature for each ax
    for feature2, ax in zip(features_group_2, axes2_list):

        for genre, color in zip(genres, genre_colors):

            genre_subset = df_columns[df_columns['playlist_genre'] == genre] # filter
            genre_yearly = genre_subset.groupby('Year')[feature2].mean() # averages for each year
            smoothed2= genre_yearly.rolling(5, center=True).mean() # smooth trend lines
             
            ax.plot(
                genre_yearly.index,
                genre_yearly,
                linewidth=1.5,
                color=color,
                linestyle="--",
                alpha=0.5,
                )

             # smooth lines
            ax.plot(
                genre_yearly.index,
                smoothed2,
                linewidth=1.7,
                color=color,
                alpha=0.8,
                label=genre
                )

        # labels, titles
        ax.set_title(feature2)
        ax.set_xticks(decade_ticks)
        ax.set_xlabel("Years")
        ax.set_ylabel("Numerical Feature") 
        ax.legend()

    plt.tight_layout()
    fig2.savefig("figure2.png")
# ----------------------------------------------------------------------------------------
# ML KNN Model and Confusion Matrix Visualization
# Data Visualization: Part 2

    # feature x: numerical song features, drop non-numerical 
    X = numerical_columns.drop(columns=['Year', 'Decade'])
    y = df_columns['playlist_genre'] # target vector y is song genre

    # train model (training/testing features)
    X_train, X_test, y_train, y_test = \
        train_test_split( \
        X, y, test_size=0.25, random_state=100, stratify=y)
    
    # scale numerical features for uniform magnitude 
    scaler = StandardScaler()
    scaler.fit(X_train)

    # transform, wrap back into df
    X_train = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    # use KNN Classifier Model with neighbors = 5
    model_knn = KNeighborsClassifier(n_neighbors=5)
    model_knn.fit(X_train, y_train)

    # make predictions
    y_pred = model_knn.predict(X_test)

    # print accuracy of the model
    print('Model Accuracy:',model_knn.score(X_test, y_test),"\n")

    # confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=model_knn.classes_)

    fig, axes = plt.subplots()
    cm_display.plot(ax=axes)

    axes.set(title="")
    fig.savefig('genre_matrix')

    # classification report
    class_rep = classification_report(y_test, y_pred)
    print("class rep is: ", class_rep)
# ----------------------------------------------------------------------------------------
# ML KNN Model Testing -- User Input 
    print("This program will predict a song's genre.")
    print("Notes:\n")
    print("         I recommend the website 'tunebat.com' at your own discrection to look up values for a song.\n")
    print("         If an attribute value is unknown, hit enter. The df average will be used.")
    print("         'Loudness' is measured in decibels (dB). This value is usually negative.\n")
    print("         The values entered MUST be decimals < 1, with the exception of 'loudness'. The website\n ")
    print("         uses integers (ex: if danceability is 60, this should be input as 0.6)")

    user_input_feat = [] # empty list to append user's input
    proceed = True # flag variable for tracking if user wants to proceed or quit

    start_prog = input("Please enter to start, or 'q' to quit.\n").strip()

    if start_prog == 'q': # check if user want to quit
        proceed = False # set bool var to false if they do
 
    for feature_name in X.columns: # for feature in numerical columns

        # if false, loop until finished and hits conditional branching outside loop
        # if, true, prog will proceed
        if proceed: 

            # calculate safe numbers in case user does not have/know a value for that feature
            # feature, avg, min, and max for default values
            f_avg = X[feature_name].mean()
            f_min = X[feature_name].min()
            f_max = X[feature_name].max()

            valid_input = False # dummy var to stop looping if needed

            while not valid_input: 
            # loop will run until user enters valid input, then break out to for loop to move on to next feature
                print("Enter a value WITHIN the feature range. Should be decimal < 1 with exception of 'loudness' (or 'q' to quit).")
                print("Feature: "+feature_name.upper()+"\n")
                print("Average:",f_avg )
                print("Maximum:",f_max )
                print("Minimum:",f_min,"\n" )

                user_input = input("Enter a value: \n").strip() # get value 

                if user_input.strip() == "": # if they hit 'enter'
                    print("Average used.") # just use avg
                    user_input_feat.append(f_avg) # add to user list
                    valid_input = True # stop looping, move onto next feature

                # check for quit condition again 
                elif user_input.lower() == 'q':
                    valid_input = True
                    proceed = False 

                else: # if numeric, modify string so that decimals/negatives do not crash prog
                    string_check = user_input # checker variable
                    # remove non-numeric exceptions (negative sign and decimals)
                    if string_check[0] == '-': # check for negative sign 

                        string_check = string_check[1:] # remove negative sign, store every after
                    
                    split_val = string_check.split('.') # split string at decimal points

                        # join to create a purely numerical string 
                        # check if this new string only contains numbers and if they didn't hit enter
                    if "".join(split_val).isdigit() and string_check != "": 

                        value_float = float(user_input) # turn original input into decimal point
                        user_input_feat.append(value_float) # append to list of values
                        valid_input = True # stop looping, move onto next feature
                        
                    else: # if non-numeric input
                        print("Invalid input. Please either hit the 'enter' key or enter a numerical value.\n")

    # Once finished, perform final case check:
    # case 1: predict genre using user input (if proceeded)
    # case 2: or quit progam

    # analyze user's df, use column with vector X (feature columns)
    if proceed: # if they input all values, make a prediction

        # test the accuracy of the model using new input
        user_df = pd.DataFrame([user_input_feat], columns=X.columns) # wrap into df to match headers
        user_scaled = scaler.transform(user_df) # scale/transform user's values
        user_predict = model_knn.predict(user_scaled) # make a prediction 
        print("result: ", "".join(user_predict))

    elif proceed == False: # if they quit, print goodbye

        print("You've quit this program, goodbye!")

if __name__ == '__main__':
    main()
