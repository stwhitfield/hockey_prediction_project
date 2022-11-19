# Author: Shawn Whitfield
# Version: 1
# Date: 2022-11-06

import DataScraping
from tidy2 import TidyData
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#acquire raw data
path = "raw_data/"
allseasons = ["20152016",'20162017','20172018','20182019','20192020']
DataScraping.download_season_jsons(allseasons, path)

# Tidy the data to an initial file
TidyData.append_header(TidyData.tidy_file)
TidyData.cleanData(TidyData.data_dir)

# Read the tidied file into a dataframe
df = pd.read_csv('tidy2.csv', encoding = 'utf-8')

# Clean the file of NA in coordinateX or coordinateY
# Note: unclear if they want us to do this or see the error of our ways
df = df.dropna(axis=0, subset=['eventSide','coordinateX','coordinateY'])

df['coordinateX'] = pd.to_numeric(df['coordinateX'],errors='coerce')
df['coordinateY'] = pd.to_numeric(df['coordinateY'],errors='coerce')
df['emptyNet'] = df['emptyNet'].astype(bool)

# Helper functions to make particular features:
def get_distance(eventSide,x,y):
    """
    Calculates the euclidean distance from the event coordinates (x,y) to the goal, 
    based on what side of the rink (Eventside) the event is.
    Returns the euclidean distance rounded to 4 decimal places.
    """
    event_pos = np.array([x,y])
    # Set the goal position based on the event side (the eventSide marks the team making the shot, the goal is on other side)
    if eventSide == 'right':
        goal_pos = np.array([-89.0,0.0])
    else:
        goal_pos = np.array([89.0,0.0])
    # Distance is the norm of a-b
    
    distance = np.linalg.norm(event_pos - goal_pos)
    
    return round(distance,4)

def get_angle(eventSide,x,y):
    """
    Calculates the angle between the goal (treated as the origin) and the event coordinates (x,y).
    Returns the angle in degrees, rounded to 4 decimal places.
    """
    # Set the goal position as (0,0)
    goal_pos = np.array([0.0,0.0])
    
    # Adjust the event coordinates to account for the goal being at position (0,0)
    # x is moved left or right by 89 depending on eventSide, y is unchanged.
    if eventSide == 'right':
        event_pos = np.array([x+89.0,y])
    else:
        # if eventSide team is on the left, flip it so it's on the right and then adjust
        # This is done so the angle is correct with relation to the net
        event_pos = np.array([(-x)+89.0,y])
        
    # Angle from origin to point (x,y) is np.arctan2()
    angle = np.arctan2(event_pos[1],event_pos[0])
    
    # Convert the angle to degrees and return
    return round(np.rad2deg(angle),4)

def bool_to_digit(x):
    """
    Turns a True to 1 and anything else to 0.
    """
    if x == True:
        return 1
    else:
        return 0

    
# Use clunky list comprehensions to get lists that apply functions that use different dataframe columns 
distanceFromNet = [get_distance(df['eventSide'][i],df['coordinateX'][i],df['coordinateY'][i]) for i,r, in df.iterrows()]
angleFromNet = [get_angle(df['eventSide'][i],df['coordinateX'][i],df['coordinateY'][i]) for i,r, in df.iterrows()] 
isGoal = df['eventType']
emptyNet = [bool_to_digit(df['emptyNet'][i]) for i,r in df.iterrows()]

# Save the lists into a new dataframe
data = pd.DataFrame({'gameId': df['gameId'].astype(str),
                    'distanceFromNet': distanceFromNet ,
                     'angleFromNet': angleFromNet,
                     'isGoal': isGoal,
                     'emptyNet': emptyNet
                    })

#"You will use the 2015/16 - 2018/19 regular season data to create your training and validation sets"
# Filter data for regular season only
data = data[data['gameId'].astype(str).str[4:6] == '02']

# Now that data are treated the same, divide into train,test datasets
train_data = data[(data['gameId'].str[:4] == '2015') | (data['gameId'].str[:4] == '2016') | (data['gameId'].str[:4] == '2017') | (data['gameId'].str[:4] == '2018')]
test_data = data[data['gameId'].str[:4] == '2019']

# Export train_data and test_data as csv
train_data.to_csv('fe_train_data.csv')
test_data.to_csv('fe_test_data.csv')


# Make a histogram of shot counts (goals and no-goals separated), binned by distance
sns.histplot(data=train_data[train_data['isGoal'] ==0], x='distanceFromNet', color = 'grey', label = 'shots').set(xlabel='distance from net (ft)', ylabel='count')
sns.histplot(data=train_data[train_data['isGoal'] ==1], x='distanceFromNet', color = 'black', label = 'goals')
plt.legend()
plt.tight_layout()
plt.savefig('2-1a_dist_from_net.png')
plt.clf()

# Make a histogram of shot counts (goals and no-goals separated), binned by angle
sns.histplot(data=train_data[train_data['isGoal'] ==0], x='angleFromNet', color = 'grey', label = 'shots').set(xlabel='angle from net (degrees)', ylabel='count')
sns.histplot(data=train_data[train_data['isGoal'] ==1], x='angleFromNet', color = 'black', label = 'goals')
plt.legend()
plt.tight_layout()
plt.savefig('2-1b_angle_from_net.png')
plt.clf()

# Make a 2D histogram where one axis is the distance and the other is the angle. No need to separate goals and no-goals.
# The hint says to check out jointplots, but I feel this looks better.
sns.displot(train_data,x='distanceFromNet',y='angleFromNet', color = 'grey').set(xlabel='distance from net (ft)', ylabel='angle from net (degrees)')
plt.tight_layout()
plt.savefig('2-1c_2D_hist.png')
plt.clf()

# Jointplot is here if you want
# sns.jointplot(train_data,x='distanceFromNet',y='angleFromNet').set(xlabel='distance from net (ft)', ylabel='angle from net (degrees)')
# plt.tight_layout()
# plt.savefig('2-1c_2D_hist_jointplot.png')


#Now, create two more figures relating the goal rate, 
# i.e. #goals / (#no_goals + #goals), to the distance, and goal rate to the angle of the shot.

bin_num = 200

# Calculate distance histogram for shots that are goals
goal_hist, goal_bins = np.histogram(train_data[train_data['isGoal'] == 1]['distanceFromNet'], 
                                    bins = bin_num, 
                                    range = (0,200))

# Calculate distance histogram for all shots
all_hist, all_bins = np.histogram(train_data['distanceFromNet'], bins = bin_num, range = (0,200))

# Get the ratio: shots that are goals over all goals, binned by distance
goal_rate_hist = np.nan_to_num(goal_hist/all_hist)

plt.hist(goal_bins[:-1], goal_bins, weights=goal_rate_hist, color = 'grey')
plt.xlabel('distance from net (ft)')
plt.ylabel('goal rate (goals / all shots)')
plt.tight_layout()
plt.savefig('2-2a_goal_rate_dist.png')
plt.clf()

bin_num = 180

#Calculate angle histogram for shots that are goals
goal_hist, goal_bins = np.histogram(train_data[train_data['isGoal'] == 1]['angleFromNet'], 
                                    bins = bin_num, 
                                    range = (-180.0,180.0))
#Calculate angle histogram for all shots
all_hist, all_bins = np.histogram(train_data['angleFromNet'], bins = bin_num, range = (-180.0,180.0))

# Get the ratio: shots that are goals over all goals, binned by angle
goal_rate_hist = np.nan_to_num(goal_hist/all_hist)

plt.hist(goal_bins[:-1], goal_bins, weights=goal_rate_hist, color = 'grey')
plt.xlabel('angle from net (ft)')
plt.ylabel('goal rate (goals / all shots)')
plt.tight_layout()
plt.savefig('2-2b_goal_rate_angle.png')
plt.clf()


# From the outline:
# Finally, let’s do some quick checks to see if our data makes sense. 
# Unfortunately we don’t have time to do automated anomaly detection, 
# but we can use our “domain knowledge” for some quick sanity checks! 
# The domain knowledge is that “it is incredibly rare to score a non-empty 
# net goal on the opposing team from within your defensive zone”. Knowing 
# this, create another histogram, this time of goals only, binned by distance, 
# and separate empty net and non-empty net events. Include this figure in your 
# blogpost and discuss your observations. Can you find any events that have 
# incorrect features (e.g. wrong x/y coordinates)? If yes, prove that one 
# event has incorrect features.

# Make a histogram of goals only, binned by distance, and separate empty net and non-empty net events
sns.histplot(data = train_data[train_data['isGoal'] == 1], x='distanceFromNet', hue = 'emptyNet').set(xlabel='distance from net (ft)', ylabel='count')
# plt.yscale('log') # Plot on log scale so it's easier to see empty net events
plt.tight_layout()
plt.savefig('2-3a_goals_emptyNet.png')
plt.clf()

#Statement: “it is incredibly rare to score a non-empty net goal on the opposing team from within your defensive zone”
# define: defensive zone is 60 ft from the closest goal (team's goal) or 75 ft from the boards
unlikely = train_data[(train_data['distanceFromNet'] > 125) & (train_data['emptyNet'] == 0) & (train_data['isGoal'] == 1)]
print(unlikely.tail(20))

# go back to original dataframe and look up the event associated with the row
print(df.loc[960673])
