#!/usr/bin/env python
# coding: utf-8

# Primary author: Anuj Saini <br>
# Secondary author: Shawn Whitfield <br>
# Version: 2<br>
# <br>
# 2022-11-13
# SW changes: <br>
# -Changed feature titles to be more informative or clearer <br>
# -Changed Last Angle From Net feature to be the difference between it and the new angle <br>

import json
import pandas as pd
import csv
import os


current_periods = []
current_teams = tuple()
nullToken = None


def valueOrNull(key, obj):
        if key in obj:
            value = obj[key]
        else:
            value = nullToken
        return value


def getPlayInfo(play):
        about = play["about"]
        periodTime = about["periodTime"]
        period = about["period"]
        eventTeam = play["team"]["name"]
        if(period == 5):
            eventSide = "shootout"
        else:
            team = current_teams.index(eventTeam)
            current_period = current_periods[period - 1]
            if team == 0:
                eventSide = valueOrNull("rinkSide", current_period["home"])
            else:
                eventSide = valueOrNull("rinkSide", current_period["away"])
        result = play["result"]
        eventType = result["event"]
        if eventType == "Goal":
            eventType = 1
        else:
            eventType = 0
        coordinates = play["coordinates"]
        coordinateX = valueOrNull("x", coordinates)
        coordinateY = valueOrNull("y", coordinates)
        shooterName = play["players"][0]["player"]["fullName"]
        goalieName = play["players"][-1]["player"]["fullName"]
        shotType = valueOrNull("secondaryType", result)
        if eventType == 1:
            emptyNet = valueOrNull("emptyNet", result)
            strength = result["strength"]["code"]
        else:
            emptyNet = nullToken
            strength = nullToken
        return [eventType, eventTeam, period, periodTime, eventSide, coordinateX, coordinateY, shooterName, goalieName, shotType, emptyNet, strength]

import numpy as np
def get_distance(x1,x2, y1,y2):
    """
    Calculates the euclidean distance from the event coordinates (x,y) to the goal, 
    based on what side of the rink (Eventside) the event is.
    Returns the euclidean distance rounded to 4 decimal places.
    """
    coord1 = np.array([x1,x2])
    coord2 = np.array([y1,y2])
    
    
    # Distance is the norm of a-b
    try:
        distance = np.linalg.norm(coord1 - coord2)
    except:
        distance = 0
    return round(distance,4)


#Get time differece
import datetime as dt
def get_time_diff(t1, t2):
    start_dt = dt.datetime.strptime(t1, '%M:%S')
    end_dt = dt.datetime.strptime(t2, '%M:%S')
    diff = (end_dt - start_dt) 
    return diff.seconds 


f = open('tidy_features.csv', 'w', newline='')

# create the csv writer
csv_writer = csv.writer(f)

csv_writer.writerow(["gameId", "season", "teamHome", "teamAway", "eventType", "eventTeam", "period", "periodTime",
                        "eventSide", "coordinateX", "coordinateY", "shooterName", "goalieName", "shotType", 
                        "emptyNet", "strength", "last_coordinateX", "last_coordinateY", "lastEventType",
                        "last_periodTime","last_distanceFromNet","time_from_lastEvent", "rebound", "speed"])


for filename in os.listdir("raw_data"):
    # Parse the filename
    name = filename.split('.json')[0]
    is_json = filename[-5:] == '.json'
    print(filename)
    # If it's a json:
    if is_json:
        # Open and load file
        json_file = open("raw_data/"+filename)
        data = json.load(json_file)
        gameId = data["gamePk"]
        season = data["gameData"]["game"]["season"]
        teamHome = data["gameData"]["teams"]["home"]["name"]
        teamAway = data["gameData"]["teams"]["away"]["name"]
        current_teams = (teamHome, teamAway)
        periods = data["liveData"]["linescore"]["periods"]
        current_periods = periods
        allplays=data["liveData"]["plays"]["allPlays"]
        for i in range(0,len(allplays)):
            if allplays[i]["result"]["event"] == "Shot" or allplays[i]["result"]["event"] == "Goal":
                play= getPlayInfo(allplays[i])
                last_coordinates = allplays[i-1]["coordinates"]
                last_coordinateX = valueOrNull("x", last_coordinates)
                last_coordinateY = valueOrNull("y", last_coordinates)
                last_event = allplays[i-1]["result"]["event"]
                last_periodTime = allplays[i-1]["about"]["periodTime"]
                if last_coordinateX is None :
                    last_coordinateX = play[5]
                if last_coordinateY is None :
                    last_coordinateY = play[6]
                lastDistance = get_distance(play[5], play[6], last_coordinateX, last_coordinateY)
                time_last_event = get_time_diff(last_periodTime, play[3])
                rebound=False
                if last_event=="Shot":
                    rebound=True
                speed=0
                if time_last_event!=0:
                    speed=lastDistance/time_last_event
                csv_writer.writerow([gameId,season,teamHome,teamAway] + play + [last_coordinateX, last_coordinateY,
                                    last_event,last_periodTime, lastDistance,time_last_event, rebound, speed])
    json_file.close()
    #break              
f.close()

#Reading csv
df = pd.read_csv('tidy_features.csv')


def get_distance_from_post(eventSide,x,y):
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


def periodTime_to_seconds(period, periodTime):
    """
    Takes a period (numpy.int64) and a periodTime (string in format 'mm:ss')  
    and returns the number of seconds since the beginning of the game.
    """
    periodSecs = (period-1)*20*60 # periods are 20 mins, 60 secs per min
    
    mins = int(periodTime.split(':')[0]) * 60 # multiply num mins by 60
    secs = int(periodTime.split(':')[1]) # seconds are fine as is
    totalSecs = periodSecs + mins + secs
    return totalSecs


# Use clunky list comprehensions to get lists that apply functions that use different dataframe columns 
df["distanceFromNet"] = [get_distance_from_post(df['eventSide'][i],df['coordinateX'][i],df['coordinateY'][i]) for i,r, in df.iterrows()]
df["angleFromNet"] = [get_angle(df['eventSide'][i],df['coordinateX'][i],df['coordinateY'][i]) for i,r, in df.iterrows()] 
df["angleFromNetLastEvent"] = [get_angle(df['eventSide'][i],df['last_coordinateX'][i],df['last_coordinateY'][i]) for i,r, in df.iterrows()] 
df["isGoal"] = df['eventType']
df["emptyNet"] = [bool_to_digit(df['emptyNet'][i]) for i,r in df.iterrows()]
df['gameSeconds'] = [periodTime_to_seconds(df['period'][i],df['periodTime'][i])  for i,r in df.iterrows()]


#df['angleFromNetLastEvent'] = df["rebound"]==True
df["changeShotAngle"] = np.where(df["rebound"] == True, (df['angleFromNet'] - df['angleFromNetLastEvent']), 0)



# features: <br>
# <br>
# Game seconds (in seconds)(int)<br>
# Game period (int)<br>
# Coordinates (x,y, separate columns) (floats)<br>
# Shot distance (float)<br>
# Shot angle (in degrees)(float)<br>
# Shot type (str) <br>
# <br>
# Last event type (str) <br>
# Coordinates of the last event (x, y, separate columns) <br>
# Time from the last event (in seconds) (int) <br>
# Distance from the last event (float) <br>
# <br>
# Rebound (bool): True if the last event was also a shot, otherwise False (bool) <br>
# Change in shot angle; only include if the shot is a rebound, otherwise 0. (in degrees)(float) <br>
# “Speed”: defined as the distance from the previous event, divided by the time since the previous event. (float) <br>


# Get only the features the task requires
df = df[['gameId','gameSeconds','period','coordinateX', 'coordinateY','distanceFromNet','angleFromNet','shotType',
       'lastEventType','time_from_lastEvent','last_distanceFromNet','rebound','changeShotAngle','speed','isGoal']]


# output to csv
df.to_csv('feat_engineering_ii_data.csv')
