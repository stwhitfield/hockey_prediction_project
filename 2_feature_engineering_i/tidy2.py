import json
import pandas as pd
from csv import writer
import os

class TidyData:
    data_dir = os.getcwd() + "/raw_data/"
    tidy_file = "tidy2.csv"
    nullToken = None
    current_periods = []
    current_teams = tuple()

    @staticmethod
    def append_header(file_name):
        with open(file_name, 'a+', newline='') as write_obj:
            csv_writer = writer(write_obj)
            csv_writer.writerow(["gameId", "season", "teamHome", "teamAway", "eventType", "eventTeam", "period", "periodTime", "eventSide", "coordinateX", "coordinateY", "shooterName", "goalieName", "shotType", "emptyNet", "strength"])
    
    @staticmethod
    def valueOrNull(key, obj):
        if key in obj:
            value = obj[key]
        else:
            value = TidyData.nullToken
        return value
    
    @staticmethod
    def getPlayInfo(play):
        about = play["about"]
        periodTime = about["periodTime"]
        period = about["period"]
        eventTeam = play["team"]["name"]
        if(period == 5):
            eventSide = "shootout"
        else:
            team = TidyData.current_teams.index(eventTeam)
            current_period = TidyData.current_periods[period - 1]
            if team == 0:
                eventSide = TidyData.valueOrNull("rinkSide", current_period["home"])
            else:
                eventSide = TidyData.valueOrNull("rinkSide", current_period["away"])
        result = play["result"]
        eventType = result["event"]
        if eventType == "Goal":
            eventType = 1
        else:
            eventType = 0
        coordinates = play["coordinates"]
        coordinateX = TidyData.valueOrNull("x", coordinates)
        coordinateY = TidyData.valueOrNull("y", coordinates)
        shooterName = play["players"][0]["player"]["fullName"]
        goalieName = play["players"][-1]["player"]["fullName"]
        shotType = TidyData.valueOrNull("secondaryType", result)
        if eventType == 1:
            emptyNet = TidyData.valueOrNull("emptyNet", result)
            strength = result["strength"]["code"]
        else:
            emptyNet = TidyData.nullToken
            strength = TidyData.nullToken
        return [eventType, eventTeam, period, periodTime, eventSide, coordinateX, coordinateY, shooterName, goalieName, shotType, emptyNet, strength]

    @staticmethod
    def cleanData(folder_path):
        for filename in os.listdir(folder_path):
        
            # Parse the filename
            name = filename.split('.json')[0]
            is_json = filename[-5:] == '.json'

            # If it's a json:
            if is_json:
                # Open and load file
                json_file = open(f"{folder_path}{filename}")
                data = json.load(json_file)

                with open(TidyData.tidy_file, 'a+', newline='') as write_obj:
                    csv_writer = writer(write_obj)
                    gameId = data["gamePk"]
                    season = data["gameData"]["game"]["season"]
                    teamHome = data["gameData"]["teams"]["home"]["name"]
                    teamAway = data["gameData"]["teams"]["away"]["name"]
                    TidyData.current_teams = (teamHome, teamAway)
                    periods = data["liveData"]["linescore"]["periods"]
                    TidyData.current_periods = periods
                    plays = [x for x in data["liveData"]["plays"]["allPlays"] if x["result"]["event"] == "Shot" or x["result"]["event"] == "Goal"]
                    playData = map(TidyData.getPlayInfo, plays)

                    for play in playData:
                        csv_writer.writerow([gameId,season,teamHome,teamAway] + play)
                json_file.close()