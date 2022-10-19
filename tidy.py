import json
import pandas as pd
from csv import writer
import os

class TidyData:
    data_dir = os.getcwd() + "/raw_data/"
    tidy_file = "tidy.csv"
    nullToken = None
    current_periods = []
    current_teams = tuple()

    @staticmethod
    def append_header(file_name):
        with open(file_name, 'a+', newline='') as write_obj:
            csv_writer = writer(write_obj)
            csv_writer.writerow(["gameId", "teamHome", "teamAway", "eventType", "eventTeam", "period", "periodTime", "eventSide", "coordinateX", "coordinateY", "shooterName", "goalieName", "shotType", "emptyNet", "strength"])
    
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
        #json_files = [pos_json for pos_json in os.listdir(folder_path) if pos_json.endswith('.json')]
        json_files = sorted(filter(lambda x: os.path.isfile(os.path.join(folder_path, x)), os.listdir(folder_path)))

        for filename in json_files:
            json_file = open(os.path.join(folder_path, filename))
            data = json.load(json_file)

            with open(TidyData.tidy_file, 'a+', newline='') as write_obj:
                csv_writer = writer(write_obj)
                gameId = data["gamePk"]
                teamHome = data["gameData"]["teams"]["home"]["name"]
                teamAway = data["gameData"]["teams"]["away"]["name"]
                TidyData.current_teams = (teamHome, teamAway)
                periods = data["liveData"]["linescore"]["periods"]
                TidyData.current_periods = periods
                plays = [x for x in data["liveData"]["plays"]["allPlays"] if x["result"]["event"] == "Shot" or x["result"]["event"] == "Goal"]
                playData = map(TidyData.getPlayInfo, plays)

                for play in playData:
                    csv_writer.writerow([gameId,teamHome,teamAway] + play)
            json_file.close()

TidyData.append_header(TidyData.tidy_file)
TidyData.cleanData(TidyData.data_dir)

# How to read it?
# df = pd.read_csv(TidyData.tidy_file, na_filter=False)
# print(df)
