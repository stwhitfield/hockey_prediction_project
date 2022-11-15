import json
import pandas as pd
from csv import writer
import os
from collections import deque
from datetime import timedelta

class TidyData:
    data_dir = os.getcwd() + "/raw_data"
    tidy_file = "tidy.csv"
    null_token = None

    def __init__(self, current_periods = [], current_teams = tuple()):
        self.current_periods = current_periods
        self.current_teams = current_teams
        self.penalty_boxes = {
            current_teams[0]: deque(),
            current_teams[1]: deque()
        }

    @staticmethod
    def append_header(file_name):
        with open(file_name, 'a+', newline='') as write_obj:
            csv_writer = writer(write_obj)
            csv_writer.writerow(["gameId", "season", "teamHome", "teamAway", "eventType", "eventTeam", "period", "periodTime", "eventSide", "coordinateX", "coordinateY", "shooterName", "goalieName", "shotType", "emptyNet", "strength", "homePlayersOnIce", "awayPlayersOnIce"])
    
    @staticmethod
    def valueOrNull(key, obj):
        if key in obj:
            value = obj[key]
        else:
            value = TidyData.null_token
        return value

    @staticmethod
    def toTime(minutes):
        if type(minutes) == int:
            return timedelta(minutes = minutes)
        else:
            min = int(minutes.split(":")[0])
            sec = int(minutes.split(":")[1])
            return timedelta(minutes = min, seconds = sec)
    
    @staticmethod
    def toFullTimeLength(period, delta_time):
        periods_time = TidyData.toTime(20 * (period - 1))
        total_time = periods_time + delta_time
        return total_time

    def getOpponentTeam(self, eventTeam):
        team_index = self.current_teams.index(eventTeam)
        if team_index == 1:
            return self.current_teams[0]
        else:
            return self.current_teams[1]

    def onPenalty(self, eventTeam, period, periodTime, penaltySeverity, penaltySecondaryType, penaltyMinutes, player):
        penalty = {
            "severity": penaltySeverity,
            "minutes": TidyData.toTime(penaltyMinutes),
            "player": player
        }

        if (penaltySeverity != "Major" and penaltySecondaryType != "Fighting") and penaltySeverity != "Penalty Shot": # When is a fight both player got 5 minutes penalty but there still is 5v5 on the ice
            if penaltyMinutes == 2:
                penalty["remainingMinors"] = 1
            if penaltyMinutes == 4:
                penalty["remainingMinors"] = 2
            elif penaltyMinutes == 6:
                penalty["remainingMinors"] = 3

            penalty["finishTime"] = TidyData.toFullTimeLength(period, TidyData.toTime(periodTime) + penalty["minutes"])
            self.penalty_boxes[eventTeam].appendleft(penalty)

    def onGoal(self, eventTeam, period, periodTime):
        opponent_penalty_box = self.penalty_boxes[self.getOpponentTeam(eventTeam)]
        if (len(opponent_penalty_box) > len(self.penalty_boxes[eventTeam])) and len(opponent_penalty_box) > 0:
            if opponent_penalty_box[-1]["severity"] == "Minor":
                if opponent_penalty_box[-1]["remainingMinors"] == 1:
                    # print("GOAL OF " + eventTeam + ":    " + str(period) + "  - " + periodTime)
                    opponent_penalty_box.pop()
                else:
                    opponent_penalty_box[-1]["remainingMinors"] -= 1
                    opponent_penalty_box[-1]["minutes"] = TidyData.toTime(2 * opponent_penalty_box[-1]["remainingMinors"])
                    opponent_penalty_box[-1]["finishTime"] = TidyData.toFullTimeLength(
                        period,
                        TidyData.toTime(periodTime) + opponent_penalty_box[-1]["minutes"] 
                    )
            
    def checkPenaltyBoxes(self, period, periodTime):
        if period == 5:
            return 1, 1
        for team in self.current_teams:
            for player in self.penalty_boxes[team].copy():
                if TidyData.toFullTimeLength(period, TidyData.toTime(periodTime)) > player["finishTime"]:
                    self.penalty_boxes[team].remove(player)
        homePlayersOnIce = 5 - len(self.penalty_boxes[self.current_teams[0]])
        awayPlayersOnIce = 5 - len(self.penalty_boxes[self.current_teams[1]])
        # print(self.penalty_boxes)
        # print(str(period) + "-" + periodTime + ":   " + str(homePlayersOnIce) + " Vs. " + str(awayPlayersOnIce))
        return homePlayersOnIce, awayPlayersOnIce

    def getPlayInfo(self, play):
        about = play["about"]
        periodTime = about["periodTime"]
        period = about["period"]
        homePlayersOnIce, awayPlayersOnIce = self.checkPenaltyBoxes(period, periodTime)
        eventTeam = play["team"]["name"]
        result = play["result"]
        eventType = result["event"]
        if(period == 5):
            eventSide = "shootout"
        else:
            team = self.current_teams.index(eventTeam)
            current_period = self.current_periods[period - 1]
            if team == 0:
                eventSide = TidyData.valueOrNull("rinkSide", current_period["home"])
            else:
                eventSide = TidyData.valueOrNull("rinkSide", current_period["away"])
            if eventType == "Penalty":
                player = play["players"][0]["player"]["fullName"]
                penaltySeverity = result["penaltySeverity"]
                penaltySecondaryType = result["secondaryType"]
                penaltyMinutes = result["penaltyMinutes"]
                self.onPenalty(eventTeam, period, periodTime, penaltySeverity, penaltySecondaryType, penaltyMinutes, player)

        if eventType == "Goal":
            eventType = 1
            self.onGoal(eventTeam, period, periodTime)
            homePlayersOnIce, awayPlayersOnIce = self.checkPenaltyBoxes(period, periodTime)
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
            emptyNet = self.null_token
            strength = self.null_token
        return [eventType, eventTeam, period, periodTime, eventSide, coordinateX, coordinateY, shooterName, goalieName, shotType, emptyNet, strength, homePlayersOnIce, awayPlayersOnIce]

    @staticmethod
    def cleanData(folder_path):
        json_files = sorted(filter(lambda x: os.path.isfile(os.path.join(folder_path, x)), os.listdir(folder_path)))
        for filename in json_files:
            print(filename)
            json_file = open(os.path.join(folder_path, filename))
            data = json.load(json_file)

            with open(TidyData.tidy_file, 'a+', newline='') as write_obj:
                csv_writer = writer(write_obj)
                gameId = data["gamePk"]
                season = data["gameData"]["game"]["season"]
                teamHome = data["gameData"]["teams"]["home"]["name"]
                teamAway = data["gameData"]["teams"]["away"]["name"]
                td = TidyData(current_teams = (teamHome, teamAway))
                periods = data["liveData"]["linescore"]["periods"]
                td.current_periods = periods
                plays = [x for x in data["liveData"]["plays"]["allPlays"] if x["result"]["event"] == "Shot" or x["result"]["event"] == "Goal" or x["result"]["event"] == "Penalty"]
                playData = map(td.getPlayInfo, plays)

                for play in playData:
                    csv_writer.writerow([gameId,season,teamHome,teamAway] + play)
            json_file.close()

TidyData.append_header(TidyData.tidy_file)
TidyData.cleanData(TidyData.data_dir)

# How to read it?
# df = pd.read_csv(TidyData.tidy_file, na_filter=False)
# print(df)
