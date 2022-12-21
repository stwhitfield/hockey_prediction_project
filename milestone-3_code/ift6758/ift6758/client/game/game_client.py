'''Doc'''
import os
import math
import json
import logging
from collections import deque
from datetime import timedelta
import pandas as pd
from numpy import arctan
import requests
from ift6758.ift6758.client.serving.serving_client import ServingClient

# Env variables
PATH_TO_JSON = './'
serving_client = ServingClient()
# Configuring log
logging.basicConfig(
    filename = os.environ.get("FLASK_LOG", "flask.log"),
    level = logging.INFO,
    format = '%(asctime)s %(levelname)s %(name)s : %(message)s'
)
logger = logging.getLogger(__name__)

class GameClient:
    '''Doc'''
    def __init__(self):
        logger.info('------- game_client/__init__ ---------')
        self.path_to_json = PATH_TO_JSON
        self.previous_cordinate_x = ''
        self.previous_cordinate_y = ''
        self.previous_time = 0
        self.registered_events = 0
        self.previous_event_type = ""
        self.previous_shot_distance = 0
        self.previous_shot_angle = 0
        self.current_game_id = ''
        self.goals = [0,0]
        self.predicted_goals = [0,0]
        self.test_data = {"team":[], "events":[]}
        self.current_teams = tuple(["",""])
        self.penalty_boxes = {}

    @staticmethod
    def value_or_null(key, obj):
        '''Doc'''
        if key in obj:
            value = obj[key]
        else:
            value = 0
        return value

    @staticmethod
    def get_time_seconds(mm_ss):
        '''Doc'''
        minute, second = mm_ss.split(":")
        period_time = ( int(minute) * 60 ) + int(second)
        return period_time

    @staticmethod
    def to_time(minutes):
        """Doc"""
        if isinstance (minutes, int):
            return timedelta(minutes = minutes)
        minu = int(minutes.split(":")[0])
        sec = int(minutes.split(":")[1])
        return timedelta(minutes = minu, seconds = sec)

    def to_full_time_length(self, period, delta_time):
        """Doc"""
        periods_time = self.to_time( 20 * ( period - 1 ) )
        total_time = periods_time + delta_time
        return total_time

    def get_distance (self, team, coordinate_x, coordinate_y, period):
        """Doc"""
        goal_distance_from_center = ( 200 / 2 ) - 11
        if team == self.current_teams[0]:
            # Event driven by the home team shooting to the rigth side of ice rink
            if period % 2 == 0:
                distance_x = goal_distance_from_center - coordinate_x
                distance = math.sqrt((distance_x ** 2) + (coordinate_y ** 2))
                try:
                    angle_radians = arctan( coordinate_y / distance_x )
                    angle_degree = angle_radians * ( 180 / 3.1416 )
                except ZeroDivisionError:
                    angle_radians = arctan( coordinate_y / 0.0000000001 )
                    angle_degree = round(angle_radians * ( 180 / 3.1416 ),-1)
                return round(distance,4), round(angle_degree,4)
            # Event driven by the home team shooting to the left side of ice rink
            distance_x = goal_distance_from_center + coordinate_x
            logger.info('dxcvxvdsdssx %s', distance_x)
            logger.info('d %s', coordinate_y)
            distance = math.sqrt((distance_x ** 2) + (coordinate_y ** 2))
            try:
                angle_radians = arctan( coordinate_y / distance_x )
                angle_degree = angle_radians * ( 180 / 3.1416 )
            except ZeroDivisionError:
                angle_radians = arctan( coordinate_y / 0.0000000001 )
                angle_degree = round(angle_radians * ( 180 / 3.1416 ),-1)
            return round(distance,4), round(angle_degree,4)
        # Event driven by the away team shooting to the left side of ice rink
        if period % 2 == 0:
            distance_x = goal_distance_from_center + coordinate_x
            distance = math.sqrt((distance_x ** 2) + (coordinate_y ** 2))
            try:
                angle_radians = arctan( coordinate_y / distance_x )
                angle_degree = angle_radians * ( 180 / 3.1416 )
            except ZeroDivisionError:
                angle_radians = arctan( coordinate_y / 0.0000000001 )
                angle_degree = round(angle_radians * ( 180 / 3.1416 ),-1)
            return round(distance,4), round(angle_degree,4)
        # Event driven by the away team shooting to the right side of ice rink
        distance_x = goal_distance_from_center - coordinate_x
        distance = math.sqrt((distance_x ** 2) + (coordinate_y ** 2))
        try:
            angle_radians = arctan( coordinate_y / distance_x )
            angle_degree = angle_radians * ( 180 / 3.1416 )
        except ZeroDivisionError:
            angle_radians = arctan( coordinate_y / 0.0000000001 )
            angle_degree = round(angle_radians * ( 180 / 3.1416 ),-1)
        return round(distance,4), round(angle_degree,4)

    def get_coordinates(self, event):
        '''Doc'''
        coordinates = event["coordinates"]
        coordinate_x = self.value_or_null("x", coordinates)
        coordinate_y = self.value_or_null("y", coordinates)
        return coordinate_x, coordinate_y

    def check_penalty_boxes(self, period, period_time):
        '''Doc'''
        time = self.to_full_time_length(period, self.to_time(period_time))
        if period == 5:
            return [1,1,0]
        for team in self.current_teams:
            for player in self.penalty_boxes[team].copy():
                if time > player["finishTime"]:
                    self.penalty_boxes[team].remove(player)
        home_players_on_ice = 5 - len(self.penalty_boxes[self.current_teams[0]])
        away_players_on_ice = 5 - len(self.penalty_boxes[self.current_teams[1]])
        time_since_pp = 0
        if home_players_on_ice > away_players_on_ice:
            time_since_pp = (time - self.penalty_boxes[self.current_teams[1]][0]["startTime"]).seconds
        elif away_players_on_ice > home_players_on_ice:
            time_since_pp = (time - self.penalty_boxes[self.current_teams[0]][0]["startTime"]).seconds
        return [home_players_on_ice, away_players_on_ice, time_since_pp]

    def on_penalty(self, event_team, about, result, player):
        """Doc"""
        period_time = about["periodTime"]
        period = about["period"]
        penalty_severity = result["penaltySeverity"]
        penalty_secondary_type = result["secondaryType"]
        penalty_minutes = result["penaltyMinutes"]
        penalty = {
            "startTime": self.to_full_time_length(period, self.to_time(period_time)),
            "severity": penalty_severity,
            "minutes": self.to_time(penalty_minutes),
            "player": player
        }
        if (penalty_severity != "Major" and penalty_secondary_type != "Fighting"):
            if penalty_severity != "Penalty Shot":
                # When is a fight both player got 5 minutes penalty but there still is 5v5 on ice
                if penalty_minutes == 2:
                    penalty["remainingMinors"] = 1
                if penalty_minutes == 4:
                    penalty["remainingMinors"] = 2
                elif penalty_minutes == 6:
                    penalty["remainingMinors"] = 3

            penalty["finishTime"] = self.to_full_time_length(
                period,
                self.to_time(period_time) + penalty["minutes"]
            )
            self.penalty_boxes[event_team].appendleft(penalty)

    def on_goal(self, event_team, period, period_ime):
        """Doc"""
        opponent_penaltybox = self.penalty_boxes[self.current_teams[1]]
        if (len(opponent_penaltybox) > len(self.penalty_boxes[event_team])) and len(opponent_penaltybox) > 0:
            if opponent_penaltybox[-1]["severity"] == "Minor":
                if opponent_penaltybox[-1]["remainingMinors"] == 1:
                    opponent_penaltybox.pop()
                else:
                    opponent_penaltybox[-1]["remainingMinors"] -= 1
                    opponent_penaltybox[-1]["minutes"] = self.to_time(2 * opponent_penaltybox[-1]["remainingMinors"])
                    opponent_penaltybox[-1]["finishTime"] = self.to_full_time_length(
                        period,
                        self.to_time(period_ime) + opponent_penaltybox[-1]["minutes"] 
                    )

    @staticmethod
    def get_live_games():
        '''Doc'''
        game_ids = ["Error connecting ..."]
        #
        game_status = {
            "game_ids" : [],
            "game_status": []
        }
        #
        url = 'https://statsapi.web.nhl.com/api/v1/schedule'
        response = requests.get(url, timeout = 10)
        if response.status_code != 404:
            logger.info("Getting live games from api: %s", url)
            game_ids = []
            response_json = json.loads(response.content)
            games = response_json["dates"][0]["games"]
            for game in games:
                # if game["status"]["abstractGameState"] == 'Live':
                home_team = game['teams']['home']['team']['name']
                away_team = game['teams']['away']['team']['name']
                game_id = str(game['gamePk'])
                game_ids.append(home_team + ' vs. ' + away_team + " (" + game_id + ")")
                #
                game_status["game_ids"].append(home_team + ' vs. ' + away_team + " (" + game_id + ")")
                game_status["game_status"].append(game["status"]["abstractGameState"])
                #
            return game_ids, game_status
        logger.error("Error connecting to api: %s", url)
        return game_ids

    @staticmethod
    def get_game_raw_data(game_id):
        '''Documentation'''
        game_data = {}
        url = f'https://statsapi.web.nhl.com/api/v1/game/{game_id}/feed/live/'
        response = requests.get(url, timeout = 10)
        if response.status_code != 404:
            logger.info('Getting game data (%s) from: %s', game_id, url)
            game_data = json.loads(response.content)
            with open(str(game_id)+'.json', 'w', encoding='utf-8') as file:
                json.dump(game_data, file, ensure_ascii=False, indent=4)
            file.close()
            return game_data
        logger.error("Error connecting to api: %s", url)
        return game_data

    @staticmethod
    def get_game_gral_info(data):
        '''Documentation'''
        gral_info ={
            'home_team': data["gameData"]["teams"]["home"]["name"],
            'away_team': data["gameData"]["teams"]["away"]["name"],
            'remaining_time': data["liveData"]["linescore"]["currentPeriodTimeRemaining"],
            'current_period': data["liveData"]["linescore"]["currentPeriod"],
        }
        return gral_info
    
    def goals_count(self, goals, value, prediction):
        '''Doc'''
        if prediction:
            if goals == self.current_teams[0]:
                self.predicted_goals[0] += value
                return 0
            self.predicted_goals[1] += value
            return 0
        self.goals[1] = int(goals["away"])
        self.goals[0] = int(goals["home"])
        return 0

    def get_play_info(self, event):
        '''Doc'''
        mapping_events = {
            'Blocked Shot': 0,
            'Faceoff': 1,
            'Game Official': 2,
            'Giveaway': 3,
            'Goal': 4,
            'Hit': 5,
            'Missed Shot': 6,
            'Official Challenge': 7,
            'Penalty': 8,
            'Period End': 9,
            'Period Official': 10,
            'Period Ready': 11,
            'Period Start': 12,
            'Shootout Complete': 13,
            'Shot': 14,
            'Stoppage': 15,
            'Takeaway': 16
        }
        mapping_shot = {
            'Backhand': 0,
            'Deflected': 1,
            'Slap Shot': 2,
            'Snap Shot': 3,
            'Tip-In': 4,
            'Wrap-around': 5,
            'Wrist Shot': 6,
            #Review
            'Poke': 7,
            'Batted': 8
            }
        # Register statistics from discarded events
        if isinstance(event, list) and event[0] == "skip" :
            self.previous_cordinate_x = event[2]
            self.previous_cordinate_y = event[3]
            self.previous_time = int(self.get_time_seconds(event[1]))
            try:
                self.previous_event_type = mapping_events[event[4]]
            except KeyError:
                pass
            return "skip"

        about = event["about"]
        period_time = about["periodTime"]
        period_time_seconds = self.get_time_seconds(period_time)
        period = about["period"]
        bonus = self.check_penalty_boxes(period, period_time)
        eventTeam = event["team"]["name"]
        result = event["result"]
        eventType = result["event"]
        event_type_encoded = mapping_events[eventType]
        coordinates = event["coordinates"]
        coordinateX = self.value_or_null("x", coordinates)
        coordinateY = self.value_or_null("y", coordinates)
        shotType = self.value_or_null("secondaryType", result)
        shot_distance, shot_angle = self.get_distance(eventTeam ,coordinateX, coordinateY, period)
        total_time = period_time_seconds + ((period * 20*60) - 20*60)
        empty_net = 0
        rebound = 0
        change_shot_angle = 0.0

        self.goals_count(about["goals"], 0, False)
        if period == 5:
            pass
        else:
            if eventType == "Penalty":
                player = event["players"][0]["player"]["fullName"]
                self.on_penalty(eventTeam, about, result, player)
        self.previous_shot_distance = round(
            math.sqrt( ( coordinateX - self.previous_cordinate_x ) ** 2 + ( coordinateY - self.previous_cordinate_y ) ** 2 ),
            4
        )
        # period_time = int(GameClient.get_time_seconds(period_time))
        # totalTime = period_time + ((period * 20*60) - 20*60)
        if eventType == "Goal":
            empty_net = 1
            self.on_goal(eventTeam, period, period_time)
            bonus = self.check_penalty_boxes(period, period_time)
        elif eventType == "Penalty":
            self.previous_time = period_time_seconds
            self.previous_cordinate_x = coordinateX
            self.previous_cordinate_y = coordinateY
            self.previous_event_type = event_type_encoded
            return "skip"
        if shotType == 0:
            shot_type_encoded = "NA"
        else:
            shot_type_encoded = mapping_shot[shotType]
        if self.previous_event_type == mapping_events["Shot"]:
            rebound = 1
            change_shot_angle = round(shot_angle - self.previous_shot_angle,4)
        elif self.previous_event_type == mapping_events["Stoppage"]:
            self.previous_cordinate_x = coordinateX
            self.previous_cordinate_y = coordinateY
            self.previous_shot_distance = 0.0
        last_time = period_time_seconds - self.previous_time
        try:
            speed = round(self.previous_shot_distance / last_time,4)
        except ZeroDivisionError:
            speed = 0.0
        features = [
            total_time,
            period,
            coordinateX,
            coordinateY,
            shot_distance,
            shot_angle,
            shot_type_encoded,
            empty_net,
            self.previous_event_type,
            self.previous_cordinate_x,
            self.previous_cordinate_y,
            last_time,
            self.previous_shot_distance,
            rebound,
            change_shot_angle,
            speed,
            bonus[0],
            bonus[1],
            bonus[2]
        ]
        self.previous_time = period_time_seconds
        self.previous_shot_angle = shot_angle
        self.previous_cordinate_x = coordinateX
        self.previous_cordinate_y = coordinateY
        self.previous_event_type = event_type_encoded
        self.test_data["team"].append(eventTeam)
        return features

    def clean_data(self, data):
        '''Doc'''
        events = []
        events_processed = self.registered_events
        logger.info('Found %s events already processed (%s)', events_processed, self.current_game_id)
        for event in data["liveData"]["plays"]["allPlays"]:
            if event["result"]["event"] == "Shot" or event["result"]["event"] == "Goal" or event["result"]["event"] == "Penalty":
                if event["result"]["event"] != "Penalty":
                    events_processed -= 1
                if events_processed < 0:
                    events.append(event)
            else:
                coordinate_x, coordinate_y = self.get_coordinates(event)
                period_time = event["about"]["periodTime"]
                event_type = event["result"]["event"]
                try:
                    event_team = event["team"]["name"]
                except KeyError:
                    event_team = ""
                period = event["about"]["period"]
                unused_events_data = [
                    "skip",
                    period_time,
                    coordinate_x,
                    coordinate_y,
                    event_type,
                    event_team,
                    period
                ]
                events.append(unused_events_data)
        # return events
        if events_processed == 0 and self.registered_events != 0:
            logger.info('No new events found for the game')
            return 0
        play_data = map(self.get_play_info, events)
        for play in play_data:
            if play != "skip":
                dataframe = pd.DataFrame(play).T
                dataframe.columns = [
                    "time",
                    "period",
                    "coordinateX",
                    "coordinateY",
                    "shotDistance",
                    "shotAngle",
                    "shotType",
                    "emptyNet",
                    "lastEventType",
                    "lastCoordinateX",
                    "lastCoordinateY",
                    "lastTime",
                    "lastShotDistance",
                    "rebound",
                    "changeShotAngle",
                    "speed",
                    "friendlyPlayersOnIce",
                    "opposingPlayersOnIce",
                    "timeSincePP"
                ]
                # prediction = serving_client.predict(dataframe.to_dict())
                self.registered_events += 1
                # self.test_data["events"].append(play + prediction)
                self.test_data["events"].append(play)
        prediction = pd.DataFrame(self.test_data["events"])
        prediction.columns = [
            "time",
            "period",
            "coordinateX",
            "coordinateY",
            "shotDistance",
            "shotAngle",
            "shotType",
            "emptyNet",
            "lastEventType",
            "lastCoordinateX",
            "lastCoordinateY",
            "lastTime",
            "lastShotDistance",
            "rebound",
            "changeShotAngle",
            "speed",
            "friendlyPlayersOnIce",
            "opposingPlayersOnIce",
            "timeSincePP"
        ]
        logger.info('Processing %s new events (%s)', len(prediction), self.current_game_id)
        prediction = serving_client.predict(prediction.to_dict())
        for index, event in enumerate(self.test_data["events"]):
            event.append(prediction[index])
            if prediction[index] > 0.8:
                self.goals_count(self.test_data["team"][index], prediction[index], True)
        return 0

    def reset_data(self):
        '''Doc'''
        self.previous_cordinate_x = ''
        self.previous_cordinate_y = ''
        self.previous_time = 0
        self.registered_events = 0
        self.previous_event_type = ""
        self.previous_shot_distance = 0
        self.previous_shot_angle = 0
        self.current_game_id = ''
        self.goals = [0,0]
        self.predicted_goals = [0,0]
        self.test_data = {"team":[], "events":[]}
        self.current_teams = tuple(["",""])
        self.penalty_boxes = {}

    def ping_game(self, game_id, model):
        '''Doc'''
        if self.current_game_id != game_id:
            self.reset_data()
        self.current_game_id = game_id
        game_raw_data = self.get_game_raw_data(game_id)
        if game_raw_data != {}:
            gral_info = self.get_game_gral_info(game_raw_data)
            self.current_teams = (gral_info['home_team'], gral_info['away_team'])
            self.penalty_boxes = {
                gral_info['home_team']: deque(),
                gral_info['away_team']: deque()
            }
            if self.registered_events == len(game_raw_data["liveData"]["plays"]["allPlays"]):
                logger.info('No new events found for the game: %s', game_id)
            logger.info('Using %s (%s)', model, game_id)
            self.clean_data(game_raw_data)
            return gral_info, self.test_data, self.goals, self.predicted_goals
        return 0
