import requests
import json
import os

#Function to download files
def download_season_jsons(allseasons, path):
    #Traverse through all seasons
    print(f'Downloading regular season and playoff games for the following seasons: {allseasons}')
    for season in allseasons:
        season = "https://statsapi.web.nhl.com/api/v1/schedule?season="+str(season)
        season_response = requests.get(season)
        season_json = json.loads(season_response.content)

        for item in season_json["dates"]:
                games_id = map(lambda game: game["gamePk"], item["games"])

                for game_id in games_id:
                    # If the file has already been downloaded:
                    if f"{game_id}.json" in os.listdir(path):
                        #Print a message that the file is already there.
                        print(f"File {game_id}.json is already in directory {path}.")
                    else:
                        # Filter to be able to only select regular season (02) or playoff (03) games
                        game_type = str(game_id)[4:6]
                        # Go through and download all games from regular season or playoffs 
                        if game_type == '02' or game_type == '03':
                            #Traverse through all games
                            url="https://statsapi.web.nhl.com/api/v1/game/"+str(game_id)+"/feed/live/"
                            response = requests.get(url)
                            game = json.loads(response.content)
                            with open(path+str(game_id)+'.json', 'w', encoding='utf-8') as f:
                                json.dump(game, f, ensure_ascii=False, indent=4)
                                f.close()
    print(f'Downloading complete')
                                
                                
def download_game_json(gameid,path):
    url=f"https://statsapi.web.nhl.com/api/v1/game/{gameid}/feed/live/"
    response = requests.get(url)
    game = json.loads(response.content)
    if f"{gameid}.json" in os.listdir(path):
        print(f"File {gameid}.json already present in dir {path}")
    else:
        print(f'Downloading {gameid} to {path}')
        with open(path+gameid+'.json', 'w', encoding='utf-8') as f:
            json.dump(game, f, ensure_ascii=False, indent=4)
            f.close()
    print(f'Downloading complete')