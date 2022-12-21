'''Doc'''
import re
import pandas as pd
import streamlit as st
from ift6758.ift6758.client.serving.serving_client import ServingClient
from ift6758.ift6758.client.game.game_client import GameClient

DEFAULT_MODEL = "xgb-featsel-rfe-best-model-params"

if 'game_client' not in st.session_state:
    st.session_state.game_client = GameClient()
if 'serving_client' not in st.session_state:
    st.session_state.serving_client = ServingClient()
if 'live_game' not in st.session_state:
    st.session_state.live_game = st.session_state.game_client.get_live_games()
if 'live_games' not in st.session_state:
    st.session_state.live_games, st.session_state.game_status = st.session_state.game_client.get_live_games()
if 'display_info' not in st.session_state:
    st.session_state.display_info = False
if 'model_name' not in st.session_state:
    st.session_state.model_name = DEFAULT_MODEL
if 'gral_info' not in st.session_state:
    st.session_state.gral_info = ''
if 'events' not in st.session_state:
    st.session_state.events = ''
if 'goals' not in st.session_state:
    st.session_state.goals = ''
if 'predicted_goals' not in st.session_state:
    st.session_state.predicted_goals = ''

def download_model(workspace, model, version):
    """Doc"""
    if workspace == "" or model == "" or version == "":
        st.write("Please fill all the fields")
        return 0
    request, code = st.session_state.serving_client.download_registry_model(workspace, model, version)
    if code == 200:
        st.session_state.model_name = model
        if st.session_state.display_info:
            st.session_state.game_client.reset_data()
            gral_info, events, goals, predicted_goals = st.session_state.game_client.ping_game(st.session_state.live_game, st.session_state.model_name)
            return gral_info, events, goals, predicted_goals
        return st.session_state.gral_info, st.session_state.events, st.session_state.goals, st.session_state.predicted_goals
    st.write(f"{request}")
    return st.session_state.gral_info, st.session_state.events, st.session_state.goals, st.session_state.predicted_goals

#=== Title
st.title('Hockey Visualization App')
st.subheader('Current Live Games')
#=== Sidebar
with st.sidebar:
    st.header("Select Model")
    with st.form("Model form", clear_on_submit = True):
        workspace = st.text_input("Workspace:", value = "ift6758-project")
        model = st.text_input("Model:")
        version = st.text_input("Version:")
        submitted = st.form_submit_button("Submit")
        if submitted:
            if model != st.session_state.model_name:
                if workspace == "" or model == "" or version == "":
                    st.write("Please fill all the fields")
                else:
                    st.session_state.gral_info, st.session_state.events, st.session_state.goals, st.session_state.predicted_goals = download_model(workspace, model, version)
            else:
                st.write ("You are already using this model")
    st.write(f"Prediction using **{st.session_state.model_name}** model")
#=== Ping Game
with st.form("Game Selection"):
    live_game = st.selectbox(
        'Game Id:',
        st.session_state.live_games
    )
    submitted = st.form_submit_button("Ping Game")
    if submitted:
        index = st.session_state.game_status["game_ids"].index(live_game)
        if len(st.session_state.live_games) != 0 and st.session_state.game_status["game_status"][index] == "Live":
            find_regex = re.compile('[0-9]+')
            st.session_state.live_game = find_regex.findall(live_game)[0]
            st.session_state.gral_info, st.session_state.events, st.session_state.goals, st.session_state.predicted_goals = st.session_state.game_client.ping_game(st.session_state.live_game, st.session_state.model_name)
            st.session_state.display_info = True
        else:
            st.write(f"{live_game} game has not started")
#=== Info about the game
if st.session_state.display_info:
    if st.session_state.gral_info['remaining_time'] == "END" or st.session_state.gral_info['remaining_time'] == "Final":
        st.write(f"Period: {st.session_state.gral_info['current_period']}  -  {st.session_state.gral_info['remaining_time']}")
    else:
        st.write(f"Period: {st.session_state.gral_info['current_period']}  -  {st.session_state.gral_info['remaining_time']} left")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write( f"**{st.session_state.gral_info['home_team']}** (home)")

    with col2:
        st.write(f"**{st.session_state.gral_info['away_team']}** (away)")

    col1, col2 = st.columns(2)
    col1.metric("Goals: xG(actual)", f"{round(st.session_state.predicted_goals[0], 1)}({st.session_state.goals[0]})", round(st.session_state.goals[0] - st.session_state.predicted_goals[0], 1))
    col2.metric("Goals: xG(actual)", f"{round(st.session_state.predicted_goals[1], 1)}({st.session_state.goals[1]})", round(st.session_state.goals[1] - st.session_state.predicted_goals[1], 1))

    show_events = pd.DataFrame(st.session_state.events["events"])
    show_events.columns = [
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
        "timeSincePP",
        "Prediction"
    ]
    reversed_data = show_events.iloc[::-1]
    st.write(reversed_data)