"""
If you are in the same directory as this file (app.py), you can run run the app using gunicorn:
    $ gunicorn --bind 0.0.0.0:<PORT> app:app
gunicorn can be installed via:
    $ pip install gunicorn
"""
import re
import os
from pathlib import Path
import logging
from flask import Flask, jsonify, request, make_response
from flask.logging import create_logger
import pandas as pd
import joblib
from comet_ml import API
from tensorflow import keras
import xgboost

# Variables
MODEL = "default"
MODEL_NAME = "xgb-featsel-rfe-best-model-params"
# MODEL_NAME = os.environ.get("DEFAULT_MODEL")
LOG_FILE = os.environ.get("FLASK_LOG", "flask.log")
COMET_API_KEY = os.environ.get("COMET_API_KEY")
CURRENT_PATH = str(Path(__file__).parent)

comet = API(COMET_API_KEY)

app = Flask(__name__)
log = create_logger(app)


# ============== Helpful class ==============
class Daraframe:
    """Perform some validations on the received data"""
    @staticmethod
    def delete_columns(dataframe, values: int):
        """Delete undesired columns"""
        log.info("Deleting undesired columns: %s", values)
        for i in values:
            if i in dataframe.columns:
                dataframe = dataframe.drop(columns=[i],axis=1)
        return dataframe

class Model:
    """Perform some validations on the received data"""
    @staticmethod
    def generate_variables(model_schema: dict):
        """Generate variables used to download model process"""
        model_workspace = model_schema["workspace"]
        model_registry = model_schema["registry"]
        model_version = model_schema["version"]
        adittional_path = "/" + model_workspace + "/" + model_registry + "/" + model_version
        path_to_model = CURRENT_PATH + "/" + adittional_path
        model_name = model_registry
        model_path = path_to_model + "/" + model_name
        return path_to_model, model_path, model_name

    @staticmethod
    def generate_path(path_to_model: str):
        """Generates path [MODEL_NAME] / [VERSION]"""
        if not os.path.exists(path_to_model):
            os.makedirs(path_to_model)
            log.info("Path \"%s\" generated", path_to_model)
        log.info("Path \"%s\" already exists", path_to_model)
        return path_to_model

    @staticmethod
    def load_model(model_path):
        """Loads model joblib/keras"""
        if re.search("best-nn", model_path):
            model = keras.models.load_model(model_path)
            return model
        model = joblib.load(open(model_path, 'rb'))
        return model

    @staticmethod
    def validate_model_exists(model_path: str):
        """Validates if model is already downloaded"""
        if Path(model_path).is_file():
            return True
        return False

    @staticmethod
    def download_model(model_schema: dict, path_to_model: str, model_path:str):
        """Download model"""
        model_workspace = model_schema["workspace"]
        model_registry = model_schema["registry"]
        model_version = model_schema["version"]
        try:
            comet_data = comet.get_registry_model_details(
                model_workspace,
                model_registry,
                model_version
            )
            comet.download_registry_model(
                model_workspace,
                model_registry,
                model_version,
                output_path=path_to_model, expand=True, stage=None
            )
            log.info("Model %s downloaded correctly", model_registry)
            file_path = Path(path_to_model + "/" + comet_data["assets"][0]["fileName"])
            if file_path is not model_path:
                file_path.rename(model_path)
                log.info("Naming your %s model", model_registry)
            return True
        except Exception as excep:
            log.error("Failed attempt to download model %s", excep)
        return False


# ============== Flask app ==============
@app.before_first_request
def before_first_request():
    """
    Hook to handle any initialization before the first request (e.g. load model,
    setup logging handler, etc.)
    """
    global MODEL
    global MODEL_NAME
    # Setup basic logging configuration
    logging.basicConfig(
        filename=LOG_FILE,
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s : %(message)s'
    )
    # Initialization before the first request (e.g. load default model)
    model_schema = {
        "workspace": "ift6758-project",
        "registry": MODEL_NAME,
        "version": "1.0.0"
    }
    # get model name and path
    path_to_model, model_path, model_name = Model.generate_variables(model_schema)
    # check to see if the model you are querying for is already downloaded
    if not Model.validate_model_exists (model_path):
        Model.generate_path(path_to_model)
        download = Model.download_model(model_schema, path_to_model, model_path)
        # if no, try downloading the model: if it succeeds, load that model and write to
        # the log about the model change.
        if download:
            MODEL = Model.load_model(model_path)
            MODEL_NAME = model_schema['registry']
            log.info("Model %s loaded", model_name)
            return 0
        # If it fails, write to the log about the failure and keep the currently loaded model
        log.error("Failed loading %s Model, download a different one", model_name)
        return 0
    # if yes, load that model and write to the log about the model change.
    MODEL_NAME = model_schema['registry']
    MODEL = Model.load_model(model_path)
    log.info("Model %s loaded", model_name)
    return 0


@app.route("/logs", methods=["GET"])
def logs():
    """Reads data from the log file and returns them as the response"""
    # read the log file specified and return the data
    with open(LOG_FILE, encoding="utf-8") as file:
        response = file.readlines()
    return jsonify(response)  # response must be json serializable!


@app.route("/download_registry_model", methods=["POST"])
def download_registry_model():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/download_registry_model
    The comet API key should be retrieved from the ${COMET_API_KEY} environment variable.
    Recommend (but not required) json with the schema:
        {
            workspace: (required),
            model: (required),
            version: (required),
            ... (other fields if needed) ...
        }
    """
    global MODEL
    global MODEL_NAME
    # Get POST json data
    json = request.get_json()
    log.info(json)
    path_to_model, model_path, model_name = Model.generate_variables(json)
    # check to see if the model you are querying for is already downloaded
    if not Model.validate_model_exists (model_path):
        Model.generate_path(path_to_model)
        # if no, try downloading the model: if it succeeds, load that model and write to
        # the log about the model change.
        download = Model.download_model(json, path_to_model, model_path)
        if download:
            MODEL = Model.load_model(model_path)
            MODEL_NAME = json['registry']
            log.info("Model %s loaded", model_name)
            response = f"Model {model_name} loaded"
            return make_response(jsonify(response)), 200
        # If it fails, write to the log about the failure and keep the currently loaded model
        response = f"Failed loading {model_name} Model. Keep working with the previous one"
        return make_response(jsonify(response)), 400
    # if yes, load that model and write to the log about the model change.
    MODEL = Model.load_model(model_path)
    MODEL_NAME = json['registry']
    log.info("No need to download model %s", model_name)
    log.info("Model %s loaded", model_name)
    response = f"Model {model_name} loaded"
    return make_response(jsonify(response)), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/predict
    Returns predictions
    """
    log.info("Predicting using model %s", MODEL_NAME)
    json = request.get_json()
    dataframe = pd.DataFrame(json)
    dataframe = Daraframe.delete_columns(dataframe, ['gameId','IsGoal'])
    if MODEL_NAME == 'best-nn':
        dataframe = Daraframe.delete_columns(dataframe, ["time","period"])
    elif MODEL_NAME == 'log-reg-basemodel-angle':
        dataframe = dataframe.loc[:, ['shotAngle']]
    elif MODEL_NAME == 'log-reg-basemodel-distance':
        dataframe = dataframe.loc[:, ['shotDistance']]
    elif MODEL_NAME == 'log-reg-basemodel-distance-angle':
        dataframe = dataframe.loc[:, ['shotDistance', 'shotAngle']]
    try:
        response = MODEL.predict(dataframe)
        if MODEL_NAME == 'best-nn':
            response = response.ravel()
        return make_response(jsonify({"response":response.tolist()})), 200
    except Exception as excep:
        log.info("Failed processing prediction, %s", excep)
        return make_response(jsonify({"response":"Error predicting"})), 400

# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=30002)
