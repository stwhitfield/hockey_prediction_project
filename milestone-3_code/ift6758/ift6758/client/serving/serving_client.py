"""DOCUMENTATION"""
import logging
import os
import requests
import pandas as pd

LOG_FILE = os.environ.get("SERVING_LOGS", "flask.log")
APP = os.environ.get("APP")

comet_config = {
    "workspace": "ift6758-project",
    "registry": "best-svm-prob",
    "version": "1.0.0"
}

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s : %(message)s'
)
logger = logging.getLogger(__name__)

class ServingClient:
    """DOCUMENTATION"""
    def __init__(self, ip_address: str = APP, port: int = 30001, features=None):
        logger.info('___________Init ServingClient')
        self.base_url = f"http://{ip_address}:{port}"

        if features is None:
            features = ["distance"]
        self.features = features
    
    def predict(self, X: dict) -> pd.DataFrame:
        """
        Formats the inputs into an appropriate payload for a POST request, and queries the
        prediction service. Retrieves the response from the server, and processes it back into a
        dataframe that corresponds index-wise to the input dataframe.
        
        Args:
            X (Dataframe): Input dataframe to submit to the prediction service.
        """
        response = requests.post(
            f"{self.base_url}/predict", timeout = 10,
            json = X
        )
        if response.status_code == 400:
            logger.info("Failed requesting prediction: %s", response.content)
            return 0
        logger.info("Prediction process finished")
        return response.json()["response"]

    def logs(self) -> dict:
        """Get server logs"""
        logger.info("Requesting logs")
        log_request = requests.get(
            f"{self.base_url}/logs", timeout = 5
        )
        logger.info("Logs fetched")
        return log_request.json()

    def download_registry_model(self, workspace: str, model: str, version: str) -> dict:
        """
        Triggers a "model swap" in the service; the workspace, model, and model version are
        specified and the service looks for this model in the model registry and tries to
        download it. 
        See more here:
            https://www.comet.ml/docs/python-sdk/API/#apidownload_registry_model
        
        Args:
            workspace (str): The Comet ML workspace
            model (str): The model in the Comet ML registry to download
            version (str): The model version to download
        """
        model_schema = {
            "workspace": workspace,
            "registry": model,
            "version": version
        }
        download_request = requests.post(
            f"{self.base_url}/download_registry_model", timeout = 60,
            json = model_schema
        )
        if download_request.status_code != 200:
            logger.info("Failed loadding %s model", model)
            return download_request.json(), download_request.status_code
        logger.info("%s model loadded successfully", model)
        return download_request.json(), download_request.status_code
