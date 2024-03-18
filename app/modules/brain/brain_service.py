from app.config.settings import ChatwebSettings
from app.util.chatweb_client import ChatwebClient

chatwebClient = ChatwebClient()

class BrainService:
    def __init__(self):
        self.server_url  = ChatwebSettings().server_url

    def get_brain_by_id(self, brain_id: str):
        api_url = "/api/v1/brain/{}".format(brain_id)
        response = chatwebClient.get(api_url)
        return response.json().get("data")