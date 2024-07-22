from app.config.settings import ChatwebSettings
from app.modules.brain.brain_model import BrainModel, BrainType
from app.util.chatweb_client import ChatwebClient

chatwebClient = ChatwebClient()

class BrainService:
    def __init__(self):
        self.server_url  = ChatwebSettings().server_url

    def get_brain_by_id(self, brain_id: str):
        api_url = "/api/v1/brain/{}".format(brain_id)
        response = chatwebClient.get(api_url)
        brain_to_use = response.json().get("data")
        return BrainModel(**{
            "model": brain_to_use["model"],
            "brain_type": BrainType[brain_to_use["brainType"]],
            "brain_id": brain_to_use["brainId"]
        })
    def get_all_brains(self, user_id: str):
        response = chatwebClient.post("/api/v1/brain/list", {"userId": user_id})
        return response

    def find_brain_from_question(self, brain_id: str, question: str, history) -> BrainModel:
        brain_id_to_use = brain_id
        brain_to_use = None
        question = question

        # 1.0历史记录查找
        if history and not brain_id_to_use:
            question = history[0]["userMessage"]
            brain_id_to_use = history[0]["brainId"]
            brain_to_use = self.get_brain_by_id(brain_id_to_use)
        # 2.0历史记录查找
        if brain_id_to_use and not brain_to_use:
            brain_to_use = self.get_brain_by_id(brain_id_to_use)
        return brain_to_use
