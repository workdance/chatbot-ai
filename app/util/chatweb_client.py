from typing import T

import requests

from app.config.settings import ChatwebSettings


class ChatwebClient:
    def __init__(self):
        self.server_url = ChatwebSettings().server_url

    def get(self, service_id: str, data=None):
        response = requests.get(self.server_url+service_id, params=data)
        if response.status_code != 200:
            response.raise_for_status()
        else:
            return response

    def post(self, service_id:str, data: any):
        headers = {
            'Content-Type': 'application/json'
        }
        response = requests.post(self.server_url+service_id, json=data, headers=headers)
        if response.status_code != 200:
            response.raise_for_status()
        else:
            return response