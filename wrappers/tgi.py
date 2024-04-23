from .base import Base
import requests
from utils import apply_chat_template
from chat_template import ChatTemplate

class TGIModel:
    def __init__(self, model_config):
        self.api_endpoint = model_config.tgi
    
    def generate(self, prompt, config):
        # sending post request and saving response as response object
        # print(prompt)
        # print(prompt)
        data = {
            "inputs": prompt,
            "parameters": config
        }
        r = requests.post(url=self.api_endpoint, json=data)
        # print(r.status_code)
        # # extracting response text
        # print(r.json())
        answer = r.json()['generated_text']
        return answer
        

class TGI(Base):
    def __init__(self, model_name, model_config, generation_config):
        self.model_name = model_name
        self.generation_config = generation_config
        self.model = TGIModel(model_config)
        self.template_chat = ChatTemplate[self.model_name]
    
        
        
    def generate(self, inputs):
        out = []
        for prompt in inputs:
            processed_prompt = apply_chat_template(prompt, chat_template=self.template_chat)
            res = self.model.generate(processed_prompt, self.generation_config)
            out.append(res)
        
        return out