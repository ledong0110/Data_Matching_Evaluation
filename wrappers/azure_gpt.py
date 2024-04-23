from .base import Base
from openai import AzureOpenAI
import os

class GPT(Base):
    def __init__(self, model_name, model_config, generation_config):
        self.model_name = model_name
        self.generation_config = generation_config
        self.model = AzureOpenAI(
            azure_endpoint = os.getenv("AZURE_ENDPOINT"), 
            api_key=os.getenv("AZURE_KEY"),  
            api_version=os.getenv("AZURE_VERSION")
            )
        # self.template_chat = ChatTemplate[self.model_name]
    
        
        
    def generate(self, inputs):
        out = []
        for prompt in inputs:
            completion = self.model.chat.completions.create(
                model="testing", # model = "deployment_name"
                messages = prompt,
                temperature=self.generation_config["temperature"],
                max_tokens=self.generation_config["max_new_tokens"],
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
                logprobs=False
                )
            res = completion.choices[0].message.content
            out.append(res)
        
        return out