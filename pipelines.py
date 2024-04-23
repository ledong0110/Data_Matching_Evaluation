import ast
# import torch

from tqdm import tqdm
from generation_config import GenerationConfig
from wrappers import (
    TGI,
    GPT
)
from utils import *


class EvalPipeline:
    def __init__(self, task, config):
        self.task = task
        extract_task = self.task.split("_")[0]
        
        # Load pipelines
      
        if "gpt-3.5-turbo"  in config.model_name or "gpt-4" in config.model_name:
            # Load model
            self.infer_pipeline = GPT(model_name=config.model_name, model_config=config, generation_config=GenerationConfig[config.model_name])
            
        # elif "gemini-pro" in config.model_name:
        #     self.infer_pipeline = GeminiPipeline(model=config.model_name, generation_config=GenerationConfig[extract_task])
        elif config.tgi != "0":
            self.infer_pipeline = TGI(model_name=config.model_name, model_config=config, generation_config=GenerationConfig[config.model_name])
        else:
            raise NotImplementedError

        self.prompting_strategy = 0
        self.few_shot = False
        self.random_mtpc = False
        self.cot = False

    def __call__(self, ds_wrapper, ds_loader, saving_fn, start_idx=0):
        task = self.task.split("_")[0]

        if task == "entity-matching":
            return self.__entity_matching(
                ds_wrapper, ds_loader, saving_fn, start_idx
            )
        else:
            raise NotImplementedError

   
    def __entity_matching(
        self, ds_wrapper, ds_loader, saving_fn, start_idx=0
    ):
        predictions = []
        references = []

        idx = 0
        original_few_shot = ""
        selected_sample = []
        def convert_row_to_text(row):
            lfields = ds_wrapper.left_table
            rfields = ds_wrapper.right_table
            label = ds_wrapper.label
           
            col_names = lfields.keys()
            l_key_and_val_concat = []
            r_key_and_val_concat = []
            for i in col_names:
                if row[lfields[i]]:
                    l_key_and_val_concat.append(f"{i}: {row[lfields[i]]}")
                else:
                    l_key_and_val_concat.append(f"{i}: <empty>")
                
                if row[rfields[i]]:
                    r_key_and_val_concat.append(f"{i}: {row[rfields[i]]}")
                else:
                    r_key_and_val_concat.append(f"{i}: <empty>")
                    
            return " ".join(l_key_and_val_concat), " ".join(r_key_and_val_concat), row[label]
        
        def convert_to_conv(rec, flag=False):
            conv = []
            matchh = ["Yes", "No"]
            content = f"As an expert in entity matching problem. Do the following two entity refer to the same real-world entity ?```.\nENTITY 1: ```{rec[0]}```\nENTITY 2: ```{rec[1]}```\nPlease put your answer in json format ```{{ \"answer\": `Your answer is 'Yes' if it is same, 'No' if it is not same}}"
            conv.append({"role": "user", "content": content})
            if not flag:
                conv.append({"role": "assistant", "content": f"{{\"answer\": {matchh[rec[2]]}}}"})
            return conv
        original_few_shot = []   
        if self.few_shot:
            
            def format_original_fewshot(recs):
                conv = []
                for rec in recs:
                    conv.extend(convert_to_conv(rec))
                    # content = f"Do the following two entity refer to the same real-world entity ? Please put your answer in json ```{{ \"answer\": `Your answer is 1 if it is same, 0 if it is not same}}.\nENTITY 1:\n{rec[0]}\nENTITY 2:\n{rec[1]}"
                    # conv.append({"role": "user", "content": content})
                    # conv.append({"role": "assistant", "content": f"{{ \"answer\": {rec[2]} }}"})
                
                return conv

           
            classes = unique(ds_wrapper.dataset_training[ds_wrapper.label])
            selected_sample = []
            for cl in classes:
                cl_samples = ds_wrapper.dataset_training.filter(
                    lambda r: r[ds_wrapper.label] == cl
                )
                selected_sample.append(
                    cl_samples[random.randint(0, len(cl_samples))])

            original_few_shot = format_original_fewshot(list(map(convert_row_to_text, selected_sample)))

        # print(ds_wrapper.dataset_testing[0])
        for batch in tqdm(ds_wrapper.dataset_testing):
            if idx < start_idx:
                idx += 1
                continue
            
            prompts = list(map(convert_row_to_text, [batch]))
            prompts = [
                original_few_shot + convert_to_conv(p, flag=True)
                for p in prompts
            ]

            results = self.infer_pipeline.generate(
                prompts)
            predictions.extend(results)
            references.extend([x for x in [batch[ds_wrapper.label]]])
           
            idx += 1
            if idx % 100 == 0:
                print(f"Saving results of {idx} batches")
                generations = {
                    "predictions": predictions,
                    "references": references,
                    "fewshot": selected_sample,
                }

                saving_fn(generations)

        generations = {
            "predictions": predictions,
            "references": references,
            "fewshot": selected_sample,
        }
        saving_fn(generations)

    def run(
        self,
        ds_wrapper,
        ds_loader,
        saving_fn,
        start_idx=0,
        few_shot=False,
    ):
        # with torch.no_grad():
        self.few_shot = few_shot
        results = self(ds_wrapper, ds_loader, saving_fn, start_idx)
        return results
