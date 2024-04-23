import random
from datasets import load_dataset



def eval_keys(keys):
    def eval_x(x):
        if isinstance(keys, str):
            x[keys] = eval(x[keys])
        elif isinstance(keys, list):
            for key in keys:
                x[key] = eval(x[key])
        return x

    return eval_x


class DatasetWrapper:
    def __init__(self, dataset_name, prompting_strategy=0, fewshots=None) -> None:
        self.dataset_name = dataset_name
        self.prompting_strategy = prompting_strategy
        self.fewshots = fewshots
        self.dataset_training = None
        self.dataset_testing = None

        self.get_dataset_config()
        # self.get_prompt()

    # def get_prompt(self):
    #     if self.prompting_strategy not in [0, 1, 2, 3]:
    #         raise ValueError("Prompting strategy is not supported")
    #     task = self.task.split("_")[0]
    #     self.prompt = PROMPT_TEMPLATE[task][self.prompting_strategy]
    #     if task in CALIBRATION_INSTRUCTION:
    #         self.calibration_prompt = CALIBRATION_INSTRUCTION[task][
    #             self.prompting_strategy
    #         ]
    #     else:
    #         self.calibration_prompt = None

    def get_dataset_config(self):
        # Question Answering
        if self.dataset_name == "abt_buy":
            self.task = "entity-matching"
            self.dataset_testing = load_dataset(
                "csv",
                data_files="datasets/abt_buy/new_test.csv",
                split="train",
            )
            self.dataset_training = load_dataset(
                "csv",
                data_files="datasets/abt_buy/new_train.csv",
                split="train",
            )
            self.left_table = {"TITLE": "left_name", "DESCRIPTION": "left_description"}
            self.right_table = {"TITLE": "right_name", "DESCRIPTION": "right_description"}
            self.label = "label"

        elif self.dataset_name == "walmart_amazon":
            self.task = "entity-matching"
            self.dataset_testing = load_dataset(
                "csv",
                data_files="datasets/walmart_amazon/new_test.csv",
                split="train",
            )
            self.dataset_training = load_dataset(
                "csv",
                data_files="datasets/walmart_amazon/new_train.csv",
                split="train",
            )
            self.left_table = {"TITLE": "left_title", "CATEGORY": "left_category", "BRAND": "left_brand"}
            self.right_table = {"TITLE": "right_title", "CATEGORY": "right_category", "BRAND": "right_brand"}
            self.label = "label"
        
        elif self.dataset_name == "dirty_walrmart_amazon":
            self.task = "entity-matching"
            self.dataset_testing = load_dataset(
                "csv",
                data_files="datasets/dirty_walmart_amazon/new_test.csv",
                split="train",
            )
            self.dataset_training = load_dataset(
                "csv",
                data_files="datasets/dirty_walmart_amazon/new_train.csv",
                split="train",
            )
            self.left_table = {"TITLE": "left_title", "CATEGORY": "left_category", "BRAND": "left_brand"}
            self.right_table = {"TITLE": "right_title", "CATEGORY": "right_category", "BRAND": "right_brand"}
            self.label = "label"
        
        else:
            raise ValueError("Dataset is not supported")

    def get_dataset_testing(self):
        if self.dataset_testing is None:
            raise ValueError("Dataset testing is not available")
        return self.dataset_testing

    def get_dataset_training(self):
        if self.dataset_training is None:
            raise ValueError("Dataset training is not available")
        return self.dataset_training
