import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

import yaml


def load_config(config_name):
    with open(f'./DTX/{config_name}') as file:
        yaml_data = yaml.safe_load(file)
    return yaml_data


config = load_config("config.yaml")


class Evaluation:
    def __init__(
            self,
            model_id=config['model_id'],
            device=config['device'],
            dtype=torch.float16,
            bnb_4bit_quant_type=config['bnb_4bit_quant_type'],
            bnb_4bit_use_double_quant=config['bnb_4bit_use_double_quant'],
            load_in_4bit=config['load_in_4bit']
    ):
        """
        Initialize the Evaluation class with model parameters and configurations.

        :param model_id: str, the ID of the model to load
        :param device: str, the device to use for computations (default is "cuda")
        :param dtype: torch.dtype, the data type for computations (default is torch.float16)
        :param bnb_4bit_quant_type: str, the quantization type for BitsAndBytes (default is "nf4")
        :param bnb_4bit_use_double_quant: bool, whether to use double quantization (default is True)
        :param load_in_4bit: bool, whether to load the model in 4-bit mode (default is True)
        """
        self.device = device
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device,
            quantization_config=self.bnb_config,
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)


    def make_prediction(self, user_input):
        """
        make predictions based on user input
        :param user_input: str
        :return: list
        """
        input_ids = self.tokenizer.apply_chat_template(self._apply_chat_template(user_input), return_tensors="pt").to(self.device)
        output = self.model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
        prompt_len = input_ids.shape[-1]
        return self.tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True).split()


    def _apply_chat_template(self, user_input, model_response=None):
        """
        Apply a chat template to the user input.

        :param user_input: str, the input text from the user
        :param model_response: str or None, the response from the model (default is None)
        :return: list, the chat template with roles and content
        """
        if model_response:
            return [
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": model_response}
            ]

        return [
            {"role": "user", "content": user_input}
        ]


