"""
This file contains a wrapper for Huggingface models, implementing various methods used in downstream tasks.
It includes the HuggingfaceModel class that extends the functionality of the WhiteBoxModelBase class.
"""

from .model_base import WhiteBoxModelBase
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
import torch
from typing import Optional, Dict, List, Any

class NewModel(WhiteBoxModelBase):
    """
    HuggingfaceModel is a wrapper for Huggingface's transformers models.
    It extends the WhiteBoxModelBase class and provides additional functionality specifically
    for handling conversation generation tasks with various models.
    This class supports custom conversation templates and formatting,
    and offers configurable options for generation.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        model_name: str,
        generation_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initializes the HuggingfaceModel with a specified model, tokenizer, and generation configuration.

        :param Any model: A huggingface model.
        :param Any tokenizer: A huggingface tokenizer.
        :param str model_name: The name of the model being used. Refer to
            `FastChat conversation.py <https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py>`_
            for possible options and templates.
        :param Optional[Dict[str, Any]] generation_config: A dictionary containing configuration settings for text generation.
            If None, a default configuration is used.
        """
        super().__init__(model, tokenizer)
        self.model_name = model_name
        if generation_config is None:
            generation_config = {}
        self.generation_config = generation_config

    def generate(self, messages, input_field_name='input_ids', clear_old_history=True, **kwargs):
        r"""
        Generates a response for the given messages within a single conversation.

        :param list[str]|str messages: The text input by the user. Can be a list of messages or a single message.
        :param str input_field_name: The parameter name for the input message in the model's generation function.
        :param bool clear_old_history: If True, clears the conversation history before generating a response.
        :param dict kwargs: Optional parameters for the model's generation function, such as 'temperature' and 'top_p'.
        :return: A string representing the pure response from the model, containing only the text of the response.
        """
        if isinstance(messages, str):
            messages = [messages]
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": messages[0]}
            ]

            input_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(self.model.device)

            terminators = [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

        else:
            conversation_messages = []
            for index, message in enumerate(messages):
                if index % 2 == 0:
                    conversation_messages.append({"role": "user", "content": message})
                else:
                    conversation_messages.append({"role": "system", "content": message})

            input_ids = self.tokenizer.apply_chat_template(
                conversation_messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(self.model.device)

            terminators = [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]


        outputs = self.model.generate(
            input_ids,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        response = self.tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
        print(response)

        return response

    def __call__(self, *args, **kwargs):
        r"""
        Allows the HuggingfaceModel instance to be called like a function, which internally calls the model's
        __call__ method.

        :return: The output from the model's __call__ method.
        """
        return self.model(*args, **kwargs)

    def tokenize(self, *args, **kwargs):
        r"""
        Tokenizes the input using the model's tokenizer.

        :return: The tokenized output.
        """
        return self.tokenizer.tokenize(*args, **kwargs)

    def batch_encode(self, *args, **kwargs):
        return self.tokenizer(*args, **kwargs)

    def batch_decode(self, *args, **kwargs)-> List[str]:
        return self.tokenizer.batch_decode(*args, **kwargs)


    @property
    def device(self):
        return self.model.device

    @property
    def dtype(self):
        return self.model.dtype

    @property
    def bos_token_id(self):
        return self.tokenizer.bos_token_id

    @property
    def eos_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id

    @property
    #@functools.cache  # 记忆结果，避免重复计算。不影响子类的重载。
    def embed_layer(self) -> Optional[torch.nn.Embedding]:
        """
        Retrieve the embedding layer object of the model.

        This method provides two commonly used approaches to search for the embedding layer in Hugging Face models.
        If these methods are not effective, users should consider manually overriding this method.

        Returns:
            torch.nn.Embedding or None: The embedding layer object if found, otherwise None.
        """
        for module in self.model.modules():
            if isinstance(module, torch.nn.Embedding):
                return module

        for module in self.model.modules():
            if all(hasattr(module, attr) for attr in ['bos_token_id', 'eos_token_id', 'encode', 'decode', 'tokenize']):
                return module
        return None

    @property
    #@functools.cache
    def vocab_size(self) -> int:
        """
        Get the vocabulary size.

        This method provides two commonly used approaches for obtaining the vocabulary size in Hugging Face models.
        If these methods are not effective, users should consider manually overriding this method.

        Returns:
            int: The size of the vocabulary.
        """
        if hasattr(self, 'config') and hasattr(self.config, 'vocab_size'):
            return self.config.vocab_size

        return self.embed_layer.weight.size(0)


def from_pretrained(model_name_or_path: str, model_name: str, tokenizer_name_or_path: Optional[str] = None,
                    dtype: Optional[torch.dtype] = None, **generation_config: Dict[str, Any]) -> NewModel:
    """
    Imports a Hugging Face model and tokenizer with a single function call.

    :param str model_name_or_path: The identifier or path for the pre-trained model.
    :param str model_name: The name of the model, used for generating conversation template.
    :param Optional[str] tokenizer_name_or_path: The identifier or path for the pre-trained tokenizer.
        Defaults to `model_name_or_path` if not specified separately.
    :param Optional[torch.dtype] dtype: The data type to which the model should be cast.
        Defaults to None.
    :param generation_config: Additional configuration options for model generation.
    :type generation_config: dict

    :return NewModel: An instance of the NewModel class containing the imported model and tokenizer.

    .. note::
        The model is loaded for evaluation by default. If `dtype` is specified, the model is cast to the specified data type.
        The `tokenizer.padding_side` is set to 'right' if not already specified.
        If the tokenizer has no specified pad token, it is set to the EOS token, and the model's config is updated accordingly.

    **Example**

    .. code-block:: python

        model = from_pretrained('bert-base-uncased', 'bert-model', dtype=torch.float32, max_length=512)
    """
    if dtype is None:
        dtype = 'auto'

    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
        )

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map='auto', trust_remote_code=True, low_cpu_mem_usage=True, torch_dtype=dtype, quantization_config=nf4_config).eval()
    if tokenizer_name_or_path is None:
        tokenizer_name_or_path = model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)

    if tokenizer.padding_side is None:
        tokenizer.padding_side = 'right'

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id

    print(model)
    print(tokenizer)
    print("------------------------")

    return NewModel(model, tokenizer, model_name=model_name, generation_config=generation_config)
