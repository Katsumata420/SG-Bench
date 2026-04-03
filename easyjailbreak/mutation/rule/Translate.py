from typing import List

import requests
from easyjailbreak.mutation import MutationBase
from easyjailbreak.datasets import Instance

class Translate(MutationBase):
    """
    Translate is a class for translating the query to another language.
    """
    def __init__(self, attr_name='query', language='en'):
        self.attr_name = attr_name
        self.language = language
        languages_supported = {
            'en': 'English',
            'zh-CN': 'Chinese',
            'it': 'Italian',
            'vi': 'Vietnamese',
            'ar': 'Arabic',
            'ko': 'Korean',
            'th': 'Thai',
            'bn': 'Bengali',
            'sw': 'Swahili',
            'jv': 'Javanese',
            'ja': 'Japanese',
            'fr': 'French'
        }
        if self.language in languages_supported:
            self.lang = languages_supported[self.language]
        else:
            raise ValueError(f"Unsupported language: {self.language}")

    def _get_mutated_instance(self, instance) -> List[Instance]:
        """
        mutate the instance by translating the query to another language
        """
        mutated_results = []
        seed = getattr(instance, self.attr_name)
        new_instance = instance.copy()
        new_seed = seed
        new_instance.lang = self.lang
        setattr(new_instance, 'translated_query', new_seed)
        setattr(new_instance, 'origin_query', seed)
        setattr(new_instance, 'query', new_seed)
        mutated_results.append(new_instance)
        return mutated_results

    def translate(self, text, src_lang='auto'):
        """
        translate the text to another language
        """
        googleapis_url = 'https://translate.googleapis.com/translate_a/single'
        url = '%s?client=gtx&sl=%s&tl=%s&dt=t&q=%s' % (googleapis_url,src_lang,self.language,text)
        data = requests.get(url).json()
        res = ''.join([s[0] for s in data[0]])
        return res



