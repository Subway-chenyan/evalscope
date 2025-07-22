# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, List, Union

from ..custom import LynLLMModel
from .base_adapter import BaseModelAdapter


class LynLLMAdapter(BaseModelAdapter):
    """
    LynLLM Custom Model Adapter for EvalScope
    
    This adapter integrates LynLLM inference framework with EvalScope evaluation framework.
    It supports all LynLLM parameters and provides OpenAI-compatible output format.
    """
    
    def __init__(self, lynllm_model: LynLLMModel, **kwargs):
        """
        Initialize LynLLM model adapter.
        
        Args:
            lynllm_model: The LynLLM model instance.
            **kwargs: Other args.
        """
        self.lynllm_model = lynllm_model
        super(LynLLMAdapter, self).__init__(model=lynllm_model)

    def predict(self, inputs: List[Union[str, dict, list]], **kwargs) -> List[Dict[str, Any]]:
        """
        Model prediction func.

        Args:
            inputs (List[Union[str, dict, list]]): The input data. Depending on the specific model.
                str: 'xxx'
                dict: {'data': [full_prompt]}
                list: ['xxx', 'yyy', 'zzz']
            **kwargs: kwargs

        Returns:
            res (dict): The model prediction results. Format:
            {
              'choices': [
                {
                  'index': 0,
                  'message': {
                    'content': 'xxx',
                    'role': 'assistant'
                  }
                }
              ],
              'created': 1677664795,
              'model': 'gpt-3.5-turbo-0613',   # should be model_id
              'object': 'chat.completion',
              'usage': {
                'completion_tokens': 17,
                'prompt_tokens': 57,
                'total_tokens': 74
              }
            }
        """
        in_prompts = []

        # Note: here we assume the inputs are all prompts for the benchmark.
        for input_prompt in inputs:
            if isinstance(input_prompt, str):
                in_prompts.append(input_prompt)
            elif isinstance(input_prompt, dict):
                # TODO: to be supported for continuation list like truthful_qa
                in_prompts.append(input_prompt['data'][0])
            elif isinstance(input_prompt, list):
                in_prompts.append('\n'.join(input_prompt))
            else:
                raise TypeError(f'Unsupported inputs type: {type(input_prompt)}')

        return self.lynllm_model.predict(prompts=in_prompts, origin_inputs=inputs, **kwargs)
