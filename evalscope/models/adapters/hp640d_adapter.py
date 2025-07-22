# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, List, Union

from ..custom import HP640DModel
from .base_adapter import BaseModelAdapter


class HP640DAdapter(BaseModelAdapter):
    """
    HP640D Custom Model Adapter for EvalScope
    
    This adapter integrates LynLLM HP640D inference framework with EvalScope evaluation framework.
    It supports all HP640D parameters and provides OpenAI-compatible output format.
    """
    
    def __init__(self, hp640d_model: HP640DModel, **kwargs):
        """
        Initialize HP640D model adapter.
        
        Args:
            hp640d_model: The HP640D model instance.
            **kwargs: Other args.
        """
        self.hp640d_model = hp640d_model
        super(HP640DAdapter, self).__init__(model=hp640d_model)

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
              'model': 'hp640d-model',   # should be model_id
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

        return self.hp640d_model.predict(prompts=in_prompts, origin_inputs=inputs, **kwargs) 