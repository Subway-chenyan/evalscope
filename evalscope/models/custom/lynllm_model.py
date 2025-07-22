# Copyright (c) Alibaba, Inc. and its affiliates.
import time
import logging
from typing import List, Dict, Any, Union, Optional

from .custom_model import CustomModel

# Import LynLLM inference framework
try:
    from .lynxi_llm import LLMApi
    LYNLLM_AVAILABLE = True
except ImportError:
    LYNLLM_AVAILABLE = False
    LLMApi = None

logger = logging.getLogger(__name__)


class LynLLMModel(CustomModel):
    """
    LynLLM Custom Model for EvalScope
    
    This model integrates LynLLM inference framework with EvalScope evaluation framework.
    It supports all LynLLM parameters and provides OpenAI-compatible output format.
    """
    
    def __init__(self, config: dict = None, **kwargs):
        """
        Initialize LynLLM model.
        
        Args:
            config (dict): Configuration dict containing model parameters
                Required keys:
                - model_path: Path to the LynLLM model
                - device_list: Device list for inference (str or list)
                Optional keys:
                - do_sample: Whether to use sampling (default: True)
                - show_speed: Whether to show inference speed (default: True)
                - Other LynLLM parameters
            **kwargs: Additional keyword arguments
        """
        if config is None:
            config = {}
        
        super(LynLLMModel, self).__init__(config=config, **kwargs)
        
        # Check if LynLLM is available
        if not LYNLLM_AVAILABLE:
            raise ImportError(
                "LynLLM framework is required but not installed. "
                "Please install it first using: pip install LynLLM"
            )
        
        # Extract required parameters
        self.model_path = config.get('model_path')
        if not self.model_path:
            raise ValueError("model_path is required in config")
        
        self.device_list = config.get('device_list', "0")
        self.do_sample = config.get('do_sample', True)
        self.show_speed = config.get('show_speed', True)
        
        # Extract other LynLLM parameters
        self.lynllm_kwargs = {k: v for k, v in config.items() 
                              if k not in ['model_path', 'device_list', 'do_sample', 'show_speed', 'model_id']}
        
        # Initialize model
        self.model = None
        self._init_model()
        
        logger.info(f"LynLLM model initialized with path: {self.model_path}")
        logger.info(f"Device list: {self.device_list}")
        
    def _init_model(self):
        """Initialize the LynLLM model instance."""
        try:
            self.model = LLMApi(
                model_path=self.model_path,
                device_list=self.device_list,
                do_sample=self.do_sample,
                show_speed=self.show_speed,
                # **self.lynllm_kwargs
            )
            logger.info("LynLLM model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LynLLM model: {str(e)}")
            raise
    
    def _extract_prompt_from_input(self, input_item: Union[str, dict, list]) -> str:
        """
        Extract prompt string from various input formats.
        
        Args:
            input_item: Input in various formats:
                - str: Direct prompt string
                - dict: {'data': [prompt], 'messages': [...], 'system_prompt': '...'}
                - list: List of prompt strings or tuples
                
        Returns:
            str: Extracted prompt string
        """
        if isinstance(input_item, str):
            return input_item
        elif isinstance(input_item, dict):
            # Handle messages format (chat completion style)
            if input_item.get('messages'):
                messages = input_item['messages']
                prompt_parts = []
                for msg in messages:
                    role = msg.get('role', '')
                    content = msg.get('content', '')
                    if role == 'system':
                        prompt_parts.append(f"System: {content}")
                    elif role == 'user':
                        prompt_parts.append(f"User: {content}")
                    elif role == 'assistant':
                        prompt_parts.append(f"Assistant: {content}")
                return '\n'.join(prompt_parts)
            
            # Handle data format
            elif input_item.get('data'):
                data = input_item['data']
                if isinstance(data, list) and len(data) > 0:
                    if isinstance(data[0], tuple):
                        # For truthful_qa and hellaswag formats
                        return '\n'.join(''.join(item) for item in data)
                    else:
                        return str(data[0])
                return str(data)
            
            # Handle direct prompt
            else:
                return str(input_item)
                
        elif isinstance(input_item, list):
            # Join list items
            return '\n'.join(str(item) for item in input_item)
        else:
            return str(input_item)
    
    def make_request_messages(self, input_item: dict) -> list:
        """
        Make request messages for compatibility.
        This method maintains compatibility with existing EvalScope structure.
        """
        if input_item.get('messages', None):
            return input_item['messages']
        
        data: list = input_item['data']
        if isinstance(data[0], tuple):
            # for truthful_qa and hellaswag
            query = '\n'.join(''.join(item) for item in data)
            system_prompt = input_item.get('system_prompt', None)
        else:
            query = data[0]
            system_prompt = input_item.get('system_prompt', None)
            
        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': query})
        
        return messages
    
    def predict(self, prompts: List[Union[str, dict, list]], **kwargs) -> List[Dict[str, Any]]:
        """
        Model prediction function for batch inputs.
        
        Args:
            prompts: List of input prompts in various formats
            **kwargs: Additional arguments including origin_inputs, infer_cfg
            
        Returns:
            List[Dict]: List of OpenAI API compatible responses
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Please check initialization.")
        
        # Get original inputs for better context handling
        original_inputs = kwargs.get('origin_inputs', prompts)
        infer_cfg = kwargs.get('infer_cfg', {})
        
        logger.debug(f"Processing {len(prompts)} prompts")
        if infer_cfg:
            logger.debug(f"Inference config: {infer_cfg}")
        
        responses = []
        
        for i, (prompt_data, original_input) in enumerate(zip(prompts, original_inputs)):
            try:
                # Extract prompt string from input
                if isinstance(original_input, (dict, list)):
                    prompt_text = self._extract_prompt_from_input(original_input)
                else:
                    prompt_text = self._extract_prompt_from_input(prompt_data)
                
                logger.debug(f"Prompt {i+1}: {prompt_text[:100]}...")
                
                # Generate response using LynLLM
                infer_cfg.pop('max_new_tokens', None)
                start_time = time.time()
                result = self.model.generate(prompt_text, **infer_cfg)
                inference_time = time.time() - start_time
                
                # Extract content from LynLLM result
                if isinstance(result, dict):
                    content = result.get('content', str(result))
                    speed_info = {k: v for k, v in result.items() if k != 'content'}
                else:
                    content = str(result)
                    speed_info = {}
                
                logger.debug(f"Response {i+1}: {content[:100]}...")
                if speed_info:
                    logger.debug(f"Speed info: {speed_info}")
                
                # Create OpenAI API compatible response
                response = {
                    'choices': [{
                        'index': 0,
                        'message': {
                            'content': content,
                            'role': 'assistant'
                        }
                    }],
                    'created': int(time.time()),
                    'model': self.config.get('model_id', 'lynllm-model'),
                    'object': 'chat.completion',
                    'usage': {
                        'completion_tokens': len(content.split()) if content else 0,
                        'prompt_tokens': len(prompt_text.split()) if prompt_text else 0,
                        'total_tokens': len(content.split()) + len(prompt_text.split()) if content and prompt_text else 0,
                        'inference_time': inference_time
                    }
                }
                
                # Add speed information if available
                if speed_info:
                    response['usage'].update(speed_info)
                
                responses.append(response)
                
            except Exception as e:
                logger.error(f"Error processing prompt {i+1}: {str(e)}")
                # Return error response
                error_response = {
                    'choices': [{
                        'index': 0,
                        'message': {
                            'content': f"Error during inference: {str(e)}",
                            'role': 'assistant'
                        }
                    }],
                    'created': int(time.time()),
                    'model': self.config.get('model_id', 'lynllm-model'),
                    'object': 'chat.completion',
                    'usage': {
                        'completion_tokens': 0,
                        'prompt_tokens': 0,
                        'total_tokens': 0
                    },
                    'error': str(e)
                }
                responses.append(error_response)
        
        logger.info(f"Completed inference for {len(responses)} prompts")
        return responses
    
    def __del__(self):
        """Cleanup model resources."""
        if hasattr(self, 'model') and self.model is not None:
            try:
                self.model.terminate()
                logger.info("LynLLM model terminated successfully")
            except Exception as e:
                logger.warning(f"Error during model cleanup: {str(e)}")
