# Copyright (c) Alibaba, Inc. and its affiliates.
import time
import copy
import logging
from typing import List, Dict, Any, Union, Optional

from .custom_model import CustomModel

# Import LynLLM HP640D inference framework
try:
    from LynLLM.hp640d import logger
    from LynLLM.hp640d.api import (init_llm_model, run_llm_model)
    from LynLLM.serving.openai_entry.sampling_params import SamplingParams
    from LynLLM.hp640d.core.multibatch_model import MultiBatchModel
    HP640D_AVAILABLE = True
except ImportError:
    HP640D_AVAILABLE = False
    logger = None
    init_llm_model = None
    run_llm_model = None
    SamplingParams = None
    MultiBatchModel = None

logger = logging.getLogger(__name__)


class HP640DModel(CustomModel):
    """
    HP640D Custom Model for EvalScope
    
    This model integrates LynLLM HP640D inference framework with EvalScope evaluation framework.
    It supports all HP640D parameters and provides OpenAI-compatible output format.
    """
    
    def __init__(self, config: dict = None, **kwargs):
        """
        Initialize HP640D model.
        
        Args:
            config (dict): Configuration dict containing model parameters
                Required keys:
                - model_path: Path to the HP640D model
                Optional keys:
                - do_sample: Whether to use sampling (default: True)
                - max_tokens: Maximum tokens to generate (default: None)
                - ip_file: Path to file containing IP list for distributed mode
                - ray_cluster_head_ip: Master node IP for distributed mode
                - ray_cluster_head_port: Master node port for distributed mode
                - master: Whether this is a master node (default: False)
                - distributed_configs_path: Path to distributed configuration JSON
                - node_config: Path to node configuration JSON
                - Other HP640D parameters
            **kwargs: Additional keyword arguments
        """
        if config is None:
            config = {}
        
        super(HP640DModel, self).__init__(config=config, **kwargs)
        
        # Check if HP640D is available
        if not HP640D_AVAILABLE:
            raise ImportError(
                "LynLLM HP640D framework is required but not installed. "
                "Please install it first using: pip install LynLLM"
            )
        
        # Extract required parameters
        self.model_path = config.get('model_path')
        if not self.model_path:
            raise ValueError("model_path is required in config")
        
        self.do_sample = config.get('do_sample', True)
        self.max_tokens = config.get('max_tokens', None)
        
        # Extract distributed configuration parameters
        self.ip_file = config.get('ip_file', None)
        self.ray_cluster_head_ip = config.get('ray_cluster_head_ip', None)
        self.ray_cluster_head_port = config.get('ray_cluster_head_port', None)
        self.master = config.get('master', False)
        self.distributed_configs_path = config.get('distributed_configs_path', None)
        self.node_config = config.get('node_config', None)
        
        # Extract other HP640D parameters (excluding distributed config params)
        excluded_keys = ['model_path', 'do_sample', 'max_tokens', 'model_id', 
                        'ip_file', 'ray_cluster_head_ip', 'ray_cluster_head_port', 'master', 
                        'distributed_configs_path', 'node_config']
        self.hp640d_kwargs = {k: v for k, v in config.items() if k not in excluded_keys}
        
        # Initialize model
        self.model = None
        self._init_model()
        
        logger.info(f"HP640D model initialized with path: {self.model_path}")
        
    def _init_model(self):
        """Initialize the HP640D model instance."""
        try:
            # Read IP list from file if specified
            ip_list = None
            if self.ip_file is not None:
                with open(self.ip_file, "r") as f:
                    ip_list = [ip.strip() for ip in f.readlines()]
                    logger.info(f"Loaded IP list from file: {ip_list}")
            
            # Record load start time
            load_start = time.time()
            
            # Initialize model using HP640D API with distributed configuration
            if ip_list:
                # Distributed mode with IP list
                self.model = init_llm_model(
                    self.model_path,
                    ip_list=ip_list,
                    ray_cluster_head_ip=self.ray_cluster_head_ip,
                    ray_cluster_head_port=self.ray_cluster_head_port,
                    node_config_json_path=self.distributed_configs_path,
                    **self.hp640d_kwargs
                )
            else:
                # Single node or master mode
                self.model = init_llm_model(
                    self.model_path, 
                    is_master=self.master, 
                    node_config_json_path=self.node_config,
                    **self.hp640d_kwargs
                )
            
            # Log load time
            load_time = time.time() - load_start
            logger.info(f"HP640D model loaded successfully in {load_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Failed to initialize HP640D model: {str(e)}")
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
                
                # Set up generation parameters
                max_length = infer_cfg.get('max_new_tokens', self.max_tokens)
                if max_length is None:
                    max_length = self.model.max_tokens_limit - 1
                else:
                    max_length = min(max_length, self.model.max_tokens_limit - 1)
                
                sample_param = SamplingParams(max_tokens=max_length)
                
                generation_config = copy.deepcopy(self.model.generation_config)
                generation_config.do_sample = self.do_sample
                
                # Build input tokens
                input_ids = self.model.build_input(query=prompt_text)
                
                # Check input length
                status, token_count = self.model.input_overlimit(input_ids, max_length=sample_param.max_tokens)
                if status:
                    logger.warning(f"query length over limit; model support max_tokens '{self.model.max_tokens_limit}', user setting max_length '{max_length}', get tokens '{token_count}'; query: '{prompt_text}'")
                    # Return empty response for over-limit inputs
                    response = {
                        'choices': [{
                            'index': 0,
                            'message': {
                                'content': "",
                                'role': 'assistant'
                            }
                        }],
                        'created': int(time.time()),
                        'model': self.config.get('model_id', 'hp640d-model'),
                        'object': 'chat.completion',
                        'usage': {
                            'completion_tokens': 0,
                            'prompt_tokens': token_count,
                            'total_tokens': token_count,
                            'warning': 'Input length over limit'
                        }
                    }
                    responses.append(response)
                    continue
                
                # Generate response using HP640D
                start_time = time.time()
                response = ""
                result = []
                performance = None
                
                # Run inference
                try:
                    import asyncio
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    async def run_inference():
                        async for res in run_llm_model(self.model, input_ids, generation_config, timeout=36000):
                            if isinstance(res, dict):
                                return res, []
                            else:
                                result.append(res)
                        return None, result
                    
                    performance, result = loop.run_until_complete(run_inference())
                    loop.close()
                except Exception as e:
                    logger.error(f"Error in async inference: {str(e)}")
                    performance = None
                    result = []
                
                # Decode response
                response_text = self.model.tokenizer.batch_decode([result], skip_special_tokens=True)[0]
                inference_time = time.time() - start_time
                
                logger.debug(f"Response {i+1}: {response_text[:100]}...")
                if performance:
                    logger.debug(f"Performance info: {performance}")
                
                # Create OpenAI API compatible response
                response = {
                    'choices': [{
                        'index': 0,
                        'message': {
                            'content': response_text,
                            'role': 'assistant'
                        }
                    }],
                    'created': int(time.time()),
                    'model': self.config.get('model_id', 'hp640d-model'),
                    'object': 'chat.completion',
                    'usage': {
                        'completion_tokens': len(result),
                        'prompt_tokens': len(input_ids),
                        'total_tokens': len(input_ids) + len(result),
                        'inference_time': inference_time
                    }
                }
                
                # Add performance information if available
                if performance:
                    response['usage'].update(performance)
                
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
                    'model': self.config.get('model_id', 'hp640d-model'),
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
                # Stop the model
                self.model.stop()
                logger.info("HP640D model terminated successfully")
            except Exception as e:
                logger.warning(f"Error during model cleanup: {str(e)}")