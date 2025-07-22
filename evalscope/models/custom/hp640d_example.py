# Copyright (c) Alibaba, Inc. and its affiliates.
"""
HP640D Model Usage Example

This example demonstrates how to use the HP640D model with EvalScope.
"""

import logging
from evalscope.models.custom import HP640DModel
from evalscope.models.adapters import HP640DAdapter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_usage():
    """Example usage of HP640D model."""
    
    # Configuration for HP640D model
    config = {
        'model_path': '/path/to/your/hp640d/model',  # 替换为实际的模型路径
        'device_list': '0',  # 使用GPU 0
        'do_sample': True,
        'max_tokens': 512,
        'model_id': 'hp640d-example-model',
        # 其他HP640D参数
        'temperature': 0.7,
        'top_p': 0.9,
    }
    
    try:
        # 初始化HP640D模型
        logger.info("Initializing HP640D model...")
        hp640d_model = HP640DModel(config=config)
        
        # 创建适配器
        adapter = HP640DAdapter(hp640d_model)
        
        # 测试输入
        test_inputs = [
            "你好，请介绍一下你自己。",
            "什么是人工智能？",
            "请解释一下机器学习的基本概念。"
        ]
        
        # 推理配置
        infer_cfg = {
            'max_new_tokens': 256,
            'temperature': 0.7,
            'top_p': 0.9,
        }
        
        # 进行推理
        logger.info("Running inference...")
        responses = adapter.predict(test_inputs, infer_cfg=infer_cfg)
        
        # 输出结果
        for i, (input_text, response) in enumerate(zip(test_inputs, responses)):
            print(f"\n--- 输入 {i+1} ---")
            print(f"问题: {input_text}")
            print(f"回答: {response['choices'][0]['message']['content']}")
            print(f"Token使用: {response['usage']}")
            
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        raise


def example_with_dict_input():
    """Example with dictionary input format."""
    
    config = {
        'model_path': '/path/to/your/hp640d/model',
        'device_list': '0',
        'do_sample': True,
        'model_id': 'hp640d-dict-example',
    }
    
    try:
        hp640d_model = HP640DModel(config=config)
        adapter = HP640DAdapter(hp640d_model)
        
        # 使用字典格式的输入
        test_inputs = [
            {
                'data': ['请解释什么是深度学习？'],
                'system_prompt': '你是一个专业的AI助手。'
            },
            {
                'messages': [
                    {'role': 'system', 'content': '你是一个有用的助手。'},
                    {'role': 'user', 'content': '什么是自然语言处理？'}
                ]
            }
        ]
        
        responses = adapter.predict(test_inputs)
        
        for i, response in enumerate(responses):
            print(f"\n--- 字典输入 {i+1} ---")
            print(f"回答: {response['choices'][0]['message']['content']}")
            
    except Exception as e:
        logger.error(f"Error with dict input: {str(e)}")
        raise


if __name__ == "__main__":
    print("HP640D Model Example")
    print("=" * 50)
    
    # 运行示例
    example_usage()
    
    print("\n" + "=" * 50)
    print("Dictionary Input Example")
    print("=" * 50)
    
    example_with_dict_input() 