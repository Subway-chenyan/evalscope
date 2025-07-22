import argparse
from evalscope import run_task, TaskConfig
from evalscope.models.custom.lynllm_model import LynLLMModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LynLLM EvalScope Example")
    parser.add_argument(
        "--model_path", 
        type=str, 
        # required=True,
        default="/43_data/xinyi/lynllm_runtime/v0.2.0.2025070001/runtime/8_qwen3-8B_lynxi_text-text_b1_4096_ddr__LINEAR_8_num_30",
        help="Path to LynLLM model"
    )
    parser.add_argument(
        "--device_list", 
        type=str, 
        default="16,17,18,19,20,21,22,23",
        help="Device list for inference (e.g., '0' or '0,1')"
    )
    parser.add_argument(
        "--do_sample", 
        action="store_true",
        help="Enable sampling during inference"
    )
    parser.add_argument(
        "--show_speed", 
        action="store_true",
        help="Display inference speed information"
    )
    parser.add_argument(
        "--model_id", 
        type=str, 
        default="lynllm-model",
        help="Custom model identifier"
    )
    parser.add_argument(
        "--max_length", 
        type=int, 
        help="Maximum sequence length for inference"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        help="Temperature for sampling"
    )
    parser.add_argument(
        "--top_p", 
        type=float, 
        help="Top-p value for sampling"
    )
    
    args = parser.parse_args()
    # Add any additional LynLLM parameters
    if hasattr(args, 'max_length') and args.max_length:
        config['max_length'] = args.max_length
    if hasattr(args, 'temperature') and args.temperature:
        config['temperature'] = args.temperature
    if hasattr(args, 'top_p') and args.top_p:
        config['top_p'] = args.top_p
            
    config = {
        'model_path': args.model_path,
        'device_list': args.device_list,
        'do_sample': args.do_sample,
        'show_speed': args.show_speed,
        'model_id': args.model_id or 'lynllm-model'
    }
    # 实例化lynxiCustomModel
    lynllm_model = LynLLMModel(config=config)

    # 配置评测任务
    task_config = TaskConfig(
        model=lynllm_model,
        model_id='lynllm-model',  # 自定义模型ID
        datasets=['math_500'],
        eval_type='custom',  # 必须为custom
        # generation_config={
        #     'max_new_tokens': args.max_length,
        #     'temperature': args.temperature,
        #     'top_p': args.top_p,
        #     'top_k': 50,
        #     'repetition_penalty': 1.0,
        #     'do_sample': args.do_sample,
        # },
        debug=True,
        # limit=10,
    )

    # 运行评测任务
    eval_results = run_task(task_cfg=task_config)
    print(eval_results)