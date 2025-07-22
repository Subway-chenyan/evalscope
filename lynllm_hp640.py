import argparse
from evalscope import run_task, TaskConfig
import logging
from evalscope.models.custom import HP640DModel
from evalscope.models.adapters import HP640DAdapter

if __name__ == "__main__":
    config = {
        'model_path': '/1_data/liuxuan/llm_model/0715/dsd-qwen2-5-7B_b56_7680_LINEAR_8_8_kv_80_ffn1_2token_no_softmax/',
        'model_id': 'hp640d-master',
        'do_sample': True,
        # Master node configuration
        'master': True,
        'ray_cluster_head_ip': '172.16.39.1',  # Master node IP
        'ray_cluster_head_port': 6656,
        # 支持从文件读取IP列表
        'ip_file': '/1_data/liuxuan/ip_list_1-41-51.txt',  # 可选,从文件读取IP列表
    }
    
    model = HP640DModel(config=config)
    adapter = HP640DAdapter(model)

    # 配置评测任务
    task_config = TaskConfig(
        model=model,
        model_id='hp640d-master',  # 自定义模型ID
        datasets=['math_500'],
        eval_type='custom',  # 必须为custom
        dataset_dir = '/1_data/zhiwei.zhu/distributed/evalscope/data/math-500',
        # generation_config={
        #     'max_new_tokens': args.max_length,
        #     'temperature': args.temperature,
        #     'top_p': args.top_p,
        #     'top_k': 50,
        #     'repetition_penalty': 1.0,
        #     'do_sample': args.do_sample,
        # },
        debug=True,
        limit=10,
    )

    # 运行评测任务
    eval_results = run_task(task_cfg=task_config)
    print(eval_results)