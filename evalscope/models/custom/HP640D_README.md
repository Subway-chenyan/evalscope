# HP640D 模型集成说明

本文档介绍如何在 EvalScope 中使用 LynLLM HP640D 模型进行推理。

## 概述

HP640D 模型是基于 LynLLM HP640D 推理框架的自定义模型，完全按照您提供的推理流程实现：

1. 检查参数有效性
2. 构建输入 tokens
3. 检查输入长度限制
4. 执行推理
5. 解码输出

## 安装依赖

确保已安装 LynLLM 框架：

```bash
pip install LynLLM
```

## 基本使用

### 1. 导入模型

```python
from evalscope.models.custom import HP640DModel
from evalscope.models.adapters import HP640DAdapter
```

### 2. 配置模型

```python
config = {
    'model_path': '/path/to/your/hp640d/model',  # 模型路径
    'device_list': '0',  # GPU设备列表
    'do_sample': True,   # 是否使用采样
    'max_tokens': 512,   # 最大生成token数
    'model_id': 'hp640d-model',  # 模型ID
    # 其他HP640D参数
    'temperature': 0.7,
    'top_p': 0.9,
}
```

### 3. 初始化模型和适配器

```python
# 初始化模型
hp640d_model = HP640DModel(config=config)

# 创建适配器
adapter = HP640DAdapter(hp640d_model)
```

### 4. 执行推理

```python
# 输入数据
inputs = [
    "你好，请介绍一下你自己。",
    "什么是人工智能？"
]

# 推理配置
infer_cfg = {
    'max_new_tokens': 256,
    'temperature': 0.7,
    'top_p': 0.9,
}

# 执行推理
responses = adapter.predict(inputs, infer_cfg=infer_cfg)

# 处理结果
for response in responses:
    content = response['choices'][0]['message']['content']
    usage = response['usage']
    print(f"回答: {content}")
    print(f"Token使用: {usage}")
```

## 支持的输入格式

### 1. 字符串格式

```python
inputs = ["问题1", "问题2", "问题3"]
```

### 2. 字典格式（data字段）

```python
inputs = [
    {
        'data': ['请解释什么是深度学习？'],
        'system_prompt': '你是一个专业的AI助手。'
    }
]
```

### 3. 字典格式（messages字段）

```python
inputs = [
    {
        'messages': [
            {'role': 'system', 'content': '你是一个有用的助手。'},
            {'role': 'user', 'content': '什么是自然语言处理？'}
        ]
    }
]
```

### 4. 列表格式

```python
inputs = [['问题1', '问题2'], ['问题3']]
```

## 推理流程详解

### 1. 参数检查

```python
# 检查最大token限制
max_length = min(max_length, model.max_tokens_limit - 1)
sample_param = SamplingParams(max_tokens=max_length)
```

### 2. 输入处理

```python
# 构建输入tokens
input_ids = model.build_input(query=prompt_text)

# 检查输入长度
status, token_count = model.input_overlimit(input_ids, max_length=sample_param.max_tokens)
if status:
    # 处理超限情况
    return empty_response
```

### 3. 推理执行

```python
# 复制生成配置
generation_config = copy.deepcopy(self.generation_config)
generation_config.do_sample = self.do_sample

# 异步推理
async for res in run_llm_model(model, input_ids, generation_config, timeout=36000):
    if isinstance(res, dict):
        performance = res
        break
    else:
        result += res
```

### 4. 输出解码

```python
# 解码响应
response_text = tokenizer.batch_decode([result], skip_special_tokens=True)[0]
```

## 输出格式

模型返回 OpenAI API 兼容的格式：

```python
{
    'choices': [{
        'index': 0,
        'message': {
            'content': '生成的回答',
            'role': 'assistant'
        }
    }],
    'created': 1677664795,
    'model': 'hp640d-model',
    'object': 'chat.completion',
    'usage': {
        'completion_tokens': 17,
        'prompt_tokens': 57,
        'total_tokens': 74,
        'inference_time': 1.234
    }
}
```

## 错误处理

模型包含完善的错误处理机制：

1. **输入长度超限**：返回空响应并记录警告
2. **推理错误**：返回错误信息
3. **模型初始化失败**：抛出详细错误信息

## 性能优化

1. **异步推理**：使用异步生成器提高效率
2. **批量处理**：支持批量输入处理
3. **资源管理**：自动清理模型资源

## 示例代码

完整的使用示例请参考 `hp640d_example.py` 文件。

## 注意事项

1. 确保模型路径正确且模型文件存在
2. 检查GPU设备是否可用
3. 根据模型大小调整 `max_tokens` 参数
4. 监控内存使用情况，避免OOM错误

## 故障排除

### 常见问题

1. **ImportError**: 确保已正确安装 LynLLM
2. **模型路径错误**: 检查 `model_path` 配置
3. **GPU内存不足**: 减少 `max_tokens` 或使用更小的模型
4. **输入超限**: 检查输入长度，必要时截断

### 调试模式

启用详细日志：

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 更新日志

- v1.0.0: 初始版本，支持基本的HP640D模型推理
- 支持多种输入格式
- 完整的错误处理机制
- OpenAI API兼容的输出格式 