# DRAFT优化流程集成指南

本指南说明了如何将DRAFT优化流程与现有代码配合使用，以自动优化工具文档，提高工具的可用性和理解性。

## 一、DRAFT优化流程概述

DRAFT（Document Refinement and Analysis Framework for Tools）是一个工具文档优化框架，包含三个核心角色：

- **Explorer**：生成用户查询和参数，探索工具的使用场景
- **Analyzer**：分析使用示例，提供改进建议
- **Rewriter**：根据建议优化工具文档

## 二、集成位置

DRAFT优化流程已集成到以下文件中：

- **from_plan_to_result.py**：包含完整的DRAFT优化流程代码
  - 核心函数：`optimize_tool_documentation(tool_info)`
  - 辅助函数：`openai_response`、`openai_embedding`、`cosine_similarity`等

## 三、使用方法

### 方法1：直接运行优化示例

```bash
# 运行工具文档优化
python from_plan_to_result.py --mode optimize
```

这将优化示例工具列表，并将结果保存到`optimized_tools.json`文件中。

### 方法2：在代码中调用优化函数

```python
# 导入优化函数
from from_plan_to_result import optimize_tool_documentation

# 定义工具信息
tool_info = {
    "category": "academic",
    "name": "search_paper_id",
    "description": "根据关键词搜索论文ID",
    "required_parameters": {
        "keywords": "论文关键词列表"
    },
    "optional_parameters": {
        "years": "发表年份列表",
        "size": "返回结果数量"
    }
}

# 调用优化流程
optimized_tool = optimize_tool_documentation(tool_info)
print(f"优化后描述: {optimized_tool['description']}")
```

### 方法3：与问题处理流程配合使用

```python
# 在问题处理前优化工具文档
from from_plan_to_result import optimize_tool_documentation, process_questions

# 优化工具文档
tool_info = {...}  # 工具信息
optimized_tool = optimize_tool_documentation(tool_info)

# 使用优化后的工具信息更新系统
update_tool_documentation(optimized_tool)

# 处理问题
process_questions("query.json", "output.json")
```

## 四、与现有代码的配合

### 1. 与caller.py的配合

`caller.py`中的`TaskExecutor`类负责执行API调用，可以通过以下方式与DRAFT配合：

```python
# 在caller.py中添加工具优化功能
from from_plan_to_result import optimize_tool_documentation

class TaskExecutor:
    def __init__(self, tasks):
        self.tasks = {task["name"]: task for task in tasks}
        # 其他初始化代码

    def optimize_tools(self):
        """
        优化所有工具的文档
        """
        for task_name, task in self.tasks.items():
            # 获取工具信息
            tool_info = get_tool_info(task_name)
            # 优化工具文档
            optimized_tool = optimize_tool_documentation(tool_info)
            # 更新工具信息
            update_tool_info(optimized_tool)
```

### 2. 与问题处理流程的配合

可以在处理问题前先优化工具文档，提高工具的可理解性：

```python
# 在process_single_question函数中添加工具优化
async def process_single_question(item: Dict, idx: int) -> Dict:
    # 优化工具文档
    optimize_all_tools()

    # 原有问题处理代码
    question = item.get("question", "").strip()
    # ...
```

## 五、配置说明

DRAFT优化流程使用以下配置：

- **OpenAI API配置**：
  - `api_base`: https://xiaoai.plus/v1
  - `api_key`:

- **优化参数**：
  - `episodes`: 最大优化轮数（默认5轮）
  - `temperature`: 0.2（控制生成的随机性）
  - `top_p`: 1.0（控制生成的多样性）
  - `max_tokens`: 2000（最大生成token数）
  - `model`: gpt-4o（使用的模型）

## 六、优化效果示例

### 示例1：search_paper_id

- **原始描述**：根据关键词搜索论文ID
- **优化后描述**：This tool allows users to search for academic paper IDs using specified keywords. It requires a list of keywords as input and can optionally filter results by publication years and limit the number of results returned.

### 示例2：search_author_id

- **原始描述**：根据姓名搜索作者ID
- **优化后描述**：This tool enables users to search for author IDs by providing a name. It accepts an author's name as a required parameter and offers optional parameters to specify the organization and limit the number of results returned.

## 七、故障排除

### 常见问题及解决方案

1. **API调用失败**
   - 检查API密钥和base URL是否正确
   - 确保网络连接正常

2. **优化结果不理想**
   - 增加优化轮数（episodes参数）
   - 调整temperature参数

3. **提示模板加载失败**
   - 确保prompts目录存在且包含所需文件
   - 检查文件编码是否为UTF-8

## 八、最佳实践

1. **定期优化**：定期运行DRAFT优化流程，保持工具文档的质量
2. **批量优化**：一次性优化多个工具，提高效率
3. **人工审核**：对优化后的文档进行人工审核，确保准确性
4. **持续改进**：根据实际使用情况，不断调整优化参数

## 九、使用示例

### 示例1：优化单个工具

```python
from from_plan_to_result import optimize_tool_documentation

tool_info = {
    "category": "academic",
    "name": "search_paper_detail",
    "description": "根据论文ID获取论文详情",
    "required_parameters": {
        "paper_ids": "论文ID列表"
    },
    "optional_parameters": {}
}

optimized_tool = optimize_tool_documentation(tool_info)
print(f"优化前: {tool_info['description']}")
print(f"优化后: {optimized_tool['description']}")
```

### 示例2：批量优化工具

```python
from from_plan_to_result import optimize_tool_documentation
import json

# 加载工具列表
with open("tools.json", "r", encoding="utf-8") as f:
    tools = json.load(f)

# 批量优化
optimized_tools = []
for tool in tools:
    optimized_tool = optimize_tool_documentation(tool)
    optimized_tools.append(optimized_tool)

# 保存优化结果
with open("optimized_tools.json", "w", encoding="utf-8") as f:
    json.dump(optimized_tools, f, ensure_ascii=False, indent=2)
```

## 十、总结

DRAFT优化流程是一个强大的工具文档优化框架，可以：

- 自动生成更清晰、更详细的工具文档
- 提高工具的可用性和理解性
- 与现有代码无缝集成
- 适应不同类型的工具和使用场景

通过定期使用DRAFT优化流程，可以持续提高工具文档的质量，从而提升整个系统的用户体验和性能。
