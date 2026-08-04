import asyncio
import os
import sys
import traceback
import json
import re
from typing import Dict, Any, Optional
import signal

# 安全执行代码的环境
class CodeExecutor:
    def __init__(self, tools: Dict[str, Any], max_execution_time: int = 30):
        self.tools = tools
        self.max_execution_time = max_execution_time  # 最大执行时间（秒）
        self.dangerous_patterns = [
            r'import\s+os',
            r'import\s+sys',
            r'import\s+subprocess',
            r'import\s+eval',
            r'import\s+exec',
            r'import\s+open',
            r'import\s+file',
            r'import\s+builtins',
            r'import\s+__import__',
            r'os\.',
            r'sys\.',
            r'subprocess\.',
            r'eval\(',
            r'exec\(',
            r'open\(',
            r'file\(',
            r'__import__\(',
            r'builtins\.',
            r'globals\(',
            r'locals\(',
            r'vars\(',
            r'dir\(',
            r'getattr\(',
            r'setattr\(',
            r'delattr\(',
            r'hasattr\(',
            r'__dict__',
            r'__class__',
            r'__bases__',
            r'__subclasses__',
            r'__getattribute__',
            r'__setattribute__',
            r'__delattribute__',
            r'__get__',
            r'__set__',
            r'__delete__',
            r'__call__',
            r'__new__',
            r'__init__',
            r'__exit__',
            r'__enter__',
            r'__repr__',
            r'__str__',
            r'__format__',
            r'__bytes__',
            r'__sizeof__',
            r'__dir__',
            r'__class_getitem__',
            r'__instancecheck__',
            r'__subclasscheck__',
            r'__subclasshook__',
            r'__sizeof__',
            r'__reduce__',
            r'__reduce_ex__',
            r'__getstate__',
            r'__setstate__',
            r'__getnewargs__',
            r'__getnewargs_ex__',
            r'__getinitargs__',
            r'__weakrefoffset__',
            r'__dictoffset__',
            r'__base__',
            r'__mro__',
            r'__qualname__',
            r'__module__',
            r'__doc__',
            r'__annotations__',
            r'__builtins__'
        ]
    
    async def execute_code(self, code: str) -> Dict[str, Any]:
        """安全执行生成的代码"""
        try:
            # 清理代码，去除可能的markdown标记
            code = self._clean_code(code)
            
            # 代码安全审查
            if not self._is_safe_code(code):
                return {"error": "代码包含潜在的安全风险，执行被拒绝"}
            
            # 创建执行环境
            exec_globals = {
                'asyncio': asyncio,
                'json': json,
                **self.tools  # 注入工具函数
            }
            
            exec_locals = {}
            
            # 执行代码
            exec(code, exec_globals, exec_locals)
            
            # 检查是否定义了run函数
            if 'run' not in exec_locals:
                return {"error": "代码中未定义run函数"}
            
            # 执行run函数
            run_func = exec_locals['run']
            if not asyncio.iscoroutinefunction(run_func):
                return {"error": "run函数必须是异步函数"}
            
            # 执行run函数（带超时）
            try:
                result = await asyncio.wait_for(run_func(), timeout=self.max_execution_time)
                return result
            except asyncio.TimeoutError:
                return {"error": f"代码执行超时（超过{self.max_execution_time}秒）"}
            
        except Exception as e:
            error_msg = f"代码执行失败: {str(e)}\n{traceback.format_exc()}"
            return {"error": error_msg}
    
    def _clean_code(self, code: str) -> str:
        """清理代码，去除可能的markdown标记"""
        # 去除markdown代码块标记
        code = code.strip()
        if code.startswith('```python'):
            code = code[9:]
        elif code.startswith('```'):
            code = code[3:]
        
        if code.endswith('```'):
            code = code[:-3]
        
        return code.strip()
    
    def _is_safe_code(self, code: str) -> bool:
        """检查代码是否安全"""
        for pattern in self.dangerous_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                return False
        return True