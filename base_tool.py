from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Type
from dataclasses import dataclass
from enum import Enum
import time
from functools import wraps
from dataclasses import dataclass, field 

class ToolCategory(Enum):
    """工具类别"""
    SEARCH = "search"
    LAW = "law"
    DATABASE = "database"


@dataclass
class ToolResult:
    """工具执行结果"""
    success: bool
    content: str
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


def timer_decorator(func):
    """执行时间装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        
        if isinstance(result, ToolResult):
            result.execution_time = elapsed
        return result
    return wrapper


class BaseTool(ABC):
    """工具基类"""
    
    def __init__(self, name: str, description: str, category: ToolCategory):
        self.name = name
        self.description = description
        self.category = category
    
    @abstractmethod
    @timer_decorator
    def execute(self, **kwargs) -> ToolResult:
        """执行工具"""
        pass
    
    def validate_params(self, **kwargs) -> bool:
        """参数验证"""
        return True
    
    def format_result(self, result: ToolResult) -> str:
        """格式化结果"""
        return result.content


class ToolRegistry:
    """工具注册器（单例）"""
    
    _instance = None
    _tools: Dict[str, BaseTool] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def register(self, tool: BaseTool):
        """注册工具"""
        self._tools[tool.name] = tool
    
    def get(self, name: str) -> Optional[BaseTool]:
        """获取工具"""
        return self._tools.get(name)
    
    def list_tools(self) -> Dict[str, BaseTool]:
        """列出所有工具"""
        return self._tools.copy()
    
    def get_by_category(self, category: ToolCategory) -> Dict[str, BaseTool]:
        """按类别获取工具"""
        return {name: tool for name, tool in self._tools.items() 
                if tool.category == category}


# 全局工具注册器
tool_registry = ToolRegistry()