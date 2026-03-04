from typing import List, Dict, Any, Optional
from openai import OpenAI

from config import config


class DeepSeekClient:
    """DeepSeek客户端封装（单例）"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, 'client'):
            return
        
        if not config.api.deepseek_key:
            raise ValueError("DEEPSEEK_API_KEY 未配置")
        
        self.client = OpenAI(
            api_key=config.api.deepseek_key,
            base_url=config.api.deepseek_base_url
        )
        self.model = config.api.deepseek_model
        print("✅ DeepSeek客户端初始化成功")
    
    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        """发送聊天请求"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"❌ LLM调用失败: {e}")
            raise


# 全局LLM实例
_llm_instance = None

def get_llm() -> DeepSeekClient:
    """获取LLM实例"""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = DeepSeekClient()
    return _llm_instance