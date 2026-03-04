import os
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv

# 加载.env文件
load_dotenv()

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.absolute()


@dataclass
class PathConfig:
    """路径配置"""

    # 数据目录
    data_dir: Path = PROJECT_ROOT / "data"
    chroma_dir: Path = data_dir / "chroma"
    db_path: Path = data_dir / "database.db"

    # 模型缓存目录
    models_dir: Path = PROJECT_ROOT / "data" / "llm" / "cache"

    # 法律文件
    law_file: Path = data_dir / "law.txt"

    def __post_init__(self):
        """确保目录存在"""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 数据目录: {self.data_dir}")
        print(f"📁 模型缓存目录: {self.models_dir}")
        print(f"📁 Chroma目录: {self.chroma_dir}")


@dataclass
class APIConfig:
    """API密钥配置"""

    deepseek_key: str = field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", ""))
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek-chat"

    qweather_key: str = field(default_factory=lambda: os.getenv("QWEATHER_API_KEY", ""))

    def __post_init__(self):
        """验证API密钥"""
        if not self.deepseek_key:
            print("⚠️ 警告: DEEPSEEK_API_KEY 未配置，LLM功能将不可用")
        if not self.qweather_key:
            print("ℹ️ 提示: QWEATHER_API_KEY 未配置，天气查询将使用模拟数据")


@dataclass
class ModelConfig:
    """在线模型配置"""

    # 默认使用的模型
    default_model: str = "BAAI/bge-small-zh-v1.5"

    # 备用模型
    fallback_model: str = "BAAI/bge-base-zh-v1.5"

    # HuggingFace 镜像源配置
    hf_mirror: str = "https://hf-mirror.com"

    # 是否使用镜像源
    use_mirror: bool = True

    # 模型下载超时时间（秒）
    timeout: int = 30


@dataclass
class VectorStoreConfig:
    """向量存储配置"""

    collection_name: str = "law_documents"
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 3

    # 检索时返回的最大结果数
    max_results: int = 5

    # 相似度阈值
    similarity_threshold: float = 0.5


@dataclass
class PromptConfig:
    """提示词配置"""

    # 工具输出格式模板
    output_templates: Dict[str, str] = field(
        default_factory=lambda: {
            "rag_law": """【法律条文】
📚 法律名称：{law_name}
📖 条文编号：{article_number}
📝 条文内容：{content}
📌 关键词：{keywords}""",
            "sql_agent": """【数据库查询结果】
📊 查询类型：{query_type}
🔢 结果数量：{count}
📋 数据详情：
{data}
⏱️ 查询耗时：{query_time}""",
            "error": """【错误信息】
❌ 错误类型：{error_type}
📝 错误详情：{error_message}
💡 建议：{suggestion}""",
            "no_result": """【查询结果】
ℹ️ 提示：{message}""",
        }
    )

    # SQL 转换提示词
    sql_prompt: str = """你是一个SQL专家。请将以下自然语言问题转换为SQLite SQL查询语句。
直接返回SQL语句，不要任何解释。

数据库表结构：
{schema_text}

问题：{question}

SQL:"""


@dataclass
class AgentConfig:
    """Agent配置"""

    # 意图识别规则
    intent_keywords: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "weather": [
                "天气",
                "气温",
                "下雨",
                "晴天",
                "多云",
                "温度",
                "台风",
                "气象",
            ],
            "rag_law": [
                "法律",
                "法条",
                "规定",
                "条文",
                "劳动法",
                "刑法",
                "民法典",
                "合同法",
                "加班费",
                "赔偿",
                "劳动合同",
            ],
            "sql_agent": [
                "数据库",
                "查询",
                "用户",
                "产品",
                "表",
                "数据",
                "统计",
                "数量",
                "多少",
                "列表",
                "库存",
            ],
        }
    )

    # 系统提示词
    system_prompt: str = (
        """你是一个智能助手，可以帮助用户查询天气、法律条文和数据库信息。"""
    )

    # 是否显示详细日志
    verbose: bool = False


class Config:
    """统一配置入口（单例）"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "initialized"):
            return

        self.paths = PathConfig()
        self.api = APIConfig()
        self.models = ModelConfig()
        self.vector_store = VectorStoreConfig()
        self.agent = AgentConfig()
        self.prompts = PromptConfig()
        self.initialized = True

        # 设置环境变量
        self._setup_environment()

    def _setup_environment(self):
        """设置环境变量"""
        if self.models.use_mirror:
            os.environ["HF_ENDPOINT"] = self.models.hf_mirror
        os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(self.models.timeout)

    def print_info(self):
        """打印配置信息"""
        print("\n" + "=" * 60)
        print("🤖 智能助手配置信息")
        print("=" * 60)
        print(f"📁 数据目录: {self.paths.data_dir}")
        print(f"📁 模型缓存: {self.paths.models_dir}")
        print(f"📁 Chroma目录: {self.paths.chroma_dir}")
        print(f"\n📦 模型配置:")
        print(f"   - 默认模型: {self.models.default_model}")
        print(f"   - 备用模型: {self.models.fallback_model}")
        print(
            f"   - 镜像源: {self.models.hf_mirror if self.models.use_mirror else '官方源'}"
        )
        print("=" * 60)


# 全局配置实例
config = Config()


def get_config() -> Config:
    """获取全局配置实例"""
    return config


def use_mirror(enable: bool = True):
    """设置是否使用 HuggingFace 镜像"""
    config.models.use_mirror = enable
    config._setup_environment()


if __name__ == "__main__":
    config.print_info()
