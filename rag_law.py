from pathlib import Path
from typing import List, Optional, Dict, Any
import re
from datetime import datetime
import shutil

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

from config import config
from base_tool import BaseTool, ToolResult, ToolCategory, tool_registry


class HuggingFaceEmbeddings(Embeddings):
    """使用HuggingFace在线模型的嵌入封装"""

    # 可用的中文模型列表（按推荐程度排序）
    # 将 bge-base-zh-v1.5 作为默认模型（768 维），
    # 以兼容之前已经构建好的向量库。
    AVAILABLE_MODELS = [
        {
            "name": "BAAI/bge-base-zh-v1.5",
            "description": "中型中文模型，效果更好",
            "size": "~1.1GB",
        },
        {
            "name": "BAAI/bge-small-zh-v1.5",
            "description": "小型中文模型，速度快，效果不错",
            "size": "~400MB",
        },
    ]

    def __init__(self, model_name: str = None):
        """
        初始化HuggingFace嵌入模型

        Args:
            model_name: HuggingFace模型名称，如果为None则使用默认模型
        """
        self.model = None
        self.model_name = model_name or self.AVAILABLE_MODELS[0]["name"]
        self._load_model()

    def _load_model(self):
        """加载模型"""
        print(f"\n📦 正在加载模型: {self.model_name}")

        try:
            # 加载模型
            self.model = SentenceTransformer(self.model_name)

            # 获取模型信息
            dim = self.model.get_sentence_embedding_dimension()

            print(f"✅ 模型加载成功！向量维度: {dim}")

        except Exception as e:
            print(f"⚠️ 模型 {self.model_name} 加载失败: {e}")
            print("🔄 尝试使用备用模型...")

            # 尝试使用备用模型
            for model_info in self.AVAILABLE_MODELS[1:]:
                try:
                    print(f"📦 尝试备用模型: {model_info['name']}")
                    self.model = SentenceTransformer(model_info["name"])
                    self.model_name = model_info["name"]

                    dim = self.model.get_sentence_embedding_dimension()
                    print(f"✅ 备用模型加载成功！向量维度: {dim}")
                    return

                except Exception as e2:
                    print(f"❌ 备用模型加载失败: {e2}")
                    continue

            # 如果所有模型都失败
            error_msg = "\n❌ 所有模型都无法加载。请检查网络连接或手动下载模型。"
            raise Exception(error_msg)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入多个文档"""
        if self.model is None:
            raise RuntimeError("模型未正确初始化")

        try:
            embeddings = self.model.encode(texts, show_progress_bar=False)
            return embeddings.tolist()
        except Exception as e:
            print(f"❌ 文档嵌入失败: {e}")
            raise

    def embed_query(self, text: str) -> List[float]:
        """嵌入单个查询"""
        if self.model is None:
            raise RuntimeError("模型未正确初始化")

        try:
            embedding = self.model.encode(text)
            return embedding.tolist()
        except Exception as e:
            print(f"❌ 查询嵌入失败: {e}")
            raise


class LawVectorStore:
    """法律向量存储管理类"""

    _instance = None
    _vector_store = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._init_vector_store()
            self._initialized = True

    def _init_vector_store(self):
        """初始化向量存储"""
        print("\n🔧 初始化法律向量存储...")

        # 先检查是否存在有效的向量存储
        if config.paths.chroma_dir.exists() and any(config.paths.chroma_dir.iterdir()):
            try:
                # 先创建嵌入模型（但只用于加载，不重新创建向量库）
                self.embeddings = HuggingFaceEmbeddings()

                print(f"📂 加载现有向量存储: {config.paths.chroma_dir}")
                self._vector_store = Chroma(
                    persist_directory=str(config.paths.chroma_dir),
                    embedding_function=self.embeddings,
                    collection_name=config.vector_store.collection_name,
                )
                # 测试连接
                count = self._vector_store._collection.count()
                print(f"✅ 加载现有向量存储成功，包含 {count} 个文档")
                return
            except Exception as e:
                print(f"⚠️ 加载现有存储失败: {e}")
                print("🔄 将重新创建向量存储...")

        # 如果没有现有存储或加载失败，创建新存储
        self._create_vector_store()

    def _create_vector_store(self):
        """创建新的向量存储"""
        print("\n🔄 创建新的向量存储...")

        # 创建嵌入模型
        self.embeddings = HuggingFaceEmbeddings()

        # 确保法律文件存在
        self._ensure_law_file()

        # 加载文档
        print(f"📄 加载法律文件: {config.paths.law_file}")
        loader = TextLoader(str(config.paths.law_file), encoding="utf-8")
        documents = loader.load()
        print(f"✅ 加载了 {len(documents)} 个文档")

        # 拆分文档
        print("✂️ 正在拆分文档...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.vector_store.chunk_size,
            chunk_overlap=config.vector_store.chunk_overlap,
            separators=["\n\n", "\n", "。", "，", " ", ""],
        )
        splits = splitter.split_documents(documents)
        print(f"✅ 文档拆分为 {len(splits)} 个片段")

        # 创建向量存储
        print("💾 正在创建向量存储...")
        self._vector_store = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=str(config.paths.chroma_dir),
            collection_name=config.vector_store.collection_name,
        )
        print(f"✅ 向量存储创建完成，已保存到: {config.paths.chroma_dir}")

    def _ensure_law_file(self):
        """确保法律文件存在，不存在则创建示例"""
        if config.paths.law_file.exists():
            return

        print("📝 创建示例法律文件")
        example_content = """中华人民共和国劳动法
第四条 用人单位应当依法建立和完善规章制度，保障劳动者享有劳动权利和履行劳动义务。

第四十四条 有下列情形之一的，用人单位应当按照下列标准支付高于劳动者正常工作时间工资的工资报酬：
（一）安排劳动者延长工作时间的，支付不低于工资的百分之一百五十的工资报酬；
（二）休息日安排劳动者工作又不能安排补休的，支付不低于工资的百分之二百的工资报酬；
（三）法定休假日安排劳动者工作的，支付不低于工资的百分之三百的工资报酬。

中华人民共和国民法典
第五百零二条 依法成立的合同，自成立时生效，但是法律另有规定或者当事人另有约定的除外。

中华人民共和国刑法
第二百三十四条 故意伤害他人身体的，处三年以下有期徒刑、拘役或者管制。"""

        config.paths.law_file.write_text(example_content, encoding="utf-8")
        print(f"✅ 示例法律文件创建完成: {config.paths.law_file}")

    def search(self, query: str, top_k: int = None) -> List[Document]:
        """搜索相关法律条文"""
        if top_k is None:
            top_k = config.vector_store.top_k

        print(f"🔍 搜索: '{query}'")
        try:
            results = self._vector_store.similarity_search(query, k=top_k)
        except Exception as e:
            # 处理向量维度不一致导致的错误（例如模型更换后）
            if "Collection expecting embedding with dimension" in str(e):
                print(f"⚠️ 检测到向量维度不匹配: {e}")
                print("🔄 正在清理旧的向量库并重新构建...")
                try:
                    shutil.rmtree(config.paths.chroma_dir, ignore_errors=True)
                except Exception as clean_err:
                    print(f"⚠️ 清理旧向量库失败: {clean_err}")
                # 重新创建向量存储并重试一次
                self._create_vector_store()
                results = self._vector_store.similarity_search(query, k=top_k)
            else:
                raise
        print(f"✅ 找到 {len(results)} 条相关结果")
        return results


class LawTool(BaseTool):
    """法律查询工具"""

    def __init__(self):
        super().__init__(
            name="rag_law",
            description="查询中国法律条文，基于RAG向量检索",
            category=ToolCategory.LAW,
        )
        print("\n⚖️ 初始化法律查询工具...")
        self.store = LawVectorStore()
        print("✅ 法律查询工具初始化完成")

    def _extract_law_info(self, text: str) -> Dict[str, str]:
        """提取法律条文信息"""

        # 检测是否包含多个条文，如果是则只保留第一条
        article_pattern = r"(第[一二三四五六七八九十百千万\d]+条)"
        matches = list(re.finditer(article_pattern, text))

        if len(matches) > 1:
            # 多个条文，只取第一条
            first_article_start = matches[0].start()
            text = text[first_article_start:]
            # 找到第二条的开始位置，截断
            second_article_start = matches[1].start()
            text = text[:second_article_start]

        info = {
            "law_name": "未知",
            "article_number": "未知",
            "content": text.strip(),
            "keywords": "",
            "effectiveness": "有效",
        }

        # 提取法律名称
        law_patterns = [r"(中华人民共和国|中华人民)(.+?)(法)", r"(.+?)(法|条例|规定)"]
        for pattern in law_patterns:
            match = re.search(pattern, text)
            if match:
                info["law_name"] = match.group(0)
                break

        # 提取条文号
        article_pattern = r"(第[一二三四五六七八九十百千万\d]+条)"
        match = re.search(article_pattern, text)
        if match:
            info["article_number"] = match.group(1)

        # 提取关键词
        info["keywords"] = text[:30] + "..."

        return info

    def execute(self, query: str) -> ToolResult:
        """执行法律查询"""
        try:
            if not query or not isinstance(query, str):
                return ToolResult(
                    success=False,
                    content=config.prompts.output_templates["error"].format(
                        error_type="参数错误",
                        error_message="查询参数无效",
                        suggestion="请输入有效的法律查询关键词",
                    ),
                    error="查询参数无效",
                )

            docs = self.store.search(query)

            if not docs:
                return ToolResult(
                    success=True,
                    content=config.prompts.output_templates["no_result"].format(
                        message=f"未找到与「{query}」相关的法律条文"
                    ),
                )

            # 取第一条最相关的结果
            doc = docs[0]
            law_info = self._extract_law_info(doc.page_content)

            # 使用模板格式化输出
            content = config.prompts.output_templates["rag_law"].format(
                law_name=law_info["law_name"],
                article_number=law_info["article_number"],
                content=law_info["content"],
                keywords=law_info["keywords"],
            )

            return ToolResult(
                success=True, content=content, metadata={"doc_count": len(docs)}
            )

        except Exception as e:
            return ToolResult(
                success=False,
                content=config.prompts.output_templates["error"].format(
                    error_type="法律查询失败",
                    error_message=str(e),
                    suggestion="请稍后重试",
                ),
                error=f"法律查询失败: {str(e)}",
            )


# 注册工具
tool_registry.register(LawTool())
