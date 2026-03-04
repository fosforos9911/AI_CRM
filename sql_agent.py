import sqlite3
from typing import Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime

from config import config
from base_tool import BaseTool, ToolResult, ToolCategory, tool_registry
from llm_client import get_llm


@dataclass
class TableSchema:
    """数据库表结构"""
    name: str
    columns: List[Tuple[str, str]]  # (column_name, column_type)


class DatabaseManager:
    """数据库管理器（单例）"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.db_path = config.paths.db_path
        self._init_database()
        self._initialized = True
    
    def _init_database(self):
        """初始化数据库"""
        print(f"📊 初始化数据库: {self.db_path}")
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # 创建表
            self._create_tables(cursor)
            
            # 插入示例数据
            self._insert_sample_data(cursor)
            
            conn.commit()
        
        print("✅ 数据库初始化完成")
    
    def _create_tables(self, cursor):
        """创建表"""
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                age INTEGER,
                email TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS products (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                price REAL,
                stock INTEGER
            )
        ''')
    
    def _insert_sample_data(self, cursor):
        """插入示例数据"""
        # 检查是否已有数据
        cursor.execute("SELECT COUNT(*) FROM users")
        if cursor.fetchone()[0] > 0:
            return
        
        users_data = [
            ('张三', 25, 'zhangsan@example.com'),
            ('李四', 30, 'lisi@example.com'),
            ('王五', 28, 'wangwu@example.com'),
            ('赵六', 35, 'zhaoliu@example.com'),
        ]
        cursor.executemany(
            "INSERT INTO users (name, age, email) VALUES (?, ?, ?)",
            users_data
        )
        
        products_data = [
            ('笔记本电脑', 5999.99, 50),
            ('智能手机', 2999.99, 100),
            ('平板电脑', 1999.99, 30),
            ('无线耳机', 499.99, 200),
        ]
        cursor.executemany(
            "INSERT INTO products (name, price, stock) VALUES (?, ?, ?)",
            products_data
        )
        
        print("✅ 示例数据插入成功")
    
    def get_connection(self):
        """获取数据库连接"""
        return sqlite3.connect(str(self.db_path))
    
    def get_schema(self) -> List[TableSchema]:
        """获取数据库结构"""
        schemas = []
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # 获取所有表
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            
            for (table_name,) in tables:
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = [(col[1], col[2]) for col in cursor.fetchall()]
                schemas.append(TableSchema(table_name, columns))
        
        return schemas
    
    def execute_query(self, sql: str) -> Tuple[bool, str, Optional[List]]:
        """执行SQL查询"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(sql)
                
                # 判断是否是SELECT查询
                is_select = sql.strip().upper().startswith("SELECT")
                
                if is_select:
                    rows = cursor.fetchall()
                    if rows:
                        columns = [desc[0] for desc in cursor.description]
                        return True, "查询成功", (columns, rows)
                    else:
                        return True, "查询无结果", None
                else:
                    conn.commit()
                    return True, f"操作成功，影响行数: {cursor.rowcount}", None
                    
        except Exception as e:
            return False, f"SQL执行错误: {str(e)}", None


class SQLTool(BaseTool):
    """数据库查询工具"""
    
    def __init__(self):
        super().__init__(
            name="sql_agent",
            description="使用自然语言查询SQLite数据库",
            category=ToolCategory.DATABASE
        )
        self.db = DatabaseManager()
        self.llm = get_llm()
    
    def _natural_language_to_sql(self, question: str) -> str:
        """自然语言转SQL"""
        
        # 获取数据库结构
        schemas = self.db.get_schema()
        schema_desc = []
        for table in schemas:
            cols = ", ".join([f"{name}({type})" for name, type in table.columns])
            schema_desc.append(f"{table.name}表: {cols}")
        
        schema_text = "\n".join(schema_desc)
        
        # 使用配置中的SQL提示词
        prompt = config.prompts.sql_prompt.format(
            schema_text=schema_text,
            question=question
        )
        
        sql = self.llm.chat([{"role": "user", "content": prompt}], temperature=0)
        
        # 清理SQL语句
        sql = sql.strip().replace("```sql", "").replace("```", "").strip()
        
        return sql
    
    def _format_query_data(self, columns: List[str], rows: List[tuple]) -> str:
        """格式化查询数据"""
        if not rows:
            return "无数据"
        
        formatted_rows = []
        for i, row in enumerate(rows, 1):
            row_data = []
            for j, col_name in enumerate(columns):
                row_data.append(f"    {col_name}: {row[j]}")
            formatted_rows.append(f"  [{i}]\n" + "\n".join(row_data))
        
        return "\n\n".join(formatted_rows)
    
    def _determine_query_type(self, sql: str) -> str:
        """确定查询类型"""
        sql_upper = sql.strip().upper()
        if "COUNT" in sql_upper:
            return "统计查询"
        elif "SELECT" in sql_upper:
            if "users" in sql.lower():
                return "用户查询"
            elif "products" in sql.lower():
                return "产品查询"
            else:
                return "数据查询"
        else:
            return "操作查询"
    
    def execute(self, query: str) -> ToolResult:
        """执行数据库查询"""
        try:
            # 自然语言转SQL
            sql = self._natural_language_to_sql(query)
            
            # 执行SQL
            success, message, data = self.db.execute_query(sql)
            
            if not success:
                return ToolResult(
                    success=False,
                    content=config.prompts.output_templates["error"].format(
                        error_type="SQL执行错误",
                        error_message=message,
                        suggestion="请尝试重新描述您的问题"
                    ),
                    error=message
                )
            
            # 格式化结果
            if data is None:
                content = config.prompts.output_templates["no_result"].format(
                    message=message
                )
            else:
                columns, rows = data
                formatted_data = self._format_query_data(columns, rows)
                query_type = self._determine_query_type(sql)
                
                content = config.prompts.output_templates["sql_agent"].format(
                    query_type=query_type,
                    count=len(rows),
                    data=formatted_data,
                    query_time=f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                )
            
            return ToolResult(
                success=True,
                content=content,
                metadata={
                    "sql": sql,
                    "row_count": len(rows) if data and len(data) == 2 else 0
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                content=config.prompts.output_templates["error"].format(
                    error_type="数据库查询失败",
                    error_message=str(e),
                    suggestion="请检查数据库连接或重试"
                ),
                error=f"数据库查询失败: {str(e)}"
            )


# 注册工具
tool_registry.register(SQLTool())