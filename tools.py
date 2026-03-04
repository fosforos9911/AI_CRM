from base_tool import tool_registry

# 导入工具（注册会自动发生）
import rag_law
import sql_agent
import weather


# 为保持向后兼容，提供旧的接口
def get_tool(name: str):
    """获取工具"""
    return tool_registry.get(name)


def get_all_tools():
    """获取所有工具"""
    return list(tool_registry.list_tools().values())


def get_tools_dict():
    """获取工具字典（工具名 -> 工具）"""
    return tool_registry.list_tools()


# 兼容旧的tools变量
tools = get_all_tools()
tools_dict = get_tools_dict()
