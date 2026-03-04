import re
from typing import TypedDict, List, Dict, Any, Optional, Literal

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage

from config import config
from llm_client import get_llm
from tools import get_tool, get_all_tools


# 定义状态类型
class AgentState(TypedDict):
    """Agent状态"""

    messages: List[Any]
    intent: Optional[str]
    parameters: Dict[str, Any]
    tool_results: List[Dict[str, Any]]
    final_response: Optional[str]


class IntentRecognizer:
    """意图识别器"""

    def __init__(self):
        self.keywords = config.agent.intent_keywords
        self.llm = get_llm()

    def rule_based_match(self, text: str) -> tuple:
        """基于规则的意图匹配"""
        text_lower = text.lower()

        for intent, keywords in self.keywords.items():
            if any(kw in text_lower for kw in keywords):
                return intent, {"query": text}

        return "unknown", {}

    def recognize(self, text: str) -> tuple:
        """意图识别"""
        return self.rule_based_match(text)


class ToolExecutor:
    """工具执行器"""

    def execute_tool(self, tool_name: str, params: Dict) -> Dict:
        """执行单个工具

        Args:
            tool_name: 工具名称
            params: 工具参数
        """
        tool = get_tool(tool_name)
        if not tool:
            return {
                "tool": tool_name,
                "success": False,
                "content": f"未知工具: {tool_name}",
                "error": "Tool not found",
            }

        result = tool.execute(**params)

        return {
            "tool": tool_name,
            "success": result.success,
            "content": result.content,
            "error": result.error,
            "execution_time": result.execution_time,
        }


# 初始化组件
intent_recognizer = IntentRecognizer()
tool_executor = ToolExecutor()


def intent_recognition_node(state: AgentState) -> AgentState:
    """意图识别节点"""
    print("\n🎯 意图识别")

    last_msg = state["messages"][-1]
    if not isinstance(last_msg, HumanMessage):
        return state

    user_input = last_msg.content
    print(f"用户输入: {user_input}")

    intent, params = intent_recognizer.recognize(user_input)
    state["intent"] = intent
    state["parameters"] = params

    print(f"识别结果: {intent} -> {params}")

    return state


def execute_tools_node(state: AgentState) -> AgentState:
    """执行工具节点"""
    print("\n🛠️ 执行工具")

    if state["intent"] == "unknown" or not state["parameters"]:
        print("没有需要执行的工具")
        return state

    result = tool_executor.execute_tool(state["intent"], state["parameters"])
    state["tool_results"] = [result]

    if result["success"]:
        print(f"✅ 工具执行成功，耗时: {result['execution_time']:.2f}秒")
    else:
        print(f"❌ 工具执行失败: {result['error']}")

    return state


def response_generation_node(state: AgentState) -> AgentState:
    """生成响应节点"""
    print("\n💬 生成响应")

    if state.get("tool_results"):
        results = []
        for result in state["tool_results"]:
            if result["success"]:
                results.append(result["content"])
            else:
                results.append(f"抱歉，{result['error']}")

        final_response = "\n\n".join(results)
    else:
        tools_list = "\n".join(
            [f"• {tool.name}: {tool.description}" for tool in get_all_tools()]
        )

        final_response = f"""您好！我是您的智能助手，可以帮您：

{tools_list}

请问有什么可以帮您的？"""

    state["final_response"] = final_response
    state["messages"].append(AIMessage(content=final_response))

    print(f"响应: {final_response[:100]}...")

    return state


def route_after_intent(
    state: AgentState,
) -> Literal["execute_tools", "response_generation"]:
    """意图识别后的路由"""
    if state["intent"] != "unknown" and state.get("parameters"):
        return "execute_tools"
    return "response_generation"


def build_agent_graph():
    """构建Agent图"""
    workflow = StateGraph(AgentState)

    workflow.add_node("intent_recognition", intent_recognition_node)
    workflow.add_node("execute_tools", execute_tools_node)
    workflow.add_node("response_generation", response_generation_node)

    workflow.set_entry_point("intent_recognition")

    workflow.add_conditional_edges("intent_recognition", route_after_intent)

    workflow.add_edge("execute_tools", "response_generation")
    workflow.add_edge("response_generation", END)

    return workflow.compile()


def run_agent(user_input: str) -> str:
    """运行Agent

    Args:
        user_input: 用户输入
    """
    graph = build_agent_graph()

    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "intent": None,
        "parameters": {},
        "tool_results": [],
        "final_response": None,
    }

    final_state = graph.invoke(initial_state)
    return final_state["final_response"]


def main():
    """主函数"""
    print("=" * 60)
    print("🤖 智能助手启动")
    print("=" * 60)

    tools = get_all_tools()
    for tool in tools:
        print(f"  • {tool.name}: {tool.description}")
    print("=" * 60)
    print("输入'退出'结束程序")
    print("=" * 60)

    while True:
        user_input = input("\n👤 用户: ").strip()

        if user_input.lower() in ["退出", "exit", "quit"]:
            print("再见！")
            break

        if not user_input:
            continue

        response = run_agent(user_input)
        print(f"\n🤖 助手: {response}")


if __name__ == "__main__":
    main()
