"""
Streamlit 客户端 - 智能助手Web界面
"""

import streamlit as st
import time
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
import sys

# 修复 Streamlit 上下文问题
try:
    from streamlit.web import cli as stcli
    from streamlit import runtime
except ImportError:
    stcli = None
    runtime = None

# 确保在正确的上下文中运行
if __name__ == "__main__" and runtime is not None and not runtime.exists():
    # 如果不是在 Streamlit 环境中运行，则通过 Streamlit 启动
    sys.argv = ["streamlit", "run", __file__]
    sys.exit(stcli.main())
else:
    # 正常的 Streamlit 代码从这里开始
    from main import run_agent
    from tools import get_all_tools

    # 页面配置
    st.set_page_config(
        page_title="智能助手",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # 自定义CSS样式 - 极简设计
    st.markdown("""
    <style>
        /* 全局字体 */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
        
        * {
            font-family: 'Inter', sans-serif;
        }
        
        /* 主内容区域 */
        .main-content {
            max-width: 900px;
            margin: 0 auto;
            padding: 0 1rem;
        }
        
        /* 聊天消息样式 - 无框线设计 */
        .chat-message {
            margin: 0;
        }
        
        .user-section {
            background-color: #f3f4f6;
            padding: 1.5rem 2rem;
            margin: 0;
            border-radius: 0;
            color: #1f2937;
        }
        
        .assistant-section {
            background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
            padding: 1.5rem 2rem;
            margin: 0;
            border-radius: 0;
            color: white;
        }
        
        .message-content {
            font-size: 1rem;
            line-height: 1.6;
            max-width: 800px;
            margin: 0 auto;
            white-space: pre-wrap;
        }
        
        .message-role {
            font-size: 0.8rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
            opacity: 0.7;
            max-width: 800px;
            margin-left: auto;
            margin-right: auto;
        }
        
        .assistant-section .message-role {
            color: rgba(255,255,255,0.8);
        }
        
        .user-section .message-role {
            color: #4b5563;
        }
        
        .time-tag {
            font-size: 0.7rem;
            margin-top: 0.5rem;
            opacity: 0.5;
            max-width: 800px;
            margin-left: auto;
            margin-right: auto;
        }
        
        .assistant-section .time-tag {
            color: rgba(255,255,255,0.6);
        }
        
        .user-section .time-tag {
            color: #6b7280;
        }
        
        /* 侧边栏样式 - 简洁 */
        .sidebar-content {
            padding: 1rem;
        }
        
        .sidebar-section {
            background: #f8fafc;
            border-radius: 16px;
            padding: 1.2rem;
            margin-bottom: 1.5rem;
            border: 1px solid #e2e8f0;
        }
        
        .sidebar-section-title {
            font-weight: 600;
            color: #1e293b;
            margin-bottom: 1rem;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        /* 工具卡片 - 极简 */
        .tool-item {
            padding: 0.5rem 0;
            border-bottom: 1px solid #e2e8f0;
            font-size: 0.9rem;
        }
        
        .tool-item:last-child {
            border-bottom: none;
        }
        
        .tool-name {
            font-weight: 600;
            color: #1e293b;
        }
        
        .tool-desc {
            color: #64748b;
            font-size: 0.8rem;
            margin-top: 0.2rem;
        }
        
        /* 统计卡片 */
        .stat-card {
            background: white;
            padding: 1.2rem;
            border-radius: 16px;
            text-align: center;
            border: 1px solid #e2e8f0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        }
        
        .stat-value {
            font-size: 2rem;
            font-weight: 700;
            color: #1e293b;
            line-height: 1.2;
        }
        
        .stat-label {
            font-size: 0.8rem;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        /* 标签页样式 */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2rem;
            background: transparent;
            border-bottom: 1px solid #e2e8f0;
            padding: 0 1rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            font-size: 1rem;
            font-weight: 500;
            color: #64748b;
        }
        
        .stTabs [aria-selected="true"] {
            color: #2563eb !important;
        }
        
        /* 隐藏默认元素 */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* 空状态提示 */
        .empty-state {
            text-align: center;
            padding: 4rem 1rem;
            color: #94a3b8;
        }
        
        .empty-state h2 {
            font-weight: 300;
            font-size: 1.5rem;
            margin-bottom: 1rem;
        }
        
        /* 移除所有边框和圆角 */
        .stApp {
            background: white;
        }
        
        /* 移除无意义的白色横框 */
        .st-emotion-cache-1jzia57 {
            display: none;
        }
        
        hr {
            display: none;
        }
    </style>
    """, unsafe_allow_html=True)

    # 初始化session state
    def init_session_state():
        """初始化会话状态"""
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        if "conversation_history" not in st.session_state:
            st.session_state.conversation_history = []
        
        if "response_times" not in st.session_state:
            st.session_state.response_times = []
        
        if "total_queries" not in st.session_state:
            st.session_state.total_queries = 0

    # 侧边栏
    def render_sidebar():
        """渲染侧边栏"""
        with st.sidebar:
            st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
            
            # 工具列表
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.markdown('<div class="sidebar-section-title">🛠️ 可用工具</div>', unsafe_allow_html=True)
            
            tools = get_all_tools()
            for tool in tools:
                st.markdown(f"""
                <div class="tool-item">
                    <div class="tool-name">{tool.name}</div>
                    <div class="tool-desc">{tool.description}</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # 系统状态
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.markdown('<div class="sidebar-section-title">系统状态</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "查询数",
                    st.session_state.total_queries,
                    delta=None
                )
            with col2:
                if st.session_state.response_times:
                    avg_time = sum(st.session_state.response_times) / len(st.session_state.response_times)
                    st.metric(
                        "平均响应",
                        f"{avg_time:.1f}s",
                        delta=None
                    )
            st.markdown('</div>', unsafe_allow_html=True)
            
            # 清空对话按钮
            if st.button("🗑️ 清空对话", use_container_width=True):
                st.session_state.messages = []
                st.session_state.conversation_history = []
                st.session_state.total_queries = 0
                st.session_state.response_times = []
                st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)

    # 主聊天界面
    def render_chat_interface():
        """渲染主聊天界面"""
        
        # 聊天历史显示 - 无框线设计
        chat_container = st.container()
        
        with chat_container:
            st.markdown('<div class="main-content">', unsafe_allow_html=True)
            if not st.session_state.messages:
                # 空状态提示
                st.markdown("""
                <div class="empty-state">
                    <h2>👋 你好，我是智能助手</h2>
                    <p style="color: #94a3b8;">开始对话吧</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                for message in st.session_state.messages:
                    if message["role"] == "user":
                        st.markdown(f"""
                        <div class="chat-message">
                            <div class="user-section">
                                <div class="message-role">👤 你</div>
                                <div class="message-content">{message["content"]}</div>
                                <div class="time-tag">{message.get("timestamp", "")}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="chat-message">
                            <div class="assistant-section">
                                <div class="message-role">🤖 助手</div>
                                <div class="message-content">{message["content"]}</div>
                                <div class="time-tag">{message.get('time', 0):.1f}秒 · {message.get("timestamp", "")}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # 使用 Streamlit 内置聊天输入，自动固定底部并支持回车发送
        user_input = st.chat_input("输入您的问题...")
        
        return user_input

    # 统计信息页面
    def render_stats_page():
        """渲染统计信息页面"""
        st.markdown("## 使用统计")
        
        if not st.session_state.response_times:
            st.info("暂无统计数据，开始对话吧！")
            return
        
        # 统计卡片
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{st.session_state.total_queries}</div>
                <div class="stat-label">总查询数</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            avg_time = sum(st.session_state.response_times) / len(st.session_state.response_times)
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{avg_time:.1f}s</div>
                <div class="stat-label">平均响应</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            max_time = max(st.session_state.response_times)
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{max_time:.1f}s</div>
                <div class="stat-label">最长响应</div>
            </div>
            """, unsafe_allow_html=True)
        
        # 响应时间趋势图
        st.markdown("### 响应时间趋势")
        
        df = pd.DataFrame({
            "查询序号": list(range(1, len(st.session_state.response_times) + 1)),
            "响应时间(秒)": st.session_state.response_times
        })
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["查询序号"],
            y=df["响应时间(秒)"],
            mode='lines+markers',
            name='响应时间',
            line=dict(color="#32eb25", width=3),
            marker=dict(size=8, color="#32eb25")
        ))
        
        fig.update_layout(
            title=None,
            xaxis_title="查询次数",
            yaxis_title="响应时间 (秒)",
            hovermode='x',
            template="plotly_white",
            height=400,
            margin=dict(l=0, r=0, t=0, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 最近查询列表
        st.markdown("### 最近查询")
        
        recent_queries = []
        for i, msg in enumerate(reversed(st.session_state.messages)):
            if msg["role"] == "user":
                recent_queries.append({
                    "时间": msg.get("timestamp", "未知"),
                    "查询": msg["content"][:50] + "..." if len(msg["content"]) > 50 else msg["content"],
                    "响应": f"{msg.get('time', 0):.1f}s" if "time" in msg else "未知"
                })
            if len(recent_queries) >= 10:
                break
        
        if recent_queries:
            df_queries = pd.DataFrame(recent_queries)
            st.dataframe(
                df_queries,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "时间": st.column_config.TextColumn("时间", width="small"),
                    "查询": st.column_config.TextColumn("查询内容", width="large"),
                    "响应": st.column_config.TextColumn("响应时间", width="small")
                }
            )

    # 主函数
    def main():
        """主函数"""
        
        # 初始化session state
        init_session_state()
        
        # 渲染侧边栏
        render_sidebar()
        
        # 主内容区域
        tab1, tab2 = st.tabs(["对话", "统计"])
        
        with tab1:
            user_input = render_chat_interface()
            
            # 处理用户输入
            if user_input:
                # 添加用户消息
                st.session_state.messages.append({
                    "role": "user",
                    "content": user_input,
                    "timestamp": datetime.now().strftime("%H:%M")
                })
                
                # 显示加载提示
                with st.spinner("模型思考中..."):
                    start_time = time.time()
                    
                    try:
                        response = run_agent(user_input)
                        
                        elapsed_time = time.time() - start_time
                        
                        st.session_state.total_queries += 1
                        st.session_state.response_times.append(elapsed_time)
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response,
                            "time": elapsed_time,
                            "timestamp": datetime.now().strftime("%H:%M")
                        })
                        
                    except Exception as e:
                        error_msg = f"抱歉，处理您的请求时出现错误：{str(e)}"
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_msg,
                            "time": time.time() - start_time,
                            "timestamp": datetime.now().strftime("%H:%M")
                        })
                
                st.rerun()
        
        with tab2:
            render_stats_page()

    # 运行主函数
    if __name__ == "__main__":
        main()