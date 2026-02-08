from langgraph.graph import StateGraph, END

# 创建图
workflow = StateGraph(dict)

# 定义节点
def 接收用户输入(state):
    """接收用户输入"""
    user_input = "help"  # 模拟用户输入
    state["用户输入"] = user_input
    print(f"用户说: {user_input}")
    return state

def 分析意图(state):
    """分析用户想干什么"""
    if "help" in state["用户输入"]:
        state["需要帮助"] = True
    else:
        state["需要帮助"] = False
    return state

def 提供帮助(state):
    """提供帮助"""
    print("正在提供帮助...")
    state["响应"] = "这是帮助内容"
    return state

def 正常回复(state):
    """正常回复"""
    print("正常回复中...")
    state["响应"] = "这是正常回复"
    return state

def 生成最终结果(state):
    """最终处理"""
    print("生成最终结果")
    state["最终输出"] = f"处理完成，响应是: {state['响应']}"
    return state

# 路由函数 - 关键！
def 判断是否需要帮助(state):
    """决定下一步去哪"""
    if state.get("需要帮助"):
        return "去帮助"  # 返回目标节点名称
    else:
        return "去正常回复"

# 添加节点
workflow.add_node("接收", 接收用户输入)
workflow.add_node("分析", 分析意图)
workflow.add_node("帮助", 提供帮助)
workflow.add_node("正常", 正常回复)
workflow.add_node("最终", 生成最终结果)

# 设置起点
workflow.set_entry_point("接收")

# 添加固定连接
workflow.add_edge("接收", "分析")  # 接收 → 分析

# 添加条件分支 - 关键！
workflow.add_conditional_edges(
    "分析",  # 从哪个节点开始分支
    判断是否需要帮助,  # 用哪个函数决定方向
    {
        "去帮助": "帮助",      # 如果函数返回"去帮助"，就去"帮助"节点
        "去正常回复": "正常"   # 如果函数返回"去正常回复"，就去"正常"节点
    }
)

# 合并分支
workflow.add_edge("帮助", "最终")    # 帮助 → 最终
workflow.add_edge("正常", "最终")    # 正常 → 最终
workflow.add_edge("最终", END)      # 最终 → 结束

# 编译并运行
app = workflow.compile()

print("\n=== 测试1: 用户需要帮助 ===")
最终状态1 = app.invoke({"用户输入": "help"})
print(f"最终结果: {最终状态1['最终输出']}")

print("\n=== 测试2: 用户正常提问 ===")
最终状态2 = app.invoke({"用户输入": "你好"})
print(f"最终结果: {最终状态2['最终输出']}")