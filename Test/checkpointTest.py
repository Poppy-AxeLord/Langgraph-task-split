import sqlite3
import random
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, END
from pydantic import BaseModel


# 1. 初始化SQLite检查点（复用）
def init_sqlite_checkpoint():
    conn = sqlite3.connect("langgraph_checkpoints.db", check_same_thread=False)
    checkpoint = SqliteSaver(conn)
    checkpoint.setup()
    return checkpoint


# 2. 状态结构：仅保留count和name（双字段）
class SimpleState(BaseModel):
    count: int = 0  # 自增计数器
    name: str = ""  # 随机人名


# 人名池
NAME_POOL = ["张三", "李四", "王五", "赵六", "钱七", "孙八", "周九", "吴十"]


# 3. 核心节点：count+1 + 随机人名
def add_one_and_random_name(state: SimpleState):
    if isinstance(state, dict):
        state = SimpleState(**state)

    new_count = state.count + 1
    new_name = random.choice(NAME_POOL)

    print(f"🔢 自增count: {state.count} → {new_count} | 📛 随机人名: {new_name}")
    return {"count": new_count, "name": new_name}


# 4. 构建图
def build_graph(checkpoint):
    graph_builder = StateGraph(SimpleState)
    graph_builder.add_node("add_one_and_random_name", add_one_and_random_name)
    graph_builder.set_entry_point("add_one_and_random_name")
    graph_builder.add_edge("add_one_and_random_name", END)

    compiled_graph = graph_builder.compile(
        checkpointer=checkpoint,
        interrupt_after=["add_one_and_random_name"]
    )
    print("✅ 图编译成功（已绑定检查点）")
    return compiled_graph


# 5. 断点续跑核心逻辑（完全通过LangGraph API验证，不手动查库）
if __name__ == "__main__":
    # 初始化
    checkpoint = init_sqlite_checkpoint()
    graph = build_graph(checkpoint)
    config = {"configurable": {"thread_id": "random_name_test_001"}}

    # 第一步：检查断点（仅用LangGraph API）
    print("\n===== 第一步：读取断点状态（LangGraph API） =====")
    saved_state = graph.get_state(config)

    if saved_state and saved_state.values:
        current_count = saved_state.values.get("count", 0)
        current_name = saved_state.values.get("name", "")
        current_state = SimpleState(count=current_count, name=current_name)
        print(f"🔍 读取到断点：count={current_count} | 上一次人名={current_name}")
    else:
        current_state = SimpleState(count=0, name="")
        print(f"🔍 无断点，初始状态：count=0 | name=''")

    # 第二步：运行并写入检查点
    print("\n===== 第二步：运行并写入检查点 =====")
    result = graph.invoke(current_state, config=config)
    print(f"✅ 本次运行结果：count={result['count']} | 本次人名={result['name']}")

    # 第三步：验证最新状态（仅用LangGraph API）
    print("\n===== 第三步：验证最新断点状态 =====")
    final_state = graph.get_state(config)
    if final_state:
        final_count = final_state.values.get("count", 0)
        final_name = final_state.values.get("name", "")
        print(f"✅ 最新断点状态（已存入数据库）：")
        print(f"   → count: {final_count}")
        print(f"   → name: {final_name}")
    else:
        print("❌ 未找到最新状态")

    # （可选）验证：连续运行3次，看断点是否持续生效
    print("\n===== 可选：连续续跑验证 =====")
    for i in range(2):
        print(f"\n--- 续跑第{i + 1}次 ---")
        latest_state = graph.get_state(config)
        run_state = SimpleState(**latest_state.values) if latest_state else SimpleState()
        run_result = graph.invoke(run_state, config=config)
        print(f"续跑结果：count={run_result['count']} | name={run_result['name']}")


    # 查看最后的数据
    # final_state = graph.get_state(config)
    # print(f"数据库中最后的数据：{final_state.values}")

    # 关闭连接
    checkpoint.conn.close()