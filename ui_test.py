import os
import time
import hashlib
from typing import TypedDict, List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI

# 引入 LangGraph 核心组件
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

# ----- 1. 环境配置 -----
load_dotenv()
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE_URL"),
)

# ----- 2. 基础状态定义 (保持原有结构) -----
class AgentState(TypedDict):
    """简化版工作流状态容器"""
    original_task: str                    # 原始任务
    subtasks: List[Dict[str, Any]]        # 子任务列表
    reasoning: str                        # 思考过程
    current_step: str                     # 当前步骤
    complexity: str                       # 'simple' 或 'complex'
    needs_deep_analysis: bool             # 是否需要深度分析
    # 循环控制字段
    current_decomposition_level: int      # 当前正在拆解的层级
    max_decomposition_level: int          # 最大拆解层级
    needs_further_decomposition: bool     # 是否需要进一步拆解
    processing_queue: List[Dict[str, Any]] # 待处理任务队列

# ----- 3. 核心工具函数 (保持原有逻辑) -----
def call_qwen(prompt: str, model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")) -> str:
    """调用LLM模型（流式输出）"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            stream=True
        )

        full_response = ""
        for chunk in response:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                full_response += content
                time.sleep(0.01)
        
        print()
        return full_response.strip()
        
    except Exception as e:
        error_msg = f"LLM调用失败: {str(e)}"
        print(error_msg)
        return error_msg

def is_task_actionable(task_description: str) -> bool:
    """判断任务是否足够具体可执行"""
    prompt = f"""判断以下任务是否已经足够具体、可直接执行：
    
任务描述: "{task_description}"

判断标准：
- 可执行：一个明确的动作，有清晰的产出，单人1-3天内可完成
- 需拆解：范围较广，需要进一步分解

请只回复一个字：'是' 或 '否'"""
    
    response = call_qwen(prompt).strip()
    return '是' in response

# ----- 4. LangGraph 节点函数（全部返回AgentState） -----
def input_task_node(state: AgentState) -> AgentState:
    """初始化任务 - LangGraph节点"""
    print(f"\n📥 [节点：接收任务]")
    print(f"   原始任务: {state['original_task']}")
    
    state['subtasks'] = []
    state['current_step'] = "任务已接收"
    state['current_decomposition_level'] = 0
    state['max_decomposition_level'] = 2  # 最大拆解2层
    state['needs_further_decomposition'] = True
    
    # 初始化待处理队列
    state['processing_queue'] = [{
        "task": state['original_task'],
        "level": 0,
        "parent_id": None
    }]
    
    return state

def assess_complexity_node(state: AgentState) -> AgentState:
    """评估任务复杂度 - LangGraph节点"""
    print(f"\n🔍 [节点：复杂度评估] 正在分析任务复杂度...")
    
    current_task = state['processing_queue'][0]["task"] if state['processing_queue'] else state['original_task']
    
    prompt = f"""请判断以下任务的复杂程度（简单/复杂）。

任务："{current_task}"

判断标准：
- 简单：目标明确、范围小、常规操作
- 复杂：涉及多因素、多步骤、需要策略思考

只需回复一个字：'简单' 或 '复杂'"""
    
    response = call_qwen(prompt).strip()
    
    if '复杂' in response:
        state['complexity'] = 'complex'
        state['needs_deep_analysis'] = True
        print(f"   ✅ 评估结果：复杂任务")
    else:
        state['complexity'] = 'simple'
        state['needs_deep_analysis'] = False
        print(f"   ✅ 评估结果：简单任务")
    
    state['current_step'] = f"复杂度评估完成: {state['complexity']}"
    return state

def deep_analysis_node(state: AgentState) -> AgentState:
    """深度分析（仅复杂任务） - LangGraph节点"""
    if not state['needs_deep_analysis']:
        return state
    
    current_task = state['processing_queue'][0]["task"] if state['processing_queue'] else state['original_task']
    
    print(f"\n🧠 [节点：深度分析] 对复杂任务进行深度分析...")
    
    prompt = f"""对以下复杂任务进行深度分析：

任务：{current_task}

请分析：
1. 核心目标与成功标准
2. 主要挑战和风险
3.关键依赖关系和资源需求
4. 建议的拆解方向

用简洁的要点格式回答。"""
    
    print("   💭 深度分析内容: ", end="", flush=True)
    analysis = call_qwen(prompt)
    
    state['reasoning'] = f"\n【深度分析】\n{analysis}\n"
    state['current_step'] = "深度分析完成"
    return state

def decompose_task_node(state: AgentState) -> AgentState:
    """核心拆解逻辑 - LangGraph节点"""
    if not state['processing_queue']:
        state['needs_further_decomposition'] = False
        return state
    
    # 获取当前待拆解任务
    current_item = state['processing_queue'].pop(0)
    current_task = current_item["task"]
    parent_level = current_item["level"]
    child_level = parent_level + 1
    
    state['current_decomposition_level'] = parent_level
    
    print(f"\n✂️ [节点：拆解任务]")
    print(f"   📝 父任务 (层级{parent_level+1}): {current_task}")
    print(f"   📊 将拆解为层级{child_level+1}的子任务")
    
    # 根据复杂度决定拆解数量
    subtask_count = 3 if state['complexity'] == 'complex' else 2
    
    prompt = f"""将以下任务拆解为{subtask_count}个更具体的子任务。

父任务 (层级{parent_level+1}): {current_task}

请列出{subtask_count}个具体的子任务，每个用数字编号。
要求：每个子任务应该比父任务更具体、更可执行。
子任务将属于层级{child_level+1}。"""
    
    print(f"   📋 生成层级{child_level+1}的子任务: ", end="", flush=True)
    response = call_qwen(prompt)
    
    # 解析拆解结果
    new_subtasks = []
    lines = response.split('\n')
    
    for line in lines:
        line = line.strip()
        if len(line) < 2:
            continue
            
        first_char = line[0]
        if first_char.isdigit():
            end_pos = 1
            while end_pos < len(line) and line[end_pos].isdigit():
                end_pos += 1
                
            if end_pos < len(line) and line[end_pos] in ['.', ')', '、', ' ']:
                task_content = line[end_pos+1:].strip()
                if task_content:
                    new_subtasks.append(task_content)
    
    # 确保拆解数量符合要求
    new_subtasks = new_subtasks[:subtask_count]
    while len(new_subtasks) < subtask_count:
        new_subtasks.append(f"子任务{len(new_subtasks)+1}")
    
    # 处理拆解后的子任务
    level_tasks_count = len([t for t in state['subtasks'] if t.get("level") == child_level + 1])
    
    for i, subtask in enumerate(new_subtasks, 1):
        task_num = level_tasks_count + i
        task_id = f"L{child_level+1}_T{task_num}"
        
        task_info = {
            "id": task_id,
            "description": subtask,
            "level": child_level + 1,
            "parent_task": current_task,
            "parent_level": parent_level + 1,
            "actionable": False,
            "status": "pending"
        }
        
        # 判断是否可执行
        if child_level + 1 >= state['max_decomposition_level']:
            task_info["actionable"] = True
            task_info["status"] = "ready"
        else:
            task_info["actionable"] = is_task_actionable(subtask)
            if not task_info["actionable"]:
                # 不可执行则加入待处理队列继续拆解
                state['processing_queue'].append({
                    "task": subtask,
                    "level": child_level,
                    "parent_id": task_id
                })
        
        state['subtasks'].append(task_info)
    
    print(f"   ✅ 成功拆解出{len(new_subtasks)}个层级{child_level+1}的子任务")
    print(f"   📊 待处理队列: {len(state['processing_queue'])}个任务")
    
    return state

def evaluate_decomposition_status_node(state: AgentState) -> AgentState:
    """评估拆解状态 - LangGraph节点（仅更新状态，不返回字符串）"""
    print(f"\n📊 [节点：状态评估] 检查拆解进度...")
    
    # 达到最大层级则停止
    if state['current_decomposition_level'] + 1 >= state['max_decomposition_level']:
        print(f"   ⏹️ 已达到最大拆解层级 ({state['max_decomposition_level']})")
        state['needs_further_decomposition'] = False
        state['current_step'] = f"拆解完成，达到最大层级{state['max_decomposition_level']}"
    # 没有待处理任务则停止
    elif not state['processing_queue']:
        print(f"   ✅ 所有任务已拆解完毕")
        state['needs_further_decomposition'] = False
        state['current_step'] = f"拆解完成，当前层级{state['current_decomposition_level'] + 1}"
    # 继续拆解
    else:
        next_item = state['processing_queue'][0]
        next_task = next_item["task"]
        next_level = next_item["level"]
        
        print(f"   🔄 下一个任务: 层级{next_level+1} - '{next_task[:50]}...'")
        state['needs_further_decomposition'] = True
        state['current_step'] = f"状态评估完成，继续拆解"
    
    # 必须返回状态字典（核心修复点）
    return state

# 单独的条件判断函数
def decide_next_step(state: AgentState) -> str:
    """条件判断函数：根据状态返回下一个节点名称"""
    if state['needs_further_decomposition']:
        return "拆解任务" 
    else:
        return "输出最终结果"  

def output_final_results_node(state: AgentState) -> AgentState:
    """生成最终报告 - LangGraph节点（输出Markdown格式）"""
    print(f"\n📤 [节点：最终输出] 生成拆解报告...")
    
    total_tasks = len(state['subtasks'])
    actionable_tasks = len([t for t in state['subtasks'] if t["actionable"]])
    max_level = max([t["level"] for t in state['subtasks']]) if state['subtasks'] else 0
    
    # 构建美观的Markdown报告
    report = f"""# 智能任务拆解报告
> 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}

## 🎯 任务概览
| 项目 | 内容 |
|------|------|
| 原始任务 | {state['original_task']} |
| 任务复杂度 | {state['complexity']} |
| 拆解层级 | {max_level} 层 |
| 生成子任务总数 | {total_tasks} 个 |
| 可执行子任务数 | {actionable_tasks} 个 |

## 📋 分层拆解详情
"""
    
    # 按层级展示子任务（美化版）
    for level in range(1, max_level + 1):
        level_tasks = [t for t in state['subtasks'] if t["level"] == level]
        if level_tasks:
            actionable_in_level = len([t for t in level_tasks if t["actionable"]])
            
            report += f"\n### 第 {level} 层拆解\n"
            report += f"> 共 {len(level_tasks)} 个任务，其中 {actionable_in_level} 个可直接执行\n\n"
            
            # 使用表格展示层级任务
            report += "| 任务ID | 任务描述 | 状态 | 父任务 |\n"
            report += "|--------|----------|------|--------|\n"
            
            for task in level_tasks:
                status_text = "✅ 可执行" if task["actionable"] else "⏳ 需拆解"
                parent_short = task['parent_task'][:50] + "..." if len(task['parent_task']) > 50 else task['parent_task']
                report += f"| {task['id']} | {task['description']} | {status_text} | {parent_short} |\n"
    
    # 可执行任务清单（美化版）
    actionable_list = [t for t in state['subtasks'] if t["actionable"]]
    if actionable_list:
        report += f"\n## 🚀 可立即执行的任务清单\n\n"
        for i, task in enumerate(actionable_list, 1):
            report += f"{i}. **[{task['id']}]** {task['description']}\n"
    
    # 添加思考过程（如果有）
    if state['reasoning']:
        report += f"\n## 🧠 深度分析思考\n\n{state['reasoning']}\n"
    
    # 添加页脚
    report += f"""
---
*本报告由智能任务拆解系统自动生成 | LangGraph 驱动*
"""
    
    # 打印美化后的报告（简化版）
    print("\n" + "="*80)
    print("📄 任务拆解报告（Markdown格式）")
    print("="*80)
    print(report)
    
    # 保存为Markdown文件
    try:
        task_hash = hashlib.md5(state['original_task'].encode()).hexdigest()[:8]
        filename = f"task_report_{task_hash}.md"  # 修改为md后缀
        with open(filename, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n💾 美化版报告已保存到: {filename}")
    except Exception as e:
        print(f"⚠️  保存报告时出错: {e}")
    
    return state

# ----- 5. 构建 LangGraph 工作流 -----
def build_task_decomposition_graph() -> CompiledStateGraph:
    """构建任务拆解的LangGraph工作流"""
    # 1. 创建状态图
    graph = StateGraph(AgentState)
    
    # 2. 添加节点
    graph.add_node("接收任务", input_task_node)
    graph.add_node("评估复杂度", assess_complexity_node)
    graph.add_node("深度分析", deep_analysis_node)
    graph.add_node("拆解任务", decompose_task_node)
    graph.add_node("评估拆解状态", evaluate_decomposition_status_node)
    graph.add_node("输出最终结果", output_final_results_node)
    
    # 3. 设置入口点
    graph.set_entry_point("接收任务")
    
    # 4. 添加普通边
    graph.add_edge("接收任务", "评估复杂度")
    graph.add_edge("评估复杂度", "深度分析")
    graph.add_edge("深度分析", "拆解任务")
    graph.add_edge("拆解任务", "评估拆解状态")
    
    # 5. 添加条件边
    graph.add_conditional_edges(
        "评估拆解状态",  # 源节点
        decide_next_step,                 # 独立的条件判断函数（不加入节点列表）
        {
            "拆解任务": "拆解任务",          # 继续拆解
            "输出最终结果": "输出最终结果" # 输出结果
        }
    )
    
    # 6. 设置结束点
    graph.add_edge("输出最终结果", END)

    compiled_graph = graph.compile()
    return compiled_graph

# ----- 6. 主执行函数 -----
def run_task_decomposition(task_description: str):
    """运行LangGraph版任务拆解"""
    print("="*80)
    print(f"🚀 LangGraph版任务拆解程序")
    print("="*80)
    
    # 初始化状态
    initial_state = AgentState(
        original_task=task_description,
        subtasks=[],
        reasoning="",
        current_step="开始",
        complexity="unknown",
        needs_deep_analysis=False,
        current_decomposition_level=0,
        max_decomposition_level=2,
        needs_further_decomposition=True,
        processing_queue=[]
    )
    
    # 构建并运行图
    graph = build_task_decomposition_graph()
    result = graph.invoke(initial_state)
    
    print(f"\n✅ 任务拆解完成！")
    return result

# ----- 7. 运行测试 -----
if __name__ == "__main__":
    # 测试任务
    test_task = "制定双11市场营销战略"
    
    # 执行拆解
    result = run_task_decomposition(test_task)