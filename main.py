# cyclic_agent_human_loop_final.py
import os
import time
import sqlite3
import hashlib
from typing import TypedDict, List, Dict, Any, Optional
from dotenv import load_dotenv
from openai import OpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from persistence import PersistenceManager

# ----- 1. 环境配置 -----
load_dotenv()
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE_URL"),
)

# ----- 2. 基础状态定义 -----
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
    # 新增：持久化相关字段
    execution_id: str                     # 执行会话ID
    created_at: str                       # 创建时间
    resumed_count: int                    # 恢复次数统计
    estimated_cost: float                 # 估算总成本
    needs_human_approval: bool            # 人工审批
    approval_status: Optional[str]        # 审批状态
    approval_notes: Optional[str]         # 审批意见/驳回原因
    approved_by: Optional[str]            # 审批人姓名/ID

# ----- 4. 工具函数 -----
def call_qwen(prompt: str, model=os.getenv("OPENAI_API_MODEL")) -> str:
    """调用千问模型（流式输出）"""
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
        return full_response
        
    except Exception as e:
        return f"LLM调用失败: {str(e)}"

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

def generate_task_hash(task_description: str) -> str:
    """生成任务唯一哈希"""
    return hashlib.md5(task_description.encode()).hexdigest()[:8]

def calculate_estimated_cost(state: AgentState) -> float:
    """估算任务执行成本（模拟）"""
    base_cost = 0
    
    if state['complexity'] == 'complex':
        base_cost += 3000
    else:
        base_cost += 1000
    
    task_count = len(state['subtasks'])
    base_cost += task_count * 800
    
    if state.get('needs_deep_analysis', False):
        base_cost += 2000
    
    import random
    variance = random.uniform(0.8, 1.2)
    estimated = base_cost * variance
    
    return round(estimated, 2)

# ----- 5. 节点定义 -----
def input_task(state: AgentState) -> AgentState:
    """节点1：接收任务并初始化"""
    print(f"\n📥 [节点：接收任务]")
    print(f"   原始任务: {state['original_task']}")
    print(f"   执行ID: {state['execution_id']}")
    
    if state.get('resumed_count', 0) > 0:
        print(f"   🔄 第{state['resumed_count']}次恢复执行")
    
    state['subtasks'] = []
    state['current_step'] = "任务已接收"
    state['current_decomposition_level'] = 0
    state['max_decomposition_level'] = 2
    state['needs_further_decomposition'] = True
    state['needs_human_approval'] = False
    state['approval_status'] = None
    state['approval_notes'] = None
    state['approved_by'] = None
    state['estimated_cost'] = 0
    
    state['processing_queue'] = [{
        "task": state['original_task'],
        "level": 0,
        "parent_id": None
    }]
    
    return state

def assess_complexity(state: AgentState) -> AgentState:
    """节点2：评估任务复杂度"""
    print(f"\n🔍 [节点：复杂度评估] 正在分析任务复杂度...")
    
    if state['processing_queue']:
        current_item = state['processing_queue'][0]
        current_task = current_item["task"]
    else:
        current_task = state['original_task']
    
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

def deep_analysis(state: AgentState) -> AgentState:
    """节点3：深度分析（仅复杂任务使用）"""
    if not state['needs_deep_analysis']:
        return state
    
    if state['processing_queue']:
        current_item = state['processing_queue'][0]
        current_task = current_item["task"]
    else:
        current_task = state['original_task']
    
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
    
    if 'reasoning' not in state or not state['reasoning']:
        state['reasoning'] = ""
    
    level = state['current_decomposition_level']
    state['reasoning'] += f"\n【深度分析 - 层级{level+1}】\n{analysis}\n"
    state['current_step'] = "深度分析完成"
    return state

def decompose_task(state: AgentState) -> AgentState:
    """节点4：拆解单个任务"""
    if not state['processing_queue']:
        state['needs_further_decomposition'] = False
        return state
    
    current_item = state['processing_queue'].pop(0)
    current_task = current_item["task"]
    parent_level = current_item["level"]
    child_level = parent_level + 1
    
    state['current_decomposition_level'] = parent_level
    
    print(f"\n✂️ [节点：拆解任务]")
    print(f"   📝 父任务 (层级{parent_level+1}): {current_task}")
    print(f"   📊 将拆解为层级{child_level+1}的子任务")
    
    if state['complexity'] == 'complex':
        subtask_count = 3
    else:
        subtask_count = 2
    
    prompt = f"""将以下任务拆解为{subtask_count}个更具体的子任务。

父任务 (层级{parent_level+1}): {current_task}

请列出{subtask_count}个具体的子任务，每个用数字编号。
要求：每个子任务应该比父任务更具体、更可执行。
子任务将属于层级{child_level+1}。"""
    
    print(f"   📋 生成层级{child_level+1}的子任务: ", end="", flush=True)
    response = call_qwen(prompt)
    
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
    
    new_subtasks = new_subtasks[:subtask_count]
    while len(new_subtasks) < subtask_count:
        new_subtasks.append(f"子任务{len(new_subtasks)+1}")
    
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
            "status": "pending",
            "estimated_cost": 0
        }
        
        is_actionable = False
        if child_level + 1 >= state['max_decomposition_level']:
            is_actionable = True
        else:
            is_actionable = is_task_actionable(subtask)
        
        if is_actionable:
            task_info["actionable"] = True
            task_info["status"] = "ready"
            task_info["estimated_cost"] = 500 + (i * 100)
        else:
            task_info["actionable"] = False
            state['processing_queue'].append({
                "task": subtask,
                "level": child_level,
                "parent_id": task_id
            })
        
        state['subtasks'].append(task_info)
    
    print(f"   ✅ 成功拆解出{len(new_subtasks)}个层级{child_level+1}的子任务")
    print(f"   📊 待处理队列: {len(state['processing_queue'])}个任务")
    
    return state

def evaluate_decomposition_status(state: AgentState) -> AgentState:
    """节点5：评估拆解状态"""
    print(f"\n📊 [节点：状态评估] 检查拆解进度...")
    
    if state['current_decomposition_level'] + 1 >= state['max_decomposition_level']:
        print(f"   ⏹️ 已达到最大拆解层级 ({state['max_decomposition_level']})")
        state['needs_further_decomposition'] = False
        state['current_step'] = f"拆解完成，达到最大层级{state['max_decomposition_level']}"
        return state
    
    if not state['processing_queue']:
        print(f"   ✅ 所有任务已拆解完毕")
        state['needs_further_decomposition'] = False
        state['current_step'] = f"拆解完成，当前层级{state['current_decomposition_level'] + 1}"
        return state
    
    next_item = state['processing_queue'][0]
    next_task = next_item["task"]
    next_level = next_item["level"]
    
    print(f"   🔄 下一个任务: 层级{next_level+1} - '{next_task[:50]}...'")
    
    if next_level + 1 < state['max_decomposition_level']:
        state['needs_further_decomposition'] = True
        print(f"   🔄 需要进一步拆解到层级{next_level+2}")
    else:
        state['needs_further_decomposition'] = False
        print(f"   ⏹️ 已达到最大层级，停止拆解")
    
    state['current_step'] = f"状态评估完成，继续拆解: {state['needs_further_decomposition']}"
    return state

def budget_approval_checkpoint(state: AgentState) -> AgentState:
    """节点6：预算审批检查点"""
    print(f"\n💰 [节点：预算审批] 检查是否需要人工审批...")
    
    state['estimated_cost'] = calculate_estimated_cost(state)
    print(f"   📈 总估算成本: ${state['estimated_cost']}")
    
    APPROVAL_THRESHOLD = 10000
    
    if state['estimated_cost'] > APPROVAL_THRESHOLD:
        print(f"   ⚠️  超过审批阈值 (${APPROVAL_THRESHOLD})")
        print(f"   🛑 需要人工审批")
        print(f"\n   📋 审批详情:")
        print(f"      - 原始任务: {state['original_task']}")
        print(f"      - 生成子任务: {len(state['subtasks'])}个")
        print(f"      - 可执行任务: {len([t for t in state['subtasks'] if t['actionable']])}个")
        print(f"      - 估算成本: ${state['estimated_cost']}")
        
        state['needs_human_approval'] = True
        state['approval_status'] = 'pending'
        state['current_step'] = f"等待人工审批 (成本: ${state['estimated_cost']})"
        
        raise Interrupt({
            "type": "BUDGET_APPROVAL",
            "estimated_cost": state['estimated_cost'],
            "task_id": state['execution_id']
        })
    else:
        print(f"   ✅ 成本在预算内，无需审批")
        state['needs_human_approval'] = False
        state['approval_status'] = 'auto_approved'
        state['current_step'] = f"自动批准 (成本: ${state['estimated_cost']})"
    
    return state

def process_approval_decision(state: AgentState) -> AgentState:
    """节点7：处理审批决定"""
    print(f"\n📝 [节点：处理审批] 应用审批决定...")
    
    if state['approval_status'] in ['approved', 'auto_approved']:
        print(f"   ✅ 审批状态: {state['approval_status']}")
        
        if state['approval_status'] == 'approved':
            print(f"   审批人: {state.get('approved_by', '未知')}")
            if state.get('approval_notes'):
                print(f"   审批意见: {state['approval_notes']}")
        
        state['current_step'] = "审批通过，继续执行"
        return state
    
    elif state['approval_status'] == 'rejected':
        print(f"   ❌ 审批驳回")
        print(f"   驳回原因: {state.get('approval_notes', '未提供原因')}")
        
        for task in state['subtasks']:
            task['status'] = 'cancelled'
        
        state['current_step'] = "任务已取消（审批驳回）"
        state['needs_further_decomposition'] = False
        
        return state
    
    state['current_step'] = "审批状态未知"
    return state

def output_final_results(state: AgentState) -> AgentState:
    """节点8：生成最终报告"""
    print(f"\n📤 [节点：最终输出] 生成拆解报告...")
    
    total_tasks = len(state['subtasks'])
    actionable_tasks = len([t for t in state['subtasks'] if t["actionable"]])
    max_level = max([t["level"] for t in state['subtasks']]) if state['subtasks'] else 0
    
    report = f"""
{'='*70}
📋 智能任务拆解报告（人机协同版）
{'='*70}

🎯 原始任务：{state['original_task']}
🔧 执行ID：{state['execution_id']}
📊 任务复杂度：{state['complexity']}
💰 估算成本：${state['estimated_cost']}
🔄 完成拆解层级：{max_level} 层
📈 生成子任务：{total_tasks} 个（{actionable_tasks}个可执行）
🔄 恢复次数：{state.get('resumed_count', 0)} 次
📝 审批状态：{state['approval_status']}

"""
    
    if state['approved_by']:
        report += f"   审批人：{state['approved_by']}\n"
    if state['approval_notes']:
        report += f"   审批意见：{state['approval_notes']}\n"
    
    for level in range(1, max_level + 1):
        level_tasks = [t for t in state['subtasks'] if t["level"] == level]
        if level_tasks:
            actionable_in_level = len([t for t in level_tasks if t["actionable"]])
            
            report += f"\n📁 第{level}层拆解 ({len(level_tasks)}个任务, {actionable_in_level}个可执行):\n"
            
            for task in level_tasks:
                status_icon = "✅" if task["actionable"] and task.get("status") != "cancelled" else "⏳"
                if task.get("status") == "cancelled":
                    status_icon = "❌"
                
                cost_info = f" (${task.get('estimated_cost', 0)})" if task.get('estimated_cost') else ""
                report += f"   {status_icon} {task['id']}: {task['description']}{cost_info}\n"
                
                if level > 1 and 'parent_task' in task:
                    parent_short = task['parent_task'][:40] + "..." if len(task['parent_task']) > 40 else task['parent_task']
                    report += f"      ← 来自: {parent_short}\n"
    
    actionable_list = [t for t in state['subtasks'] if t["actionable"] and t.get("status") != "cancelled"]
    if actionable_list:
        total_cost = sum([t.get('estimated_cost', 0) for t in actionable_list])
        report += f"\n🚀 可立即执行的任务清单 ({len(actionable_list)}个, 总成本: ${total_cost}):\n"
        for i, task in enumerate(actionable_list, 1):
            cost_info = f" [${task.get('estimated_cost', 0)}]" if task.get('estimated_cost') else ""
            report += f"   {i}. [{task['id']}] {task['description']}{cost_info}\n"
    
    report += f"\n📊 执行统计："
    report += f"\n   • 总循环次数：{state['current_decomposition_level'] + 1}"
    report += f"\n   • 最终可执行任务：{actionable_tasks}/{total_tasks}"
    report += f"\n   • 最大拆解深度：{max_level}层"
    report += f"\n   • 恢复次数：{state.get('resumed_count', 0)}次"
    
    if state.get('reasoning', ''):
        report += f"\n\n💡 深度分析摘要："
        report += f"\n{state['reasoning'][:300]}..." if len(state['reasoning']) > 300 else f"\n{state['reasoning']}"
    
    report += f"\n\n{'='*70}"
    print(report)
    
    try:
        filename = f"task_report_{state['execution_id']}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"💾 详细报告已保存到: {filename}")
    except Exception as e:
        print(f"⚠️  保存报告时出错: {e}")
    
    return state

# ----- 6. 路由函数 -----
def route_by_complexity(state: AgentState) -> str:
    return

def should_continue_decomposition(state: AgentState) -> str:
    if state['needs_further_decomposition']:
        return "继续拆解"
    else:
        return "预算审批"

def route_by_approval_status(state: AgentState) -> str:
    if state['approval_status'] == 'pending':
        return "等待审批"
    elif state['approval_status'] in ['approved', 'auto_approved']:
        return "处理审批"
    elif state['approval_status'] == 'rejected':
        return "最终输出"
    else:
        return "处理审批"

# ----- 7. 构建工作流图 -----
def create_workflow():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("接收任务", input_task)
    workflow.add_node("评估复杂度", assess_complexity)
    workflow.add_node("深度分析", deep_analysis)
    workflow.add_node("拆解任务", decompose_task)
    workflow.add_node("状态评估", evaluate_decomposition_status)
    workflow.add_node("预算审批", budget_approval_checkpoint)
    workflow.add_node("处理审批", process_approval_decision)
    workflow.add_node("最终输出", output_final_results)
    
    workflow.set_entry_point("接收任务")
    workflow.add_edge("接收任务", "评估复杂度")
    
    workflow.add_conditional_edges(
        "评估复杂度",
        route_by_complexity,
        {
            "深度分析": "深度分析",
            "拆解任务": "拆解任务"
        }
    )
    
    workflow.add_edge("深度分析", "拆解任务")
    workflow.add_edge("拆解任务", "状态评估")
    
    workflow.add_conditional_edges(
        "状态评估",
        should_continue_decomposition,
        {
            "继续拆解": "拆解任务",
            "预算审批": "预算审批"
        }
    )
    
    workflow.add_conditional_edges(
        "预算审批",
        route_by_approval_status,
        {
            "等待审批": "预算审批",
            "处理审批": "处理审批",
            "最终输出": "最终输出"
        }
    )
    
    workflow.add_edge("处理审批", "最终输出")
    workflow.add_edge("最终输出", END)
    
    if persistence_manager.is_enabled():
        return workflow.compile(
            checkpointer=persistence_manager.checkpointer,
            interrupt_before=["预算审批"]
        )
    else:
        return workflow.compile()

# ----- 8. 主执行程序 -----
def run_agent_with_approval(user_id: str, task_description: str, resume: bool = False, 
                           approval_decision: Optional[str] = None,
                           approval_notes: Optional[str] = None,
                           approved_by: Optional[str] = None):
    """运行带审批的Agent"""
    
    task_hash = generate_task_hash(task_description)
    config = persistence_manager.create_thread_config(user_id, task_hash)
    
    print("\n" + "="*70)
    print(f"🔄 智能任务拆解Agent - 人机协同版")
    print(f"👤 用户: {user_id} | 📝 任务哈希: {task_hash}")
    if resume:
        print(f"⏳ 模式: 断点续生")
        if approval_decision:
            print(f"📝 携带审批决定: {approval_decision}")
    else:
        print(f"🚀 模式: 全新执行")
    print("="*70)
    
    graph = create_workflow()
    
    if resume:
        print(f"\n🔍 正在检查断点状态...")
        saved_state = graph.get_state(config)
        
        if saved_state and saved_state.next:
            print(f"✅ 找到断点，从上次中断处继续")
            print(f"   已生成子任务: {len(saved_state.values.get('subtasks', []))}个")
            print(f"   估算成本: ${saved_state.values.get('estimated_cost', 0)}")
            print(f"   审批状态: {saved_state.values.get('approval_status', 'N/A')}")
            
            initial_state = AgentState(**saved_state.values)
            initial_state['resumed_count'] = initial_state.get('resumed_count', 0) + 1
            
            if approval_decision:
                print(f"   📝 应用审批决定: {approval_decision}")
                initial_state['approval_status'] = approval_decision
                initial_state['approval_notes'] = approval_notes
                initial_state['approved_by'] = approved_by
                initial_state['needs_human_approval'] = False
        else:
            print(f"⚠️  未找到断点，从头开始执行")
            initial_state = create_initial_state(task_description, task_hash)
    else:
        initial_state = create_initial_state(task_description, task_hash)
    
    print(f"\n🎯 处理任务: {task_description}")
    print("-"*50)
    
    try:
        final_state = graph.invoke(initial_state, config=config)
        
        if persistence_manager.is_enabled():
            stats = persistence_manager.get_db_stats()
            print(f"\n💾 持久化统计:")
            print(f"   检查点数量: {stats.get('checkpoint_count', 0)}")
            print(f"   活跃会话数: {stats.get('thread_count', 0)}")
            print(f"   数据库位置: {stats.get('db_path', 'N/A')}")
        
        return final_state
        
    except Interrupt as e:
        print(f"\n{'!'*70}")
        print(f"🛑 工作流已中断 - 需要人工审批")
        print(f"{'!'*70}")
        print(f"\n📋 中断详情:")
        print(f"   任务: {task_description}")
        print(f"   估算成本: ${e.kwargs.get('estimated_cost', 0)}")
        print(f"   执行ID: {e.kwargs.get('task_id')}")
        print(f"   会话ID: {config['configurable']['thread_id']}")
        
        print(f"\n💡 下一步:")
        print(f"   使用以下命令恢复执行:")
        print(f"\n   run_agent_with_approval('{user_id}', '{task_description}',")
        print(f"                           resume=True, approval_decision='approved'/'rejected',")
        print(f"                           approval_notes='审批意见', approved_by='审批人')")
        print(f"\n{'!'*70}")
        
        raise
        
    except Exception as e:
        print(f"\n❌ 执行错误: {e}")
        raise

def create_initial_state(task_description: str, task_hash: str) -> AgentState:
    from datetime import datetime
    
    return AgentState(
        original_task=task_description,
        subtasks=[],
        reasoning="",
        current_step="开始",
        complexity="unknown",
        needs_deep_analysis=False,
        current_decomposition_level=0,
        max_decomposition_level=2,
        needs_further_decomposition=True,
        processing_queue=[],
        execution_id=f"{task_hash}_{int(time.time())}",
        created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        resumed_count=0,
        estimated_cost=0,
        needs_human_approval=False,
        approval_status=None,
        approval_notes=None,
        approved_by=None
    )

def check_and_resume_workflow():
    """检查并恢复工作流 - 先检查断点"""
    print("="*80)
    print("🎭 智能断点检测与恢复演示")
    print("📚 场景：先检查断点，再决定执行流程")
    print("="*80)
    
    user_id = "user_123"
    task_description = "制定全球市场扩张营销战略"
    task_hash = generate_task_hash(task_description)
    config = persistence_manager.create_thread_config(user_id, task_hash)
    
    print(f"\n🔍检查用户 '{user_id}' 的断点状态")
    print(f"   任务: '{task_description}'")
    print(f"   任务哈希: {task_hash}")
    
    graph = create_workflow()
    saved_state = graph.get_state(config)
    
    has_interrupt = False
    
    if saved_state and saved_state.next:
        next_node = saved_state.next[0][0] if saved_state.next[0] else None
        
        if next_node == "预算审批" or saved_state.values.get('approval_status') == 'pending':
            print(f"\n✅ 检测到审批中断点")
            print(f"   原始任务: {saved_state.values.get('original_task')}")
            print(f"   估算成本: ${saved_state.values.get('estimated_cost', 0)}")
            print(f"   已生成子任务: {len(saved_state.values.get('subtasks', []))}个")
            print(f"   可执行任务: {len([t for t in saved_state.values.get('subtasks', []) if t.get('actionable', False)])}个")
            
            has_interrupt = True
            
            print(f"\n🔷 场景：发现未完成的任务，需要恢复执行")
            print(f"   需要提供审批决定...")
            
            # 模拟审批人决定
            print(f"\n🧑‍💼 模拟审批人决策:")
            print(f"   1. 审查任务详情")
            print(f"   2. 分析成本效益")
            print(f"   3. 做出审批决定")
            print(f"\n📝 决策结果: 批准执行")
            print(f"   审批意见: '战略规划合理，预算在可控范围内'")
            print(f"   审批人: '张经理'")
            
            print(f"\n{'='*80}")
            print("🔄 第二步：携带审批决定恢复工作流")
            print("="*80)
            
            # 恢复执行
            final_state = run_agent_with_approval(
                user_id=user_id,
                task_description=task_description,
                resume=True,
                approval_decision="approved",
                approval_notes="战略规划合理，预算在可控范围内",
                approved_by="张经理"
            )
            
            if final_state:
                print(f"\n✅ 恢复执行成功")
                print(f"   最终审批状态: {final_state.get('approval_status')}")
                print(f"   生成报告: task_report_{final_state.get('execution_id')}.txt")
        else:
            print(f"\n⚠️  检测到其他类型断点")
            print(f"   下一个节点: {next_node}")
            has_interrupt = True
    else:
        print(f"\n🆕 无断点，全新开始")
    
    if not has_interrupt:
        print(f"\n🔷 场景：全新任务开始执行")
        print(f"   这是一个高成本任务，预计会触发审批中断...")
        
        print(f"\n{'='*80}")
        print("🚀 执行全新任务拆解")
        print("="*80)
        
        try:
            # 第一次执行：应该会中断
            run_agent_with_approval(
                user_id=user_id,
                task_description=task_description,
                resume=False
            )
        except Interrupt:
            print(f"\n✅ 预期中断：工作流已暂停，等待审批")
            print(f"   下次运行时会检测到这个断点并恢复")
    
    # 显示最终统计
    stats = persistence_manager.get_db_stats()
    
    print(f"\n{'='*80}")
    print("📊 系统能力验证")
    print("="*80)
    print(f"总检查点数量: {stats.get('checkpoint_count', 0)}")
    print(f"活跃会话数: {stats.get('thread_count', 0)}")
    print(f"断点检测: ✅ 自动检测未完成任务")
    print(f"人机协同: ✅ 支持审批中断与恢复")
    print(f"状态持久化: ✅ 断点保存与恢复")
    print("="*80)

if __name__ == "__main__":
    check_and_resume_workflow()
    
    if persistence_manager.checkpointer:
        persistence_manager.checkpointer.conn.close()