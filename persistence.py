# persistence.py - 独立模块，展示模块化设计
import os
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from dotenv import load_dotenv

# ----- 持久化管理器 -----
class PersistenceManager:
    """状态持久化管理器"""
    
    def __init__(self, db_path="task_decomposition.db"):
        self.db_path = db_path
        self.checkpointer = None
        self._init_checkpointer()
    
    def _init_checkpointer(self):
        """初始化SQLite检查点"""
        try:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.checkpointer = SqliteSaver(conn)
            self.checkpointer.setup()
            print(f"💾 持久化已启用: {self.db_path}")
        except Exception as e:
            print(f"❌ 持久化初始化失败: {e}")
            self.checkpointer = None
    
    def is_enabled(self):
        return self.checkpointer is not None
    
    def create_thread_config(self, user_id: str, task_hash: str):
        """创建线程配置"""
        thread_id = f"user_{user_id}_task_{task_hash}"
        return {"configurable": {"thread_id": thread_id}}
    
    def get_db_stats(self):
        """获取数据库统计"""
        if not self.checkpointer:
            return {"enabled": False}
        
        try:
            conn = self.checkpointer.conn
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM checkpoints")
            checkpoint_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT thread_id) FROM checkpoints")
            thread_count = cursor.fetchone()[0]
            
            return {
                "enabled": True,
                "checkpoint_count": checkpoint_count,
                "thread_count": thread_count,
                "db_path": self.db_path
            }
        except:
            return {"enabled": False}

# 单例实例
persistence_manager = PersistenceManager()