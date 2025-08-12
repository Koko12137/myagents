from sqlalchemy import create_engine, Column, String, DateTime, Text, MetaData, Table
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime


# 创建基类
Base = declarative_base()


class MemoryTable:
    """动态 Memory 表模型 - 支持不同环境"""
    
    def __init__(self, env_id: str):
        """初始化特定环境的 Memory 表
        
        Args:
            env_id (str): 环境ID，用作表名
        """
        self.env_id = env_id
        self.table_name = f"env_{env_id}"
        
        # 动态创建表结构
        self.table = Table(
            self.table_name,
            MetaData(),
            Column('memory_id', String(255), primary_key=True),
            Column('task_id', String(255), nullable=False, index=True),
            Column('create_time', DateTime, default=datetime.now(datetime.timezone.utc), nullable=False),
            Column('role', String(100), nullable=False),
            Column('content', Text, nullable=False),
            Column('agent_name', String(255), nullable=False),
            Column('agent_type', String(100), nullable=False)
        )


class MemoryDatabase:
    """Memory 数据库管理类 - 维护数据库连接，管理所有环境表"""
    
    def __init__(self, db_path="memory.db"):
        """初始化数据库连接"""
        self.db_path = db_path
        self.engine = create_engine(f'sqlite:///{db_path}', echo=False)
        self.Session = sessionmaker(bind=self.engine)
        self.metadata = MetaData()
        self._env_tables = {}  # 缓存已创建的环境表
        
    def create_env_table(self, env_id: str) -> bool:
        """为特定环境创建记忆表
        
        Args:
            env_id (str): 环境ID
            
        Returns:
            bool: 创建是否成功
        """
        try:
            table_name = f"env_{env_id}"
            
            # 如果表已存在，直接返回
            if self.engine.dialect.has_table(self.engine, table_name):
                return True
            
            # 创建表结构
            table = Table(
                table_name,
                self.metadata,
                Column('memory_id', String(255), primary_key=True),
                Column('task_id', String(255), nullable=False, index=True),
                Column('create_time', DateTime, default=datetime.now(datetime.timezone.utc), nullable=False),
                Column('role', String(100), nullable=False),
                Column('content', Text, nullable=False),
                Column('agent_name', String(255), nullable=False),
                Column('agent_type', String(100), nullable=False)
            )
            
            # 创建表
            table.create(self.engine)
            self._env_tables[env_id] = table
            return True
            
        except Exception as e:
            print(f"创建环境表失败: {e}")
            return False
    
    def drop_env_table(self, env_id: str) -> bool:
        """删除特定环境的记忆表
        
        Args:
            env_id (str): 环境ID
            
        Returns:
            bool: 删除是否成功
        """
        try:
            table_name = f"env_{env_id}"
            if self.engine.dialect.has_table(self.engine, table_name):
                table = Table(table_name, self.metadata)
                table.drop(self.engine)
                if env_id in self._env_tables:
                    del self._env_tables[env_id]
                return True
            return False
        except Exception as e:
            print(f"删除环境表失败: {e}")
            return False
    
    def env_table_exists(self, env_id: str) -> bool:
        """检查特定环境的表是否存在
        
        Args:
            env_id (str): 环境ID
            
        Returns:
            bool: 表是否存在
        """
        table_name = f"env_{env_id}"
        return self.engine.dialect.has_table(self.engine, table_name)
    
    def list_env_tables(self) -> list[str]:
        """列出所有环境表
        
        Returns:
            list: 环境ID列表
        """
        env_ids = []
        for table_name in self.engine.table_names():
            if table_name.startswith('env_'):
                env_id = table_name[4:]  # 去掉 'env_' 前缀
                env_ids.append(env_id)
        return env_ids
    
    def get_session(self):
        """获取数据库会话"""
        return self.Session()
    
    def close(self):
        """关闭数据库连接"""
        self.engine.dispose()


class MemoryManager:
    """Memory 管理器 - 负责特定环境的增删改查操作"""
    
    def __init__(self, db: MemoryDatabase, env_id: str):
        """初始化 Memory 管理器
        
        Args:
            db (MemoryDatabase): 数据库管理实例
            env_id (str): 环境ID
        """
        self.db = db
        self.env_id = env_id
        self.table_name = f"env_{env_id}"
        
        # 确保环境表存在
        if not self.db.env_table_exists(env_id):
            self.db.create_env_table(env_id)
    
    def add_memory(
        self, 
        memory_id: str, 
        task_id: str, 
        role: str, 
        content: str, 
        agent_name: str, 
        agent_type: str, 
        create_time: datetime = None) -> bool:
        """Add a new memory record.
        
        Args:
            memory_id (str):
                The id of the memory.
            task_id (str):
                The id of the task.
            role (str):
                The role of the memory.
            content (str):
                The content of the memory.
            agent_name (str):
                The name of the agent.
            agent_type (str):
                The type of the agent.
            create_time (datetime):
                The time when the memory is created.
        """
        session = self.db.get_session()
        try:
            # 使用原生SQL插入，因为表是动态创建的
            from sqlalchemy import text
            sql = text("""
                INSERT INTO {} (memory_id, task_id, create_time, role, content, agent_name, agent_type)
                VALUES (:memory_id, :task_id, :create_time, :role, :content, :agent_name, :agent_type)
            """.format(self.table_name))
            
            session.execute(sql, {
                'memory_id': memory_id,
                'task_id': task_id,
                'create_time': create_time or datetime.now(datetime.timezone.utc),
                'role': role,
                'content': content,
                'agent_name': agent_name,
                'agent_type': agent_type
            })
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def get_memory_by_id(self, memory_id: str) -> tuple:
        """根据 memory_id 获取记录"""
        session = self.db.get_session()
        try:
            from sqlalchemy import text
            sql = text("SELECT * FROM {} WHERE memory_id = :memory_id".format(self.table_name))
            result = session.execute(sql, {'memory_id': memory_id})
            return result.fetchone()
        finally:
            session.close()
    
    def get_memories_by_task(self, task_id: str) -> list[tuple]:
        """根据 task_id 获取所有相关记录"""
        session = self.db.get_session()
        try:
            from sqlalchemy import text
            sql = text("SELECT * FROM {} WHERE task_id = :task_id ORDER BY create_time".format(self.table_name))
            result = session.execute(sql, {'task_id': task_id})
            return result.fetchall()
        finally:
            session.close()
    
    def get_memories_by_role(self, role: str) -> list[tuple]:
        """根据 role 获取所有相关记录"""
        session = self.db.get_session()
        try:
            from sqlalchemy import text
            sql = text("SELECT * FROM {} WHERE role = :role ORDER BY create_time".format(self.table_name))
            result = session.execute(sql, {'role': role})
            return result.fetchall()
        finally:
            session.close()
    
    def get_memories_by_agent(self, agent_name: str) -> list[tuple]:
        """根据 agent_name 获取所有相关记录"""
        session = self.db.get_session()
        try:
            from sqlalchemy import text
            sql = text("SELECT * FROM {} WHERE agent_name = :agent_name ORDER BY create_time".format(self.table_name))
            result = session.execute(sql, {'agent_name': agent_name})
            return result.fetchall()
        finally:
            session.close()
    
    def get_memories_by_agent_type(self, agent_type: str) -> list[tuple]:
        """根据 agent_type 获取所有相关记录"""
        session = self.db.get_session()
        try:
            from sqlalchemy import text
            sql = text("SELECT * FROM {} WHERE agent_type = :agent_type ORDER BY create_time".format(self.table_name))
            result = session.execute(sql, {'agent_type': agent_type})
            return result.fetchall()
        finally:
            session.close()
    
    def update_memory(self, memory_id: str, **kwargs):
        """更新 memory 记录"""
        session = self.db.get_session()
        try:
            from sqlalchemy import text
            
            # 构建更新SQL
            set_clause = ", ".join([f"{k} = :{k}" for k in kwargs.keys()])
            sql = text(f"UPDATE {self.table_name} SET {set_clause} WHERE memory_id = :memory_id")
            
            params = kwargs.copy()
            params['memory_id'] = memory_id
            
            result = session.execute(sql, params)
            session.commit()
            return result.rowcount > 0
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def delete_memory(self, memory_id: str) -> bool:
        """删除 memory 记录"""
        session = self.db.get_session()
        try:
            from sqlalchemy import text
            sql = text("DELETE FROM {} WHERE memory_id = :memory_id".format(self.table_name))
            result = session.execute(sql, {'memory_id': memory_id})
            session.commit()
            return result.rowcount > 0
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def get_all_memories(self, limit: int = None) -> list[tuple]:
        """获取所有 memory 记录"""
        session = self.db.get_session()
        try:
            from sqlalchemy import text
            sql_text = f"SELECT * FROM {self.table_name} ORDER BY create_time DESC"
            if limit:
                sql_text += f" LIMIT {limit}"
            sql = text(sql_text)
            result = session.execute(sql)
            return result.fetchall()
        finally:
            session.close()


# 使用示例
if __name__ == "__main__":
    # 创建数据库实例
    db = MemoryDatabase("multi_env_memory.db")
    
    # 列出所有环境表
    print("现有环境表:", db.list_env_tables())
    
    # 为不同环境创建 Memory 管理器
    env1_manager = MemoryManager(db, "env_001")
    env2_manager = MemoryManager(db, "env_002")
    
    # 在不同环境中添加数据
    env1_manager.add_memory(
        memory_id="mem_001",
        task_id="task_001",
        role="user",
        content="环境1的记忆内容",
        agent_name="agent_1",
        agent_type="assistant"
    )
    
    env2_manager.add_memory(
        memory_id="mem_002",
        task_id="task_002",
        role="assistant",
        content="环境2的记忆内容",
        agent_name="agent_2",
        agent_type="assistant"
    )
    
    # 查询不同环境的数据
    env1_memories = env1_manager.get_memories_by_task("task_001")
    env2_memories = env2_manager.get_memories_by_task("task_002")
    
    print(f"环境1找到 {len(env1_memories)} 条记录")
    print(f"环境2找到 {len(env2_memories)} 条记录")
    
    # 列出所有环境表
    print("所有环境表:", db.list_env_tables())
    
    # 关闭连接
    db.close()
