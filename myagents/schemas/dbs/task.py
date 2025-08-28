from sqlalchemy import Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class TaskStep(Base):
	__tablename__ = "task_steps"

	id = Column(Integer, primary_key=True, autoincrement=True)
	task_id = Column(String(255), nullable=False, index=True)
	pre_task_id = Column(Integer, nullable=True)  # 前置任务ID，可为空
	prompt = Column(String, nullable=True)
	response = Column(String, nullable=True)
