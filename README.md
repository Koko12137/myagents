# My Agents

## 项目简介

My Agents 是一个强大且灵活的 AI 智能体开发框架，基于大语言模型（LLM），支持 ReAct 推理与行动范式，具备任务规划、执行与编排能力，并可无缝集成多种工具和 API。适用于复杂任务自动化、智能助手、AI 工具链等场景。

### 基本假设

1. 人都得处于环境中
2. 人可以通过工具与环境进行交互
3. 处于特定任务中的人应通过特定的工作流与环境进行交互
4. 简单环境是有状态的，简单工具是无状态的，二者可以组合成有状态的工具环境

## 特点

- 🤖 **LLM 驱动的智能体**: 利用大语言模型的力量创建智能代理
- 🔄 **ReAct 框架**: 实现 ReAct（推理与行动）框架，用于解决复杂任务
- 🛠️ **工具集成**: 通过 MCP（模型控制协议）无缝集成各种工具和 API
- 📝 **任务管理**: 复杂的任务规划、执行和编排系统
- 🔄 **异步支持**: 完整的异步支持，提供更好的性能和可扩展性

## 安装与使用方法

### 依赖安装

建议使用 Python 3.12+，并使用 uv 进行依赖管理：

```bash
pip install uv 

uv sync
```

### 配置

1. **API Key 配置**：
   - 在 `configs/api_keys.json` 中填写你的 OpenAI/Tavily 等 API Key。
   - 示例：
     ```json
     {
         "OPENAI_KEY": "sk-xxx",
         "TAVILY_KEY": "tvly-xxx"
     }
     ```
2. **Agent 配置**：
   - 编辑 `configs/agents.json`，配置 LLM 参数（模型、温度、token 等）。
   - 示例：
     ```json
     {
         "plan_llm": {"provider": "openai", "model": "gpt-4o-mini", "temperature": 1.0},
         "action_llm": {"provider": "openai", "model": "gpt-4o-mini", "temperature": 0.1}
     }
     ```

### 启动与使用

#### 1. 启动服务（Gate/MCP）

如需以服务方式运行，可配置 `configs/gate_config.json` 和 `configs/mcp_config.json`，然后：

```bash
uv run run_mcp_gate.py
```

#### 2. 命令行交互

直接运行 CLI，输入问题即可体验智能体推理与行动：

```bash
uv run cli.py --question "如何用Python实现快速排序？"
```

## 项目结构

```plain text
myagents/
│   ├── prompts/          # Prompt 模板与环境
│   │   ├── envs/         # 各类环境下的 prompt 模板
│   │   └── workflows/    # 工作流相关 prompt
│   ├── src/              # 核心源码
│   │   ├── agents/       # 智能体基类与实现
│   │   ├── envs/         # 任务、环境、接口定义
│   │   ├── llms/         # LLM 抽象与实现
│   │   ├── mcps/         # MCP 协议与工具集成
│   │   ├── workflows/    # 任务规划、行动、编排流
│   │   ├── factory.py    # Agent/LLM 工厂与配置加载
│   │   ├── logger.py     # 日志工具
│   │   ├── message.py    # 消息协议与格式
│   │   └── utils.py      # 通用工具函数
│   ├── tests/            # 测试用例
│   └── DEV.md            # 开发者文档
├── configs/              # 配置文件（API Key、Agent、MCP等）
├── cli.py                # 命令行入口
├── run_mcp_gate.py       # Gate/MCP 服务启动脚本
├── README.md             # 项目说明
├── pyproject.toml        # Python 项目依赖配置
└── ...
```

---

如需自定义 prompt、扩展工具或集成新 LLM，可参考 `prompts/` 和 `src/llms/` 目录下的实现。

欢迎 Issue 与 PR！
