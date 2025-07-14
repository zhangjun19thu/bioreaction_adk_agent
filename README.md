# 生物反应科研Agent

一个基于Google ADK框架开发的智能生物化学反应研究助手，集成了多Agent协作架构、智能查询分析、深度研究工具和高级分析功能。

## 🚀 系统特性

### 核心功能
- **智能查询分析**: 多字段模糊搜索、数据库统计、相似反应查找
- **深度研究集成**: 文献分析、多文献对比、上下文关联
- **高级分析功能**: 趋势分析、对比分析、优化建议
- **多Agent协作**: 数据分析、文献研究、优化建议三个专门Agent

### 技术架构
- **框架**: Google ADK (Agent Development Kit)
- **模型**: Gemini 2.5 Flash
- **数据**: 结构化CSV数据库 + 文献元数据
- **配置**: 集中化配置管理

## 📁 项目结构

```
bioreaction_adk_agent/
├── config.py                 # 集中配置管理
├── agent.py                  # 主Agent和子Agent定义
├── main.py                   # 启动程序
├── test_system.py           # 系统测试脚本
├── requirements.txt         # 依赖包列表
├── data/                    # 数据库文件
│   └── papers1000_database/ # 结构化数据表
└── tools/                   # 工具模块
    ├── database_loader.py   # 数据库加载器
    ├── database_query_tools.py  # 数据库查询工具
    ├── deep_research_tools.py   # 深度研究工具
    └── advanced_tools.py    # 高级分析工具
```

## ⚙️ 配置系统

### 配置文件 (config.py)
系统使用集中化配置管理，主要配置项包括：

```python
# Agent配置
AGENT_CONFIG = {
    "model": "gemini-2.5-flash",
    "app_name": "bioreaction_research_app",
    "session_service": "InMemorySessionService"
}

# 查询配置
QUERY_CONFIG = {
    "max_results": 10,
    "min_data_points": 5,
    "default_top_n": 5
}

# 分析配置
ANALYSIS_CONFIG = {
    "temperature_ranges": {...},
    "ph_ranges": {...},
    "correlation_threshold": 0.3
}
```

### 环境变量
```bash
export GEMINI_API_KEY="your_api_key_here"
```

## 🛠️ 安装和设置

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 设置环境变量
```bash
export GEMINI_API_KEY="your_gemini_api_key"
```

### 3. 验证配置
```bash
python -m bioreaction_adk_agent.test_system
```

## 🚀 使用方法

### 方法1: 使用ADK Web界面 (推荐)
```bash
# 启动开发UI
adk web
```

### 方法2: 使用Python API
```python
import asyncio
from bioreaction_adk_agent.agent import query_agent

async def main():
    response = await query_agent("查找ADK酶的反应")
    print(response)

asyncio.run(main())
```

### 方法3: 直接运行
```bash
python main.py
```

## 🔧 工具功能

### 数据库查询工具
- `get_reaction_summary`: 获取反应完整摘要
- `find_reactions_by_enzyme`: 按酶查找反应
- `find_inhibition_data`: 查找抑制剂数据
- `smart_search_reactions`: 智能多字段搜索
- `get_database_statistics`: 数据库统计信息

### 深度研究工具
- `get_summary_from_literature`: 文献摘要分析
- `analyze_multiple_literature`: 多文献对比
- `get_literature_context`: 文献上下文信息
- `find_related_literature`: 相关文献推荐

### 高级分析工具
- `analyze_reaction_trends`: 反应趋势分析
- `compare_reactions`: 反应对比分析
- `suggest_optimization`: 优化建议
- `literature_comparison`: 文献对比分析

## 📊 示例查询

### 基础查询
```
"查找ADK酶的反应"
"获取数据库统计信息"
"查找E. coli中的酶反应"
```

### 高级分析
```
"分析ADK酶在E. coli中的性能趋势"
"比较不同物种的酶活性"
"查找相似反应并分析模式"
```

### 深度研究
```
"分析文献12345的实验方法"
"对比多篇文献的结论差异"
"获取反应A:B的上下文信息"
```

## 🧪 测试系统

运行完整系统测试：
```bash
python -m bioreaction_adk_agent.test_system
```

测试包括：
- ✅ 配置验证
- ✅ 数据库加载
- ✅ 基本查询功能
- ✅ Agent查询功能
- ✅ 高级分析工具
- ✅ 深度研究工具
- ✅ 错误处理

## 🔍 故障排除

### 常见问题

1. **相对导入错误**
   ```bash
   # 在包的上一级目录运行
   python -m bioreaction_adk_agent.test_system
   ```

2. **配置错误**
   - 检查 `config.py` 中的路径配置
   - 确认数据库文件存在
   - 验证环境变量设置

3. **API密钥问题**
   ```bash
   export GEMINI_API_KEY="your_api_key"
   ```

4. **数据库加载失败**
   - 检查数据文件路径
   - 确认CSV文件格式正确
   - 验证文件权限

## 📈 性能优化

### 配置调优
- 调整 `QUERY_CONFIG.max_results` 控制查询结果数量
- 修改 `ANALYSIS_CONFIG.correlation_threshold` 调整相关性阈值
- 设置 `CACHE_CONFIG.enable` 启用缓存功能

### 内存优化
- 数据库自动加载到内存
- 支持大数据集的分页查询
- 智能缓存机制

## 🤝 贡献指南

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 📄 许可证

本项目采用 MIT 许可证。

## 🆘 支持

如有问题或建议，请：
1. 查看故障排除部分
2. 运行系统测试
3. 检查配置设置
4. 提交 Issue

---

**生物反应科研Agent** - 让生物化学研究更智能、更高效！🧬🔬 