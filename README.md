# VisAnnotator Lab - 计算社会科学标注实验台

**VisAnnotator Lab** 是一个专为社会科学研究者设计的“所见即所得” (WYSIWYG) 文本标注实验台。它旨在弥合定性调试与定量生产之间的鸿沟，让您无需编写复杂的 Python 代码，即可利用 DeepSeek 等大语言模型 (LLM) 进行大规模、高信度的文本分析。

![Status](https://img.shields.io/badge/Status-Beta-blue) ![Python](https://img.shields.io/badge/Python-3.9%2B-green) ![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)

## 🎯 核心价值

*   **可视化调试 (Interactive Debugging)**：摆脱黑盒，实时查看 Prompt 对特定样本的输出效果。
*   **结构化输出 (Structured Output)**：通过 GUI 定义 JSON Schema，确保 LLM 输出严格符合格式要求，无需后期清洗。
*   **信效度检验 (Reliability & Validity)**：内置人工标注界面与 Cohen's Kappa 计算模块，轻松验证模型与人类的一致性，满足学术发表标准。
*   **批量生产 (Batch Production)**：从调试模式无缝切换至生产模式，支持高并发批量标注，并提供独立的 Python 脚本导出功能。

## 直接运行网页版：

https://visannotatorlab.streamlit.app/

---

## 🚀 本地部署

### 1. 环境准备

确保您的电脑已安装 [Python 3.9](https://www.python.org/downloads/) 或更高版本。

### 2. 安装依赖

下载本项目代码后，在终端（Terminal 或 CMD）中运行：

```bash
pip install -r requirements.txt
```

> 依赖列表包括：`streamlit`, `pandas`, `openai`, `pydantic`, `plotly`, `scikit-learn`, `openpyxl`。

### 3. 启动应用

在终端运行以下命令启动可视化界面：

```bash
streamlit run app.py
```

启动后，浏览器会自动打开 `http://localhost:8501`。

---

## 📖 功能详解与使用指南

### 1. 数据接入 (Data Ingestion)
*   **支持格式**：CSV (`.csv`) 或 Excel (`.xlsx`)。
*   **功能**：上传文件后，系统会自动解析列名并展示前 100 行预览。
*   **演示数据**：如果您手头没有数据，可点击“加载演示数据”按钮快速体验。

### 2. 提示词与 Schema 定义 (Define)
这是核心配置模块，您可以在此定义“让模型提取什么”以及“如何提取”。

*   **Schema 编辑器**：
    *   **字段定义**：添加字段名、类型（String, Integer, Boolean, Enum, List）及描述。
    *   **Enum (枚举)**：对于情感分析或分类任务，请选择 `Enum` 类型并在选项中输入类别（如 `Positive, Negative`），这对于计算 Kappa 系数至关重要。
*   **提示词工程 (Prompt Engineering)**：
    *   **多配置管理**：点击右上角的 `+` 号可以创建多个 Prompt 组合，方便进行 A/B 测试。
    *   **模板变量**：使用 `{{列名}}` (例如 `{{content}}`) 将数据中的文本动态插入 Prompt 中。

### 3. 标注执行台 (Annotation Runner)
提供两种运行模式，兼顾调试与生产。

*   **调试模式 (Debug Mode)**：
    *   支持“前 N 行”、“随机采样”或“关键词过滤”三种采样方式。
    *   快速运行少量数据，验证 Prompt 和 Schema 是否报错。
*   **生产模式 (Production Mode)**：
    *   对全量数据进行标注。
    *   **实时进度**：显示批次处理进度条。
    *   **结果下载**：任务完成后，直接下载包含原始数据与标注结果的 CSV 文件。
    *   **离线脚本**：页面底部提供 `batch_label.py` 脚本导出功能，方便在服务器后台运行超大规模任务。

### 4. 评估与信度 (Evaluation & Reliability) ✨ *学术特色功能*
本模块专为验证数据质量设计。

*   **构建验证集 (Validation Set)**：
    *   随机抽取 N 条数据作为验证集。
    *   **人工标注界面**：在界面上直接阅读文本，并使用下拉菜单（基于您定义的 Schema）进行人工打标。
*   **对比实验 (Experiments)**：
    *   选择一个或多个 Prompt 配置。
    *   运行实验：系统会使用选定的 Prompt 对验证集进行标注。
    *   **自动化报告**：
        *   **准确率 (Accuracy)**：模型预测正确的比例。
        *   **Cohen's Kappa**：衡量模型与人工标注的一致性（排除随机偶然性），是社会科学领域的金标准。
        *   **配置间一致性**：展示不同 Prompt 组合之间的 Kappa 系数矩阵，评估模型的稳健性。

---

## ⚙️ 系统配置 (Sidebar)

在左侧边栏，您可以配置 LLM 连接信息：

*   **DeepSeek API Key**：必填。您的 API 密钥（不会上传至任何第三方，仅用于请求 DeepSeek 接口）。
*   **Base URL**：默认为 `https://api.deepseek.com`。
*   **并发数 (Concurrency)**：控制同时发起的请求数量，建议设置为 5-10。
*   **单次请求条数 (Batch Size)**：控制一次 API 调用包含多少条数据。设为 >1 可显著节省 Token 和时间。
*   **配置导入/导出**：您可以将当前的 Schema 和 Prompt 保存为 JSON 文件，或加载之前的配置。

---

## ❓ 常见问题 (FAQ)

**Q: 我的数据会被上传吗？**
A: **不会**。VisAnnotator 本地运行。数据仅在您的浏览器和本地 Python 进程中处理。只有在点击“运行”时，相关的文本片段会被发送给 DeepSeek API 进行推理，除此之外数据绝不离线。

**Q: 遇到 `JSON Decode Error` 怎么办？**
A: 这通常意味着模型没有严格遵循 JSON 格式输出。
1. 检查 Schema 定义是否清晰。
2. 尝试降低 `Temperature` (例如设为 0.1)。
3. 在 System Prompt 中强调“必须输出 JSON”。
(注意：本系统已内置强制 JSON 约束，通常只需重试即可)。

**Q: 如何计算 Cohen's Kappa？**
A: 请确保您在 Schema 中将分类字段定义为 **Enum** 类型，并在“评估与信度”页面完成了人工标注。对于自由文本 (String) 字段，仅支持计算完全匹配率。

---

## 🛠️ 技术栈

*   **Frontend**: [Streamlit](https://streamlit.io/)
*   **Data Processing**: Pandas
*   **LLM Integration**: OpenAI SDK (Compatible with DeepSeek), Asyncio
*   **Validation**: Pydantic
*   **Analysis**: Scikit-learn (Metrics)

---

**Happy Researching!** 🧪
