# WHU-2024-PO-PCIE
---

## 项目简介

本项目为武汉大学 2024 年舆情分析课程大作业实体部分，主要功能包括：

- **命名实体识别（NER）**
- **关系抽取（RE）**
- **事件抽取（EE）**

本项目基于深度学习方法，实现了对中文舆情数据的自动化信息抽取和分析。

---

## 模型来源与参考

原始模型部分参考了以下开源项目：
[PointerNet_Chinese_Information_Extraction](https://github.com/taishan1994/PointerNet_Chinese_Information_Extraction)

该项目为中文信息抽取任务提供了一个高性能的基线模型，具有较强的扩展性。

## 模型 Web 可视化与预测接口

------------------------------------------------------------------------

本项目实现了基于 Flask 的模型 Web 可视化与交互功能，包括以下模块：
- 事件抽取（Event Extraction，EE）
- 关系抽取（Relation Extraction，RE）
- 命名实体识别（Named Entity Recognition，NER）

## 项目结构

------------------------------------------------------------------------

``` plaintext
PCIE/UIE/                     # 项目主目录
├── __pycache__/              # Python 缓存文件
├── checkpoints/              # 模型检查点目录
├── data/                     # 数据目录
├── model_hub/                # 预训练模型目录
├── static/                   # Flask存储url目录
│   ├── logo.png/             # WHU logo
├── ee_index.html             # HTML 模板文件
│   ├── ee_index.html         # EE 模块的 Web 可视化页面
│   ├── ner_index.html        # NER 模块的 Web 可视化页面
│   ├── re_index.html         # RE 模块的 Web 可视化页面
├── utils/                    # 工具函数与配置文件
│   ├── __init__.py           # 工具模块初始化
│   ├── config.py             # 配置文件
│   ├── ee_data_loader.py     # EE 模块数据加载器
│   ├── ee_main.py            # EE 模块主脚本
│   ├── ee_predictor.py       # EE 模块预测脚本
│   ├── model.py              # 模型定义脚本
│   ├── ner_data_loader.py    # NER 模块数据加载器
│   ├── ner_main.py           # NER 模块主脚本
│   ├── ner_predictor.py      # NER 模块预测脚本
│   ├── re_data_loader.py     # RE 模块数据加载器
│   ├── re_main.py            # RE 模块主脚本
│   ├── re_predictor.py       # RE 模块预测脚本
└── README.md                 # 项目说明文件
```

## 功能说明

------------------------------------------------------------------------

### 1. Web 可视化

- **`ee_index.html`**: 提供事件抽取（EE）模型的输入界面与结果展示。
- **`re_index.html`**: 提供关系抽取（RE）模型的输入界面与结果展示。
- **`ner_index.html`**: 提供命名实体识别（NER）模型的输入界面与结果展示。

### 2. Flask 接口脚本

- **`ee_predict.py`**: 接收前端 EE 输入，调用事件抽取模型进行预测，并返回结果。
- **`re_predict.py`**: 接收前端 RE 输入，调用关系抽取模型进行预测，并返回结果。
- **`ner_predict.py`**: 接收前端 NER 输入，调用命名实体识别模型进行预测，并返回结果。

### 3. 前后端交互

- 用户通过 Web 界面上传文本数据。
- 前端通过 AJAX 请求将数据发送到 Flask 后端。
- Flask 调用相应的预测脚本（`ee_predict.py`、`re_predict.py`、`ner_predict.py`）处理数据。
- 后端返回预测结果，前端展示模型的预测输出。

## 启动服务

------------------------------------------------------------------------

### 1. 安装依赖

首先确保安装了 Python 和必要的依赖包：

    pip install flask

### 2. 启动 Flask 服务

运行以下脚本分别启动服务：

    python ee_predict.py  # 启动事件抽取服务
    python re_predict.py  # 启动关系抽取服务
    python ner_predict.py # 启动命名实体识别服务

服务将分别运行在以下端口：

- **事件抽取服务**: `http://127.0.0.1:1232`
- **关系抽取服务**: `http://127.0.0.1:1231`
- **命名实体识别服务**: `http://127.0.0.1:1230`

### 3. 访问 Web 页面

在浏览器中访问以下地址：

- **事件抽取界面**: <http://127.0.0.1:1232>
- **关系抽取界面**: <http://127.0.0.1:1231>
- **命名实体识别界面**: <http://127.0.0.1:1230>

## 修改说明

------------------------------------------------------------------------

在以下脚本中加入了 Flask 部分，用于处理前端请求：

- `ee_predict.py`
- `re_predict.py`
- `ner_predict.py`

这些脚本处理前端提交的数据，调用模型执行预测，并将预测结果返回给前端。具体实现逻辑可根据需求在脚本中扩展或调整。

在以下文件实现前端部分：

- `./templates`
- `./static`

## 注意事项

------------------------------------------------------------------------

- 确保模型文件和相关依赖已正确配置，避免运行时出错。
- 可以根据实际需求修改端口号或其他配置。
- 如果需要全局管理服务，可以自行编写 `app.py` 作为主入口。
#