# 手语数字识别系统

基于深度学习的手语数字实时识别系统，支持模型训练、单张图片预测和摄像头实时识别。

## 📋 项目概述

本项目使用卷积神经网络（CNN）实现手语数字（0-9）的识别，包含以下核心功能：

- **模型训练**：从 `.npy` 格式数据集训练 CNN 模型
- **单张图片预测**：使用训练好的模型识别本地图片中的手语数字
- **摄像头实时识别**：通过摄像头实时识别手势数字，支持多种图像处理优化

## ✨ 核心特性

### 模型训练
- 三层卷积神经网络结构
- 支持早停、学习率衰减等训练策略
- 自动保存最优模型

### 图像预处理（摄像头模块）
- 多颜色空间肤色检测融合（YCrCb + HSV + RGB）
- CLAHE 对比度限制自适应直方图均衡化
- 自适应 Gamma 校正（光照补偿）
- Canny 边缘检测辅助识别
- 背景减法动态背景更新
- 置信度加权投票机制

### 摄像头实时识别
- 实时手势检测和识别
- 多帧稳定确认机制
- 调试模式显示 Top-3 预测结果
- 快捷键支持（截图、重置背景等）

## 🛠️ 环境要求

### 基础环境
- Python 3.8+
- Windows / Linux / macOS

### 核心依赖库
| 库名称 | 版本要求 | 用途 |
|--------|----------|------|
| numpy | >= 1.21.0 | 数据处理 |
| tensorflow | >= 2.8.0 | 深度学习框架 |
| scikit-learn | >= 1.0.0 | 数据划分 |
| Pillow | >= 8.0.0 | 图片处理 |
| opencv-python | >= 4.5.0 | 摄像头操作和图像处理 |

## 📦 安装步骤

### 1. 克隆项目
```bash
git clone <repository-url>
cd SighLanguageDigitRecognition
```

### 2. 创建虚拟环境（推荐）

**Windows:**
```powershell
python -m venv .venv
.venv\Scripts\activate
```

**Linux/macOS:**
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. 安装依赖
```bash
pip install numpy tensorflow scikit-learn pillow opencv-python
```

### 4. 准备数据集

确保项目目录下存在以下文件（仓库已包含，无需另行准备）：
- `X.npy` - 图像数据，形状为 `(样本数, 高度, 宽度)` 或 `(样本数, 高度, 宽度, 通道)`
- `Y.npy` - 标签数据，可以是数值标签或 one-hot 编码

## 🚀 使用方法

### 1. 训练模型
```bash
python main.py
```

训练完成后会生成：
- `best_model.keras` - 最优模型文件

### 2. 预测单张图片
```bash
python main.py your_image.jpg
```

支持的图片格式：`.png`, `.jpg`, `.jpeg`, `.bmp`, `.gif`

### 3. 摄像头实时识别
```bash
python camera_recognition.py
```

**快捷键：**
| 按键 | 功能 |
|------|------|
| `q` | 退出程序 |
| `s` | 保存截图 |
| `d` | 切换调试模式（显示 Top-3 预测结果） |
| `r` | 重置背景模型 |

## 📁 文件结构

```
SighLanguageDigitRecognition/
├── main.py                 # 主程序（模型训练、图片预测）
├── camera_recognition.py   # 摄像头实时识别模块
├── README.md              # 项目说明文档
├── X.npy                  # 训练图像数据（已包含）
├── Y.npy                  # 训练标签数据（已包含）
├── best_model.keras       # 训练好的模型（运行 main.py 后生成）
└── .venv/                 # Python 虚拟环境（可选）
```

## 🔧 技术栈

### 深度学习框架
- **TensorFlow 2.x** - 深度学习训练和推理
- **Keras** - 高层神经网络 API

### 图像处理
- **OpenCV** - 实时图像处理和摄像头操作
- **PIL/Pillow** - 图片加载和预处理

### 数据处理
- **NumPy** - 数值计算和数组操作
- **scikit-learn** - 数据划分和评估

## 🧠 模型结构

```
输入层 (64x64x1)
    ↓
Conv2D(32) → BatchNorm → Conv2D(32) → BatchNorm → MaxPool → Dropout(0.25)
    ↓
Conv2D(64) → BatchNorm → Conv2D(64) → BatchNorm → MaxPool → Dropout(0.25)
    ↓
Conv2D(128) → BatchNorm → MaxPool → Dropout(0.25)
    ↓
Flatten → Dense(256) → BatchNorm → Dropout(0.5) → Dense(128) → Dropout(0.3)
    ↓
输出层 (10类, softmax)
```

## ⚠️ 注意事项

1. **数据集格式**：确保 `X.npy` 和 `Y.npy` 文件格式正确
2. **摄像头权限**：运行摄像头识别时需确保摄像头已连接并授权
3. **光照条件**：建议在光线充足的环境下使用摄像头识别
4. **手势位置**：将手放入绿色检测框内进行识别
5. **背景初始化**：首次启动摄像头会自动初始化背景模型（约10帧）

## 📝 示例输出

### 训练输出
```
加载数据...
预处理数据...
数据形状: (1649, 64, 64, 1), 类别数: 10
构建模型...
Model: "sequential"
...
开始训练...
Epoch 1/50
52/52 [==============================] - 3s 58ms/step - loss: 1.8523 - accuracy: 0.3845 - val_loss: 1.1234 - val_accuracy: 0.6567
...
评估模型...
测试集准确率: 97.85%
模型已保存为 'best_model.keras'
```

### 图片预测输出
```
预测数字: 5, 置信度: 98.56%
```

### 摄像头识别输出
```
正在加载模型: best_model.keras
模型输入形状: (64, 64, 1)
模型输出类别数: 10
======================================================================
Camera started! (High Precision Version)
Put your hand in the green box
Press 'q' to quit | 's' to save screenshot
Press 'd' to toggle debug mode | 'r' to reset background
======================================================================
背景模型初始化完成
```


