# M4 AI Vision Pro 📸

这是一个基于浏览器的高性能实时 AI 视觉识别系统，专门为体验 Apple M4 芯片的强大算力而设计。它结合了目标检测（Object Detection）和自定义物体学习（Teachable Machine）功能。

## 🌟 功能特性

- **实时目标检测**：利用 COCO-SSD 模型自动识别并追踪画面中的主要物体。
- **自定义学习 (Teachable Machine)**：
    - 结合 MobileNet 特征提取和 KNN 分类器。
    - 只需点击几下，即可让 AI 认识任何特定物体。
- **高性能渲染**：
    - 针对 M4 芯片优化（通过 WebGL/WebGPU 加速）。
    - 采用 EMA（指数移动平均）算法平滑追踪框，防止画面抖动。
- **现代化 UI**：极简毛玻璃设计，实时反馈置信度和识别结果。

## 🚀 快速启动

由于浏览器对摄像头的安全限制，本项目**必须**在 `localhost` 或 `HTTPS` 环境下运行。

### 方法一：使用 VS Code Live Server (最简单)
1. 在 VS Code 中安装 **Live Server** 扩展。
2. 右键点击 `index.html`，选择 **Open with Live Server**。

### 方法二：使用 Python 命令
在项目根目录下运行：
```bash
python3 -m http.server 8000
```
然后在浏览器访问 `http://localhost:8000`。

### 方法三：使用 Node.js (npx)
```bash
npx serve .
```
然后在浏览器访问生成的地址。

## 🛠️ 如何使用

1. **初始化**：打开页面后，允许浏览器访问摄像头。
2. **目标追踪**：将物体放在摄像头前，绿框会自动锁定面积最大的物体。
3. **学习物体 A/B**：
   - 将物体 A 放在框内，持续点击“学习物体 A”。建议从不同角度录入 10-20 个样本。
   - 对物体 B 重复上述操作。
4. **实时识别**：录入样本后，AI 会实时显示当前框内物体的名称及其置信度。
5. **重置**：点击“重置模型”清空所有已学习的特征。

## 🏗️ 技术栈

- **Core**: JavaScript (ES6+), HTML5, CSS3
- **AI Engine**: [TensorFlow.js](https://www.tensorflow.org/js)
- **Models**: 
  - [MobileNet v2](https://github.com/tensorflow/tfjs-models/tree/master/mobilenet) (特征提取)
  - [COCO-SSD](https://github.com/tensorflow/tfjs-models/tree/master/coco-ssd) (目标检测)
  - [KNN Classifier](https://github.com/tensorflow/tfjs-models/tree/master/knn-classifier) (在线学习)

---
Developed for M4 AI Capability Testing.
