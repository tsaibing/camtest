// Constants
const WEBCAM_ID = 'webcam';
const CLASS_A = '物体 A';
const CLASS_B = '物体 B';

// Elements
const webcamElement = document.getElementById(WEBCAM_ID);
const overlay = document.getElementById('loading-overlay');
const resultEl = document.getElementById('prediction-result');
const confidenceFill = document.getElementById('confidence-fill');
const confidenceText = document.getElementById('confidence-text');
const countAEl = document.getElementById('count-a');
const countBEl = document.getElementById('count-b');
const btnA = document.getElementById('btn-class-a');
const btnB = document.getElementById('btn-class-b');
const btnReset = document.getElementById('btn-reset');

// Globals
let classifier;
let mobilenetModule;
let isPredicting = false;
let counts = { [CLASS_A]: 0, [CLASS_B]: 0 };

async function setupWebcam() {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: 'user' }
            });
            webcamElement.srcObject = stream;
            return new Promise((resolve) => {
                webcamElement.addEventListener('loadeddata', () => resolve(), false);
            });
        } catch (error) {
            throw new Error("摄像头权限被拒绝或不可用 (" + error.message + ")");
        }
    } else {
        throw new Error("浏览器不支持摄像头 API (可能需要在 localhost 或 HTTPS 环境下运行)");
    }
}

async function init() {
    try {
        console.log("Setting up tfjs backend...");
        await tf.ready();
        console.log(`Using backend: ${tf.getBackend()}`);

        console.log("Loading MobileNet...");
        mobilenetModule = await mobilenet.load({version: 2, alpha: 1.0});

        console.log("Creating KNN Classifier...");
        classifier = knnClassifier.create();

        console.log("Setting up Webcam...");
        await setupWebcam();

        // 隐藏加载动画
        overlay.classList.add('hidden');
        
        // 开始视频循环
        requestAnimationFrame(predictLoop);

    } catch (err) {
        console.error("Initialization error:", err);
        document.querySelector('#loading-overlay p').innerText = `加载失败: ${err.message}`;
    }
}

// 收集样本
const addExample = async (classId) => {
    // 获取摄像头的张量数据
    const img = tf.browser.fromPixels(webcamElement);
    // 提取图像特征
    const features = mobilenetModule.infer(img, true);
    
    // 添加到 KNN 分类器
    classifier.addExample(features, classId);
    
    // 更新计数器
    counts[classId]++;
    if(classId === CLASS_A) countAEl.innerText = counts[CLASS_A];
    if(classId === CLASS_B) countBEl.innerText = counts[CLASS_B];

    // 清理内存
    img.dispose();
};

// 持续预测
const predictLoop = async () => {
    if (classifier.getNumClasses() > 0) {
        const img = tf.browser.fromPixels(webcamElement);
        const features = mobilenetModule.infer(img, true);

        // 预测
        const res = await classifier.predictClass(features, 3);
        
        // 更新 UI
        resultEl.innerText = res.label;
        const confidence = (res.confidences[res.label] * 100).toFixed(1);
        confidenceFill.style.width = `${confidence}%`;
        confidenceText.innerText = `置信度: ${confidence}%`;

        // 清理内存
        img.dispose();
    } else {
        resultEl.innerText = "等待学习...";
        confidenceFill.style.width = "0%";
        confidenceText.innerText = `置信度: 0%`;
    }

    // 继续下一帧预测
    requestAnimationFrame(predictLoop);
};

// 绑定事件
// mousedown event to capture sample, or maybe we want a continuous capture on mousedown?
// Actually, click is safer to capture single frames cleanly
btnA.addEventListener('click', () => addExample(CLASS_A));
btnB.addEventListener('click', () => addExample(CLASS_B));

btnReset.addEventListener('click', () => {
    classifier.clearAllClasses();
    counts = { [CLASS_A]: 0, [CLASS_B]: 0 };
    countAEl.innerText = 0;
    countBEl.innerText = 0;
});

// 启动应用
window.onload = init;
