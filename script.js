// Constants
const WEBCAM_ID = 'webcam';
const CLASS_A = '物体 A';
const CLASS_B = '物体 B';

// Elements
const webcamElement = document.getElementById(WEBCAM_ID);
const canvasElement = document.getElementById('overlay-canvas');
const ctx = canvasElement.getContext('2d');
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
let objectDetector;
let counts = { [CLASS_A]: 0, [CLASS_B]: 0 };
let smoothedBbox = null;
let bboxConfidence = 0;

async function setupWebcam() {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: 'user' }
            });
            webcamElement.srcObject = stream;
            return new Promise((resolve) => {
                webcamElement.addEventListener('loadeddata', () => {
                    canvasElement.width = webcamElement.videoWidth || 400;
                    canvasElement.height = webcamElement.videoHeight || 300;
                    resolve();
                }, false);
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
        
        console.log("Loading MobileNet...");
        mobilenetModule = await mobilenet.load({version: 2, alpha: 1.0});

        console.log("Loading COCO-SSD...");
        objectDetector = await cocoSsd.load();

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

// 辅助函数：根据 bbox 从视频帧裁剪图像张量
function cropImageToBox(videoElement, bbox) {
    const [x, y, width, height] = bbox;
    const offscreenCanvas = document.createElement('canvas');
    offscreenCanvas.width = width;
    offscreenCanvas.height = height;
    const offCtx = offscreenCanvas.getContext('2d');
    
    offCtx.drawImage(
        videoElement, 
        x, y, width, height, // Source rect
        0, 0, width, height  // Dest rect
    );
    return tf.browser.fromPixels(offscreenCanvas);
}

// 获取当前画面中面积最大的候选框（带时序平滑和丢帧容忍）
async function getLargestObjectBox() {
    const predictions = await objectDetector.detect(webcamElement);
    let currentLargest = null;
    
    if (predictions.length > 0) {
        currentLargest = predictions.reduce((prev, current) => {
            const prevArea = prev.bbox[2] * prev.bbox[3];
            const currArea = current.bbox[2] * current.bbox[3];
            return (prevArea > currArea) ? prev : current;
        });
    }

    if (currentLargest) {
        if (!smoothedBbox) {
            smoothedBbox = [...currentLargest.bbox];
        } else {
            // 指数移动平均 (EMA) 平滑抗抖动
            const alpha = 0.3; // 平滑系数 (0~1)，越小越平滑但跟随越慢
            for (let i = 0; i < 4; i++) {
                smoothedBbox[i] = smoothedBbox[i] * (1 - alpha) + currentLargest.bbox[i] * alpha;
            }
        }
        bboxConfidence = 15; // 允许框丢失的最大帧数（容错生命周期）
    } else {
        if (smoothedBbox) {
            bboxConfidence--; // 如果当前帧没检测到，生命值衰减
            if (bboxConfidence <= 0) {
                smoothedBbox = null; // 彻底丢失
            }
        }
    }

    return smoothedBbox ? { bbox: smoothedBbox } : null;
}

// 收集样本
const addExample = async (classId) => {
    const largestObj = await getLargestObjectBox();
    let img;
    
    if (largestObj) {
        img = cropImageToBox(webcamElement, largestObj.bbox);
    } else {
        img = tf.browser.fromPixels(webcamElement);
    }

    const features = mobilenetModule.infer(img, true);
    classifier.addExample(features, classId);
    
    counts[classId]++;
    if(classId === CLASS_A) countAEl.innerText = counts[CLASS_A];
    if(classId === CLASS_B) countBEl.innerText = counts[CLASS_B];

    img.dispose();
};

// 持续预测
const predictLoop = async () => {
    const largestObj = await getLargestObjectBox();
    
    ctx.clearRect(0, 0, canvasElement.width, canvasElement.height);

    if (largestObj) {
        const [x, y, width, height] = largestObj.bbox;
        
        // 绘制准星框 (注意 canvas 有 scaleX(-1) 翻转)
        ctx.strokeStyle = '#00ff00';
        ctx.lineWidth = 4;
        ctx.strokeRect(x, y, width, height);

        if (classifier.getNumClasses() > 0) {
            const croppedImg = cropImageToBox(webcamElement, largestObj.bbox);
            const features = mobilenetModule.infer(croppedImg, true);
            const res = await classifier.predictClass(features, 3);
            
            resultEl.innerText = res.label;
            const confidence = (res.confidences[res.label] * 100).toFixed(1);
            confidenceFill.style.width = `${confidence}%`;
            confidenceText.innerText = `置信度: ${confidence}%`;
            
            // 绘制文字 (需解决镜像反转问题)
            ctx.save();
            ctx.translate(canvasElement.width, 0);
            ctx.scale(-1, 1);
            const mirrorX = canvasElement.width - x - width;
            
            ctx.fillStyle = 'rgba(0, 255, 0, 0.7)';
            ctx.fillRect(mirrorX, y - 30, width, 30);
            ctx.fillStyle = '#000000';
            ctx.font = '20px Inter';
            ctx.fillText(`${res.label} (${confidence}%)`, mirrorX + 5, y - 8);
            ctx.restore();

            croppedImg.dispose();
        } else {
            ctx.save();
            ctx.translate(canvasElement.width, 0);
            ctx.scale(-1, 1);
            const mirrorX = canvasElement.width - x - width;
            
            ctx.fillStyle = 'rgba(255, 0, 0, 0.7)';
            ctx.fillRect(mirrorX, y - 30, width, 30);
            ctx.fillStyle = '#ffffff';
            ctx.font = '16px Inter';
            ctx.fillText(`追踪目标 (请录入)`, mirrorX + 5, y - 10);
            ctx.restore();

            resultEl.innerText = "发现目标，等待学习...";
            confidenceFill.style.width = "0%";
            confidenceText.innerText = `置信度: 0%`;
        }
    } else {
        resultEl.innerText = "正在寻找物体...";
        confidenceFill.style.width = "0%";
        confidenceText.innerText = `置信度: 0%`;
    }

    requestAnimationFrame(predictLoop);
};

btnA.addEventListener('click', () => addExample(CLASS_A));
btnB.addEventListener('click', () => addExample(CLASS_B));

btnReset.addEventListener('click', () => {
    classifier.clearAllClasses();
    counts = { [CLASS_A]: 0, [CLASS_B]: 0 };
    countAEl.innerText = 0;
    countBEl.innerText = 0;
});

window.onload = init;
