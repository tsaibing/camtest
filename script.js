// Constants
const WEBCAM_ID = 'webcam';
const CLASS_A = '物体 A';
const CLASS_B = '物体 B';
const IGNORED_CLASSES = ['person']; // 忽略检测的类别，防止人脸/人体干扰

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

let allPredictions = [];
let selectedPrediction = null; // 当前选中的物体
let bboxPersistence = 0;

// Performance optimization: skip frames for heavy tasks
let frameCounter = 0;
const DETECTION_INTERVAL = 4; // 每 4 帧运行一次目标检测 (COCO-SSD)
const PREDICTION_INTERVAL = 2; // 每 2 帧运行一次分类预测 (KNN)

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
        
        // 绑定点击事件用于选择物体
        canvasElement.addEventListener('mousedown', handleCanvasClick);
        
        // 开始视频循环
        requestAnimationFrame(predictLoop);

    } catch (err) {
        console.error("Initialization error:", err);
        document.querySelector('#loading-overlay p').innerText = `加载失败: ${err.message}`;
    }
}

// 处理画布点击：选择物体
function handleCanvasClick(event) {
    const rect = canvasElement.getBoundingClientRect();
    // 由于 CSS 镜像了 canvas，我们需要转换点击坐标
    // 逻辑坐标 = 原始坐标 (如果镜像了，x 需要反转)
    const scaleX = canvasElement.width / rect.width;
    const scaleY = canvasElement.height / rect.height;
    
    // 因为 CSS transform: scaleX(-1)，点击位置 rect.width - (clientX - rect.left) 是对应的原始坐标
    const clickX = (rect.width - (event.clientX - rect.left)) * scaleX;
    const clickY = (event.clientY - rect.top) * scaleY;

    let found = null;
    let minDistance = Infinity;

    allPredictions.forEach(pred => {
        const [x, y, width, height] = pred.bbox;
        if (clickX >= x && clickX <= x + width && clickY >= y && clickY <= y + height) {
            // 点击在框内
            const centerX = x + width / 2;
            const centerY = y + height / 2;
            const dist = Math.sqrt((clickX - centerX)**2 + (clickY - centerY)**2);
            if (dist < minDistance) {
                minDistance = dist;
                found = pred;
            }
        }
    });

    if (found) {
        selectedPrediction = { ...found, smoothedBbox: [...found.bbox] };
        console.log("Selected object:", selectedPrediction.class);
    } else {
        selectedPrediction = null;
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

// 更新并追踪物体
async function updateAllPredictions() {
    // 性能优化：只有在特定间隔帧才运行昂贵的目标检测
    if (frameCounter % DETECTION_INTERVAL === 0) {
        const rawPredictions = await objectDetector.detect(webcamElement);
        // 过滤掉黑名单中的类别
        allPredictions = rawPredictions.filter(p => !IGNORED_CLASSES.includes(p.class));
    }

    if (selectedPrediction) {
        // 尝试在当前预测中寻找最接近“已选中物体”的那个，实现简单追踪
        let bestMatch = null;
        let maxOverlap = 0;

        allPredictions.forEach(pred => {
            const overlap = calculateIOU(selectedPrediction.smoothedBbox, pred.bbox);
            if (overlap > maxOverlap) {
                maxOverlap = overlap;
                bestMatch = pred;
            }
        });

        if (bestMatch && maxOverlap > 0.2) {
            // 更新选中物体的位置（带平滑）
            const alpha = 0.4;
            for (let i = 0; i < 4; i++) {
                selectedPrediction.smoothedBbox[i] = selectedPrediction.smoothedBbox[i] * (1 - alpha) + bestMatch.bbox[i] * alpha;
            }
            selectedPrediction.class = bestMatch.class;
            bboxPersistence = 15; // 重置生命周期
        } else {
            bboxPersistence--;
            if (bboxPersistence <= 0) {
                selectedPrediction = null; // 彻底丢失目标
            }
        }
    } else if (allPredictions.length > 0 && frameCounter % DETECTION_INTERVAL === 0) {
        // 如果没手动选，默认锁定面积最大的（保持原有逻辑，仅在检测帧更新）
        const largest = allPredictions.reduce((prev, current) => {
            const prevArea = prev.bbox[2] * prev.bbox[3];
            const currArea = current.bbox[2] * current.bbox[3];
            return (prevArea > currArea) ? prev : current;
        });
        selectedPrediction = { ...largest, smoothedBbox: [...largest.bbox] };
        bboxPersistence = 15;
    }
}

// 辅助函数：计算交并比 (IOU) 用于物体追踪
function calculateIOU(bbox1, bbox2) {
    const x1 = Math.max(bbox1[0], bbox2[0]);
    const y1 = Math.max(bbox1[1], bbox2[1]);
    const x2 = Math.min(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2]);
    const y2 = Math.min(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3]);
    
    const width = Math.max(0, x2 - x1);
    const height = Math.max(0, y2 - y1);
    const intersection = width * height;
    const union = (bbox1[2] * bbox1[3]) + (bbox2[2] * bbox2[3]) - intersection;
    return intersection / union;
}

// 收集样本
const addExample = async (classId) => {
    if (!selectedPrediction) {
        alert("请先点击画面选中一个物体框进行训练！");
        return;
    }

    const img = cropImageToBox(webcamElement, selectedPrediction.smoothedBbox);
    const features = mobilenetModule.infer(img, true);
    classifier.addExample(features, classId);
    
    counts[classId]++;
    if(classId === CLASS_A) countAEl.innerText = counts[CLASS_A];
    if(classId === CLASS_B) countBEl.innerText = counts[CLASS_B];

    img.dispose();
};

// 持续预测
const predictLoop = async () => {
    frameCounter++;
    await updateAllPredictions();
    
    ctx.clearRect(0, 0, canvasElement.width, canvasElement.height);

    // 1. 绘制所有检测到的物体（淡色）
    allPredictions.forEach(pred => {
        const [x, y, width, height] = pred.bbox;
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
        ctx.setLineDash([5, 5]);
        ctx.lineWidth = 1;
        ctx.strokeRect(x, y, width, height);
        ctx.setLineDash([]);
    });

    // 2. 处理选中的物体
    if (selectedPrediction) {
        const [x, y, width, height] = selectedPrediction.smoothedBbox;
        
        // 绘制高亮准星框
        ctx.strokeStyle = '#00ff00';
        ctx.lineWidth = 4;
        ctx.strokeRect(x, y, width, height);

        if (classifier.getNumClasses() > 0) {
            // 性能优化：分类预测不需要每帧运行
            if (frameCounter % PREDICTION_INTERVAL === 0) {
                const croppedImg = cropImageToBox(webcamElement, selectedPrediction.smoothedBbox);
                const features = mobilenetModule.infer(croppedImg, true);
                const res = await classifier.predictClass(features, 3);
                
                resultEl.innerText = res.label;
                const confidence = (res.confidences[res.label] * 100).toFixed(1);
                confidenceFill.style.width = `${confidence}%`;
                confidenceText.innerText = `置信度: ${confidence}%`;
                
                croppedImg.dispose();
            }
            
            // 绘制文字（保持每帧更新位置）
            ctx.save();
            ctx.translate(canvasElement.width, 0);
            ctx.scale(-1, 1);
            const mirrorX = canvasElement.width - x - width;
            
            ctx.fillStyle = 'rgba(0, 255, 0, 0.7)';
            ctx.fillRect(mirrorX, y - 30, width, 30);
            ctx.fillStyle = '#000000';
            ctx.font = 'bold 16px Inter';
            ctx.fillText(`${resultEl.innerText}`, mirrorX + 5, y - 8);
            ctx.restore();
        } else {
            ctx.save();
            ctx.translate(canvasElement.width, 0);
            ctx.scale(-1, 1);
            const mirrorX = canvasElement.width - x - width;
            
            ctx.fillStyle = 'rgba(59, 130, 246, 0.8)';
            ctx.fillRect(mirrorX, y - 30, width, 30);
            ctx.fillStyle = '#ffffff';
            ctx.font = '14px Inter';
            ctx.fillText(`锁定: ${selectedPrediction.class}`, mirrorX + 5, y - 10);
            ctx.restore();

            resultEl.innerText = "已锁定，请开始学习...";
            confidenceFill.style.width = "0%";
            confidenceText.innerText = `置信度: 0%`;
        }
    } else {
        resultEl.innerText = "点击画面框选目标...";
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
