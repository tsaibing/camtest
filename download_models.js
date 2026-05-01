const fs = require('fs');
const https = require('https');
const path = require('path');

const models = [
    {
        name: 'mobilenet_v2',
        baseUrl: 'https://storage.googleapis.com/tfjs-models/savedmodel/mobilenet_v2_1.0_224/'
    },
    {
        name: 'coco-ssd',
        baseUrl: 'https://storage.googleapis.com/tfjs-models/savedmodel/ssdlite_mobilenet_v2/'
    }
];

async function downloadFile(url, dest) {
    return new Promise((resolve, reject) => {
        https.get(url, (response) => {
            if (response.statusCode === 302 || response.statusCode === 301) {
                downloadFile(response.headers.location, dest).then(resolve).catch(reject);
                return;
            }
            if (response.statusCode !== 200) {
                reject(new Error(`Failed to download ${url}: ${response.statusCode}`));
                return;
            }
            const file = fs.createWriteStream(dest);
            response.pipe(file);
            file.on('finish', () => {
                file.close(resolve);
            });
        }).on('error', (err) => {
            fs.unlink(dest, () => {});
            reject(err);
        });
    });
}

async function fetchModel(modelInfo) {
    const dir = path.join(__dirname, 'models', modelInfo.name);
    if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
    }

    const modelJsonUrl = modelInfo.baseUrl + 'model.json';
    const modelJsonPath = path.join(dir, 'model.json');
    
    console.log(`Downloading ${modelJsonUrl}...`);
    await downloadFile(modelJsonUrl, modelJsonPath);
    
    const modelData = JSON.parse(fs.readFileSync(modelJsonPath, 'utf8'));
    const weightsManifest = modelData.weightsManifest;
    
    for (const group of weightsManifest) {
        for (const pathStr of group.paths) {
            const binUrl = modelInfo.baseUrl + pathStr;
            const binPath = path.join(dir, pathStr);
            console.log(`Downloading ${binUrl}...`);
            await downloadFile(binUrl, binPath);
        }
    }
    console.log(`Finished downloading ${modelInfo.name}`);
}

async function main() {
    for (const model of models) {
        try {
            await fetchModel(model);
        } catch(e) {
            console.error(e);
        }
    }
}

main();
