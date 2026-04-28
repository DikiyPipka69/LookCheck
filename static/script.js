const uploadZone = document.getElementById('uploadZone');
const fileInput = document.getElementById('fileInput');
const previewImage = document.getElementById('previewImage');
const uploadContent = document.getElementById('uploadContent');
const analyzeBtn = document.getElementById('analyzeBtn');
const results = document.getElementById('results');
const resultsList = document.getElementById('resultsList');
const loader = document.getElementById('loader');

let selectedFile = null;

// Клик по зоне загрузки
uploadZone.addEventListener('click', () => {
    if (!previewImage.classList.contains('hidden')) return;
    fileInput.click();
});

// Выбор файла
fileInput.addEventListener('change', (e) => {
    handleFile(e.target.files[0]);
});

// Drag & Drop
uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('dragover');
});

uploadZone.addEventListener('dragleave', () => {
    uploadZone.classList.remove('dragover');
});

uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
    handleFile(e.dataTransfer.files[0]);
});

function handleFile(file) {
    if (!file) return;
    selectedFile = file;

    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewImage.classList.remove('hidden');
        uploadContent.classList.add('hidden');
        analyzeBtn.classList.remove('hidden');
        results.classList.add('hidden');
    };
    reader.readAsDataURL(file);
}

// Отправка на сервер
analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    loader.classList.remove('hidden');
    results.classList.add('hidden');
    analyzeBtn.disabled = true;

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
        const response = await fetch('/detect', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        showResults(data.detections);
    } catch (err) {
        resultsList.innerHTML = '<p style="color: #ef4444">Ошибка. Попробуй снова.</p>';
        results.classList.remove('hidden');
    } finally {
        loader.classList.add('hidden');
        analyzeBtn.disabled = false;
    }
});

function showResults(detections) {
    resultsList.innerHTML = '';

    if (detections.length === 0) {
        resultsList.innerHTML = '<p style="color: #6b7280">Ничего не найдено</p>';
    } else {
        detections.forEach(d => {
            resultsList.innerHTML += `
                <div class="result-item">
                    <span class="result-class">👗 ${d.class}</span>
                    <span class="result-confidence">${d.confidence}%</span>
                </div>
            `;
        });
    }

    results.classList.remove('hidden');
}

// История
const historyBtn = document.getElementById('historyBtn');
const historyPanel = document.getElementById('historyPanel');
const closeHistory = document.getElementById('closeHistory');
const overlay = document.getElementById('overlay');
const historyList = document.getElementById('historyList');

historyBtn.addEventListener('click', async () => {
    historyPanel.classList.remove('hidden');
    historyPanel.classList.add('open');
    overlay.classList.remove('hidden');
    await loadHistory();
});

closeHistory.addEventListener('click', closePanel);
overlay.addEventListener('click', closePanel);

function closePanel() {
    historyPanel.classList.remove('open');
    overlay.classList.add('hidden');
    setTimeout(() => historyPanel.classList.add('hidden'), 400);
}

async function loadHistory() {
    const response = await fetch('/history');
    const data = await response.json();

    if (data.history.length === 0) {
        historyList.innerHTML = '<p class="empty-history">История пуста</p>';
        return;
    }

    historyList.innerHTML = data.history.map(item => `
        <div class="history-item">
            <div class="history-time">${item.time}</div>
            <div class="history-file">📎 ${item.filename}</div>
            ${item.detections.length === 0
                ? '<div style="color:#6b7280;font-size:0.9em">Ничего не найдено</div>'
                : item.detections.map(d => `
                    <div class="history-detection">
                        <span>👗 ${d.class}</span>
                        <span style="color:#a855f7">${d.confidence}%</span>
                    </div>
                `).join('')
            }
        </div>
    `).join('');
}
























