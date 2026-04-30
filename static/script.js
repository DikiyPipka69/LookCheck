const uploadZone = document.getElementById('uploadZone');
const fileInput = document.getElementById('fileInput');
const previewImage = document.getElementById('previewImage');
const uploadContent = document.getElementById('uploadContent');
const analyzeBtn = document.getElementById('analyzeBtn');
const results = document.getElementById('results');
const resultsList = document.getElementById('resultsList');
const loader = document.getElementById('loader');

let selectedFile = null;

// Меняем начальный HTML контента загрузки (без кнопки)
uploadContent.innerHTML = `
    <div class="upload-icon">⬆️</div>
    <p class="upload-title">Загрузите или перетащите фото сюда</p>
    <p class="upload-hint">Поддерживаются JPG, PNG, WebP. Чем чётче фото, тем точнее результат.</p>
`;

// Клик по зоне загрузки — всегда открывает диалог выбора файла
uploadZone.addEventListener('click', () => {
    fileInput.click();
});

// Выбор файла
fileInput.addEventListener('change', (e) => {
    if (e.target.files[0]) {
        handleFile(e.target.files[0]);
    }
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
    e.stopPropagation();
    uploadZone.classList.remove('dragover');
    if (e.dataTransfer.files[0]) {
        handleFile(e.dataTransfer.files[0]);
    }
});

function handleFile(file) {
    if (!file) return;
    selectedFile = file;

    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewImage.classList.remove('hidden');

        // Меняем текст в uploadContent на подсказку о повторной загрузке (без кнопки)
        uploadContent.innerHTML = `
            <div class="upload-icon">🔄</div>
            <p class="upload-title">Фото загружено</p>
            <p class="upload-hint">Перетащите или загрузите новое фото</p>
        `;

        analyzeBtn.classList.remove('hidden');
        results.classList.add('hidden');
        resultsList.innerHTML = '';
        uploadZone.classList.remove('dragover');
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
        detections.forEach((d, i) => {
            const item = document.createElement('div');
            item.className = 'result-item';
            item.style.animationDelay = `${i * 0.1}s`;
            item.innerHTML = `
                <div class="result-top">
                    <span class="result-class">👗 ${d.class}</span>
                    <span class="result-confidence">${d.confidence}%</span>
                </div>
                <div class="progress-bar-bg">
                    <div class="progress-bar-fill" data-width="${d.confidence}"></div>
                </div>
            `;
            resultsList.appendChild(item);
        });

        setTimeout(() => {
document.querySelectorAll('.progress-bar-fill').forEach(bar => {
                bar.style.width = bar.dataset.width + '%';
            });
        }, 50);
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
    try {
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
                            <div style="display:flex;align-items:center;gap:8px">
                                <div style="width:80px;height:5px;background:#2d2d4e;border-radius:10px;overflow:hidden">
                                    <div style="width:${d.confidence}%;height:100%;background:linear-gradient(90deg,#7c3aed,#a855f7);border-radius:10px"></div>
                                </div>
                                <span style="color:#a855f7">${d.confidence}%</span>
                            </div>
                        </div>
                    `).join('')
                }
            </div>
        `).join('');
    } catch (err) {
        historyList.innerHTML = '<p class="empty-history">Ошибка загрузки истории</p>';
    }
}

// Тёмная/светлая тема
const themeBtn = document.getElementById('themeBtn');
const body = document.body;

if (localStorage.getItem('theme') === 'light') {
    body.classList.add('light');
    themeBtn.textContent = '🌙';
}

themeBtn.addEventListener('click', () => {
    body.classList.toggle('light');
    const isLight = body.classList.contains('light');
    themeBtn.textContent = isLight ? '🌙' : '☀️';
    localStorage.setItem('theme', isLight ? 'light' : 'dark');
});