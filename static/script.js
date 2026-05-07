const uploadZone = document.getElementById('uploadZone');
const fileInput = document.getElementById('fileInput');
const previewImage = document.getElementById('previewImage');
const uploadContent = document.getElementById('uploadContent');
const analyzeBtn = document.getElementById('analyzeBtn');
const results = document.getElementById('results');
const resultsList = document.getElementById('resultsList');
const loader = document.getElementById('loader');

let selectedFile = null;
let isProcessed = false; // флаг, обработано ли уже текущее фото

// смена начального HTML контента загрузки без кнопки
uploadContent.innerHTML = `
    <div class="upload-icon">⬆️</div>
    <p class="upload-title">Загрузите или перетащите фото сюда</p>
    <p class="upload-hint">Поддерживаются JPG, PNG, WebP. Чем чётче фото, тем точнее результат.</p>
`;

// клик по зоне загрузки открывает проводник
uploadZone.addEventListener('click', () => {
    fileInput.click();
});

// выбор файла
fileInput.addEventListener('change', (e) => {
    if (e.target.files[0]) {
        handleFile(e.target.files[0]);
    }
});

// drag & drop
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
    isProcessed = false; // сбрасываем флаг обработки для нового фото

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

        // Возвращаем кнопке исходное состояние
        analyzeBtn.classList.remove('hidden');
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = getAnalyzeBtnText();
        results.classList.add('hidden');
        resultsList.innerHTML = '';
        uploadZone.classList.remove('dragover');
    };
    reader.readAsDataURL(file);
}

// Функция получения текста кнопки в зависимости от языка
function getAnalyzeBtnText() {
    const currentLang = localStorage.getItem('lang') || 'ru';
    return currentLang === 'ru' ? '🔍 Определить одежду' : '🔍 Detect clothing';
}

// отправка на сервер
analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile || isProcessed) return; // если уже обработано - не запускаем

    loader.classList.remove('hidden');
    results.classList.add('hidden');
    analyzeBtn.disabled = true;
    analyzeBtn.innerHTML = currentLang === 'ru' ? '⏳ Обработка...' : '⏳ Processing...';

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
        const response = await fetch('/detect', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        showResults(data.detections);
        
        // После успешной обработки деактивируем кнопку навсегда (до нового фото)
        isProcessed = true;
        analyzeBtn.disabled = true;
        analyzeBtn.innerHTML = currentLang === 'ru' ? '✅ Обработка завершена' : '✅ Processing completed';
    } catch (err) {
        resultsList.innerHTML = '<p style="color: #ef4444">Ошибка. Повторите попытку позже</p>';
        results.classList.remove('hidden');
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = getAnalyzeBtnText();
    } finally {
        loader.classList.add('hidden');
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
let currentLang = localStorage.getItem('lang') || 'ru';

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

// Мультиязычность
const translations = {
    ru: {
        title: "Определение одежды",
        subtitle: "Загрузите фото — ИИ определит тип одежды и выделит её на изображении.",
        uploadTitle: "Загрузите или перетащите фото сюда",
        uploadHint: "Поддерживаются JPG, PNG, WebP. Чем чётче фото, тем точнее результат.",
        analyzeBtn: "🔍 Определить одежду",
        resultsTitle: "Результат анализа",
        analyzing: "Анализируем...",
        history: "История",
        emptyHistory: "История пуста",
        footer: "Умное распознавание одежды с помощью ИИ",
        notFound: "Ничего не найдено",
        error: "Ошибка. Повторите попытку позже",
        processing: "⏳ Обработка...",
        completed: "✅ Обработка завершена"
    },
    en: {
        title: "Clothing Detection",
        subtitle: "Upload a photo — AI will detect the type of clothing.",
        uploadTitle: "Upload or drag & drop photo here",
        uploadHint: "Supports JPG, PNG, WebP. The clearer the photo, the better the result.",
        analyzeBtn: "🔍 Detect clothing",
        resultsTitle: "Analysis result",
        analyzing: "Analyzing...",
        history: "History",
        emptyHistory: "History is empty",
        footer: "Smart clothing recognition powered by AI",
        notFound: "Nothing found",
        error: "Error. Please try again",
        processing: "⏳ Processing...",
        completed: "✅ Processing completed"
    }
};

const langBtn = document.getElementById('langBtn');

function applyTranslations(lang) {
    document.querySelectorAll('[data-i18n]').forEach(el => {
        const key = el.getAttribute('data-i18n');
        if (translations[lang][key]) {
            el.textContent = translations[lang][key];
        }
    });
    
    // Обновляем текст кнопки анализа, если она не в специальном состоянии
    if (!isProcessed && analyzeBtn.disabled === false) {
        analyzeBtn.innerHTML = translations[lang].analyzeBtn;
    } else if (isProcessed) {
        analyzeBtn.innerHTML = translations[lang].completed;
    } else if (analyzeBtn.disabled === true && !isProcessed) {
        analyzeBtn.innerHTML = translations[lang].processing;
    }
    
    langBtn.textContent = lang === 'ru' ? 'EN' : 'RU';
    document.documentElement.lang = lang;
}

// Применить при загрузке
applyTranslations(currentLang);

langBtn.addEventListener('click', () => {
    currentLang = currentLang === 'ru' ? 'en' : 'ru';
    localStorage.setItem('lang', currentLang);
    applyTranslations(currentLang);
});