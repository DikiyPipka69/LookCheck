const uploadZone = document.getElementById('uploadZone');
const fileInput = document.getElementById('fileInput');
const previewImage = document.getElementById('previewImage');
const uploadContent = document.getElementById('uploadContent');
const analyzeBtn = document.getElementById('analyzeBtn');
const results = document.getElementById('results');
const resultsList = document.getElementById('resultsList');
const loader = document.getElementById('loader');

let selectedFile = null;
let isProcessed = false;
let currentLang = localStorage.getItem('lang') || 'ru';

// Мультиязычность
const translations = {
    ru: {
        title: "Определение одежды",
        subtitle: "Загрузите фото — ИИ определит тип одежды и выделит её на изображении.",
        uploadTitle: "Загрузите или перетащите фото сюда",
        uploadHint: "Поддерживаются JPG, PNG, WebP. Чем чётче фото, тем точнее результат.",
        chooseBtn: "Выбрать файл",
        analyzeBtn: "🔍 Определить одежду",
        resultsTitle: "Результат анализа",
        analyzing: "Анализируем...",
        history: "История",
        historySubtitle: "Все предыдущие запросы",
        emptyHistory: "История пуста",
        footer: "Умное распознавание одежды с помощью ИИ",
        notFound: "Ничего не найдено",
        error: "Ошибка. Повторите попытку позже",
        processing: "⏳ Обработка...",
        completed: "✅ Обработка завершена",
        uploaded: "Фото загружено",
        uploadedHint: "Перетащите или загрузите новое фото"
    },
    en: {
        title: "Clothing Detection",
        subtitle: "Upload a photo — AI will detect the type of clothing.",
        uploadTitle: "Upload or drag & drop photo here",
        uploadHint: "Supports JPG, PNG, WebP. The clearer the photo, the better the result.",
        chooseBtn: "Choose file",
        analyzeBtn: "🔍 Detect clothing",
        resultsTitle: "Analysis result",
        analyzing: "Analyzing...",
        history: "History",
        historySubtitle: "All previous requests",
        emptyHistory: "History is empty",
        footer: "Smart clothing recognition powered by AI",
        notFound: "Nothing found",
        error: "Error. Please try again",
        processing: "⏳ Processing...",
        completed: "✅ Processing completed",
        uploaded: "Photo uploaded",
        uploadedHint: "Drag & drop or upload a new photo"
    }
};

function t(key) {
    return translations[currentLang][key] || key;
}

function applyTranslations(lang) {
    document.querySelectorAll('[data-i18n]').forEach(el => {
        const key = el.getAttribute('data-i18n');
        if (translations[lang][key]) {
            el.textContent = translations[lang][key];
        }
    });

    if (!isProcessed && !analyzeBtn.disabled) {
        analyzeBtn.innerHTML = t('analyzeBtn');
    } else if (isProcessed) {
        analyzeBtn.innerHTML = t('completed');
    } else if (analyzeBtn.disabled && !isProcessed) {
        analyzeBtn.innerHTML = t('processing');
    }

    langBtn.textContent = lang === 'ru' ? 'EN' : 'RU';
    document.documentElement.lang = lang;
}

// Загрузка зоны
uploadZone.addEventListener('click', () => {
    fileInput.click();
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files[0]) handleFile(e.target.files[0]);
});

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
    if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
});

function handleFile(file) {
    if (!file) return;
    selectedFile = file;
    isProcessed = false;

    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewImage.classList.remove('hidden');
        uploadContent.innerHTML = `
            <div class="upload-icon">🔄</div>
            <p class="upload-title" data-i18n-dynamic="uploaded">${t('uploaded')}</p>
            <p class="upload-hint" data-i18n-dynamic="uploadedHint">${t('uploadedHint')}</p>
        `;
        analyzeBtn.classList.remove('hidden');
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = t('analyzeBtn');
        results.classList.add('hidden');
        resultsList.innerHTML = '';
    };
    reader.readAsDataURL(file);
}

analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile || isProcessed) return;

    loader.classList.remove('hidden');
    results.classList.add('hidden');
    analyzeBtn.disabled = true;
    analyzeBtn.innerHTML = t('processing');

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
        const response = await fetch('/detect', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        showResults(data.detections);
        isProcessed = true;
        analyzeBtn.disabled = true;
        analyzeBtn.innerHTML = t('completed');
    } catch (err) {
        resultsList.innerHTML = `<p style="color:#ef4444">${t('error')}</p>`;
        results.classList.remove('hidden');
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = t('analyzeBtn');
    } finally {
        loader.classList.add('hidden');
    }
});

function showResults(detections) {
    resultsList.innerHTML = '';

    if (detections.length === 0) {
        resultsList.innerHTML = `<p style="color:#6b7280">${t('notFound')}</p>`;
    } else {
        detections.forEach((d, i) => {
            const item = document.createElement('div');
            item.className = 'result-item';
            item.style.animationDelay = `${i * 0.1}s`;

            const query = encodeURIComponent(d.class);
            const markets = [
                { name: 'Wildberries', url: `https://www.wildberries.ru/catalog/0/search.aspx?search=${query}`, color: '#cb11ab' },
                { name: 'Ozon', url: `https://www.ozon.ru/search/?text=${query}`, color: '#005bff' },
                { name: 'Яндекс', url: `https://market.yandex.ru/search?text=${query}`, color: '#ffcc00' },
            ];

            item.innerHTML = `
                <div class="result-top">
                    <span class="result-class">👗 ${d.class} (${d.color})</span>
                    <span class="result-confidence">${d.confidence}%</span>
                </div>
                <div class="progress-bar-bg">
                    <div class="progress-bar-fill" data-width="${d.confidence}"></div>
                </div>
                <div class="market-links">
                    ${markets.map(m => `
                        <a href="${m.url}" target="_blank" class="market-btn" style="border-color:${m.color};color:${m.color}">
                            ${m.name}
                        </a>
                    `).join('')}
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

// НАВИГАЦИЯ
const mainPage = document.getElementById('mainPage');
const historyPage = document.getElementById('historyPage');
const historyBtn = document.getElementById('historyBtn');
const historyGrid = document.getElementById('historyGrid');
const emptyHistory = document.getElementById('emptyHistory');
const imagePreview = document.getElementById('imagePreview');
const previewPopupImg = document.getElementById('previewPopupImg');

historyBtn.addEventListener('click', () => {
    mainPage.classList.add('hidden');
    historyPage.classList.remove('hidden');
    loadHistoryGrid();
});

document.querySelector('.logo').addEventListener('click', (e) => {
    e.preventDefault();
    historyPage.classList.add('hidden');
    mainPage.classList.remove('hidden');

    // Сброс состояния
    selectedFile = null;
    isProcessed = false;
    previewImage.classList.add('hidden');
    previewImage.src = '';
    analyzeBtn.classList.add('hidden');
    analyzeBtn.disabled = false;
    analyzeBtn.innerHTML = t('analyzeBtn');
    results.classList.add('hidden');
    resultsList.innerHTML = '';
    loader.classList.add('hidden');
    uploadContent.innerHTML = `
        <div class="upload-icon">⬆️</div>
        <p class="upload-title" data-i18n="uploadTitle">${t('uploadTitle')}</p>
        <p class="upload-hint" data-i18n="uploadHint">${t('uploadHint')}</p>
    `;
});

async function loadHistoryGrid() {
    try {
        const response = await fetch('/history');
        const data = await response.json();

        historyGrid.innerHTML = '';

        if (data.history.length === 0) {
            emptyHistory.classList.remove('hidden');
            return;
        }

        emptyHistory.classList.add('hidden');

        data.history.forEach((item, i) => {
            const card = document.createElement('div');
            card.className = 'history-card';
            card.style.animationDelay = `${i * 0.05}s`;

            const thumb = item.image_url
                ? `<img class="history-card-thumb" src="${item.image_url}" alt="photo">`
                : `<div class="history-card-thumb-placeholder">👗</div>`;

            const detectionsHtml = item.detections.length === 0
                ? `<p style="color:#6b7280;font-size:0.85em">${t('notFound')}</p>`
                : item.detections.map(d => `
                    <div class="history-card-detection">
                        <span>👗 ${d.class}</span>
                        <div style="display:flex;align-items:center;gap:6px">
                            <div class="history-card-bar">
                                <div class="history-card-bar-fill" style="width:${d.confidence}%"></div>
                            </div>
                            <span style="color:#a855f7;font-size:0.8em">${d.confidence}%</span>
                        </div>
                    </div>
                `).join('');

            card.innerHTML = `
                ${thumb}
                <div class="history-card-time">${item.time}</div>
                <div class="history-card-file">📎 ${item.filename}</div>
                <div class="history-card-detections">${detectionsHtml}</div>
            `;

            if (item.image_url) {
                card.addEventListener('mouseenter', () => {
                    previewPopupImg.src = item.image_url;
                    imagePreview.classList.remove('hidden');
                });

                card.addEventListener('mousemove', (e) => {
                    const x = e.clientX + 20;
                    const y = e.clientY - 120;
                    imagePreview.style.left = `${Math.min(x, window.innerWidth - 260)}px`;
                    imagePreview.style.top = `${Math.max(y, 10)}px`;
                });

                card.addEventListener('mouseleave', () => {
                    imagePreview.classList.add('hidden');
                });
            }

            historyGrid.appendChild(card);
        });
    } catch (err) {
        console.error('Ошибка загрузки истории:', err);
    }
}

// Тема
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

// Язык
const langBtn = document.getElementById('langBtn');

applyTranslations(currentLang);

langBtn.addEventListener('click', () => {
    currentLang = currentLang === 'ru' ? 'en' : 'ru';
    localStorage.setItem('lang', currentLang);
    applyTranslations(currentLang);

    // Если открыта страница истории — перерисовать карточки
    if (!historyPage.classList.contains('hidden')) {
        loadHistoryGrid();
    }
});

// Переводим динамические элементы
document.querySelectorAll('[data-i18n-dynamic]').forEach(el => {
    const key = el.getAttribute('data-i18n-dynamic');
    if (translations[lang][key]) {
        el.textContent = translations[lang][key];
    }
});