/**
 * Surgical Annotator — Frontend Application
 * Handles image upload, SAM 2 segmentation, VLM organ identification,
 * annotation CRUD, and COCO export.
 */

// ==========================================================================
// Organ color map (matches CSS variables)
// ==========================================================================
const ORGAN_COLORS = {
    liver:           '#ef4444',
    gallbladder:     '#22c55e',
    stomach:         '#f97316',
    spleen:          '#a855f7',
    pancreas:        '#eab308',
    colon:           '#06b6d4',
    small_intestine: '#ec4899',
    appendix:        '#14b8a6',
    kidney:          '#f43f5e',
    adrenal_gland:   '#f59e0b',
    omentum:         '#84cc16',
    mesentery:       '#6366f1',
    diaphragm:       '#0ea5e9',
    bladder:         '#d946ef',
    uterus:          '#fb923c',
    ovary:           '#a3e635',
    peritoneum:      '#64748b',
    unknown:         '#475569',
};

const ORGAN_LIST = Object.keys(ORGAN_COLORS);

// ==========================================================================
// Application State
// ==========================================================================
const state = {
    images: [],
    currentImage: null,       // { id, filename, url, width, height, ... }
    annotations: [],          // annotations for current image
    pendingMask: null,        // { mask_png_b64, mask_rle, mask_polygon, bbox, area, score, click_x, click_y }
    pendingLabel: null,       // { label, confidence, spatial_context }
    isSegmenting: false,
    isIdentifying: false,
};

// ==========================================================================
// DOM References
// ==========================================================================
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const dom = {
    // Header
    statusDotSam:     $('#status-dot-sam'),
    statusDotOllama:  $('#status-dot-ollama'),
    btnExport:        $('#btn-export'),

    // Left sidebar
    uploadZone:       $('#upload-zone'),
    fileInput:        $('#file-input'),
    imageList:        $('#image-list'),
    imageCount:       $('#image-count'),

    // Canvas
    canvasArea:       $('#canvas-area'),
    canvasEmpty:      $('#canvas-empty'),
    canvasContainer:  $('#canvas-container'),
    canvasImage:      $('#canvas-image'),
    canvasOverlay:    $('#canvas-overlay'),
    loadingOverlay:   $('#loading-overlay'),
    loadingText:      $('#loading-text'),
    clickMarkers:     $('#click-markers'),

    // Canvas toolbar
    canvasToolbar:    $('#canvas-toolbar'),
    btnClearMask:     $('#btn-clear-mask'),
    btnIdentify:      $('#btn-identify'),
    btnSaveAnn:       $('#btn-save-annotation'),  // matches HTML id

    // Right sidebar
    annotationList:   $('#annotation-list'),
    annotationCount:  $('#annotation-count'),
    annotationsEmpty: $('#annotations-empty'),

    // Toast
    toastContainer:   $('#toast-container'),

    // Export modal
    exportModal:      $('#export-modal'),
    exportModalBody:  $('#export-modal-body'),
    exportModalClose: $('#export-modal-close'),
    exportModalDownload: $('#export-modal-download'),
};


// ==========================================================================
// API Client
// ==========================================================================
const api = {
    async get(url) {
        const res = await fetch(url);
        if (!res.ok) throw new Error(`GET ${url} failed: ${res.status}`);
        return res.json();
    },

    async post(url, body) {
        const res = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || `POST ${url} failed: ${res.status}`);
        }
        return res.json();
    },

    async put(url, body) {
        const res = await fetch(url, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || `PUT ${url} failed: ${res.status}`);
        }
        return res.json();
    },

    async delete(url) {
        const res = await fetch(url, { method: 'DELETE' });
        if (!res.ok) throw new Error(`DELETE ${url} failed: ${res.status}`);
        return res.json();
    },

    async upload(file) {
        const formData = new FormData();
        formData.append('file', file);
        const res = await fetch('/api/upload/image', { method: 'POST', body: formData });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || 'Upload failed');
        }
        return res.json();
    },
};


// ==========================================================================
// Toast Notifications
// ==========================================================================
function showToast(message, type = 'info', duration = 3500) {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    dom.toastContainer.appendChild(toast);

    setTimeout(() => {
        toast.classList.add('toast-out');
        toast.addEventListener('animationend', () => toast.remove());
    }, duration);
}


// ==========================================================================
// Service Health Check
// ==========================================================================
async function checkHealth() {
    try {
        const data = await api.get('/api/health');
        const sam = data.services.sam2;
        const ollama = data.services.vlm;

        dom.statusDotSam.className = `status-dot ${sam.ready ? 'active' : 'error'}`;
        dom.statusDotOllama.className = `status-dot ${ollama.available ? 'active' : 'warning'}`;

        if (!sam.ready) showToast('SAM 2 model is not loaded — segmentation unavailable.', 'error');
        if (!ollama.available) showToast('Ollama is not running — organ identification will return "unknown".', 'warning');
    } catch {
        dom.statusDotSam.className = 'status-dot error';
        dom.statusDotOllama.className = 'status-dot error';
    }
}


// ==========================================================================
// Image Library
// ==========================================================================
async function loadImages() {
    try {
        const data = await api.get('/api/images');
        state.images = data.images || [];
        renderImageList();
    } catch (err) {
        showToast('Failed to load images.', 'error');
    }
}

function renderImageList() {
    dom.imageCount.textContent = state.images.length;

    if (state.images.length === 0) {
        dom.imageList.innerHTML = '';
        return;
    }

    dom.imageList.innerHTML = state.images.map(img => `
        <div class="image-thumb ${state.currentImage?.id === img.id ? 'active' : ''}"
             data-id="${img.id}" data-url="${img.url || ''}"
             data-width="${img.width}" data-height="${img.height}">
            <img src="${img.url || ''}" alt="${img.filename}" loading="lazy">
            <div class="image-thumb-info">
                <div class="name">${img.filename}</div>
                <div class="meta">${img.width}×${img.height}</div>
            </div>
            ${img.annotation_count > 0 ? `<span class="ann-count">${img.annotation_count}</span>` : ''}
        </div>
    `).join('');

    // Attach click handlers
    dom.imageList.querySelectorAll('.image-thumb').forEach(thumb => {
        thumb.addEventListener('click', () => selectImage(thumb.dataset));
    });
}


// ==========================================================================
// Image Selection
// ==========================================================================
async function selectImage(data) {
    const { id, url, width, height } = data;

    state.currentImage = { id, url, width: +width, height: +height };
    clearPendingMask();

    // Update active thumb
    dom.imageList.querySelectorAll('.image-thumb').forEach(t => {
        t.classList.toggle('active', t.dataset.id === id);
    });

    // Show canvas
    dom.canvasEmpty.style.display = 'none';
    dom.canvasContainer.style.display = 'flex';
    dom.canvasToolbar.style.display = 'flex';

    // Load image
    dom.canvasImage.src = url;
    dom.canvasImage.onload = () => {
        fitOverlay();
    };

    // Load annotations for this image
    await loadAnnotations(id);
}

function fitOverlay() {
    const img = dom.canvasImage;
    const canvas = dom.canvasOverlay;

    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    canvas.style.width = img.clientWidth + 'px';
    canvas.style.height = img.clientHeight + 'px';
    canvas.style.left = img.offsetLeft + 'px';
    canvas.style.top = img.offsetTop + 'px';

    redrawOverlay();
}


// ==========================================================================
// Canvas Click → Segmentation
// ==========================================================================
dom.canvasOverlay.addEventListener('click', async (e) => {
    if (!state.currentImage || state.isSegmenting) return;

    const canvasRect = dom.canvasOverlay.getBoundingClientRect();
    const containerRect = dom.canvasContainer.getBoundingClientRect();
    const scaleX = dom.canvasOverlay.width / canvasRect.width;
    const scaleY = dom.canvasOverlay.height / canvasRect.height;

    const x = Math.round((e.clientX - canvasRect.left) * scaleX);
    const y = Math.round((e.clientY - canvasRect.top) * scaleY);

    // Marker position relative to the canvas container
    const markerX = e.clientX - containerRect.left;
    const markerY = e.clientY - containerRect.top;

    await runSegmentation(x, y, markerX, markerY);
});

async function runSegmentation(x, y, markerX, markerY) {
    state.isSegmenting = true;
    showLoading('Segmenting...');

    try {
        const result = await api.post('/api/segment/image', {
            image_id: state.currentImage.id,
            x, y,
        });

        state.pendingMask = {
            mask_png_b64: result.mask_png_b64,
            mask_rle: result.mask_rle,
            mask_polygon: result.mask_polygon,
            bbox: result.bbox,
            area: result.area,
            score: result.score,
            click_x: x,
            click_y: y,
        };
        state.pendingLabel = null;

        // Place click marker
        dom.clickMarkers.innerHTML = '';
        if (markerX !== undefined && markerY !== undefined) {
            const marker = document.createElement('div');
            marker.className = 'click-marker';
            marker.style.left = markerX + 'px';
            marker.style.top = markerY + 'px';
            dom.clickMarkers.appendChild(marker);
        }

        // Draw mask on overlay
        drawMaskOverlay(result.mask_png_b64);

        // Enable toolbar buttons
        dom.btnIdentify.disabled = false;
        dom.btnSaveAnn.disabled = false;

        showToast(`Mask generated (score: ${result.score.toFixed(3)})`, 'success');
    } catch (err) {
        showToast(`Segmentation failed: ${err.message}`, 'error');
    } finally {
        state.isSegmenting = false;
        hideLoading();
    }
}

function drawMaskOverlay(maskB64) {
    const canvas = dom.canvasOverlay;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const img = new Image();
    img.onload = () => {
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

        // Draw existing approved annotation masks underneath
        redrawAnnotationMasks(ctx);
    };
    img.src = `data:image/png;base64,${maskB64}`;
}

function redrawOverlay() {
    const canvas = dom.canvasOverlay;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (state.pendingMask) {
        // Draw pending mask first, then annotation bboxes on top
        const img = new Image();
        img.onload = () => {
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            redrawAnnotationMasks(ctx);
        };
        img.src = `data:image/png;base64,${state.pendingMask.mask_png_b64}`;
    } else {
        redrawAnnotationMasks(ctx);
    }
}

function redrawAnnotationMasks(ctx) {
    // Draw bounding boxes for saved annotations
    state.annotations.forEach(ann => {
        if (!ann.bbox) return;
        try {
            const bbox = typeof ann.bbox === 'string' ? JSON.parse(ann.bbox) : ann.bbox;
            const [bx, by, bw, bh] = bbox;
            const color = ORGAN_COLORS[ann.label] || ORGAN_COLORS.unknown;

            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.setLineDash([6, 3]);
            ctx.strokeRect(bx, by, bw, bh);
            ctx.setLineDash([]);

            // Label
            ctx.font = '600 13px Inter, sans-serif';
            const labelText = ` ${ann.label} `;
            const tm = ctx.measureText(labelText);
            const lh = 18;

            ctx.fillStyle = color;
            ctx.fillRect(bx, by - lh, tm.width + 4, lh);
            ctx.fillStyle = '#0d1117';
            ctx.fillText(labelText, bx + 2, by - 4);
        } catch {}
    });
}


// ==========================================================================
// Organ Identification (VLM)
// ==========================================================================
async function runIdentification() {
    if (!state.pendingMask) return;

    state.isIdentifying = true;
    showLoading('Identifying organ...');

    try {
        const result = await api.post('/api/identify', {
            image_id: state.currentImage.id,
            mask_rle: state.pendingMask.mask_rle,
        });

        state.pendingLabel = {
            label: result.label,
            confidence: result.confidence,
            spatial_context: result.spatial_context,
        };

        const ctxStr = result.spatial_context ? `\nContext: ${result.spatial_context}` : '';
        showToast(
            `Identified: ${result.label} (${(result.confidence * 100).toFixed(0)}% confidence)${ctxStr}`,
            result.confidence >= 0.7 ? 'success' : 'warning'
        );
    } catch (err) {
        showToast(`Identification failed: ${err.message}`, 'error');
        state.pendingLabel = { label: 'unknown', confidence: 0.0 };
    } finally {
        state.isIdentifying = false;
        hideLoading();
    }
}


// ==========================================================================
// Save / CRUD Annotations
// ==========================================================================
async function saveAnnotation() {
    if (!state.pendingMask || !state.currentImage) return;

    const label = state.pendingLabel?.label || 'unknown';
    const confidence = state.pendingLabel?.confidence || 0.0;

    try {
        await api.post('/api/annotations', {
            image_id: state.currentImage.id,
            label,
            confidence,
            mask_rle: state.pendingMask.mask_rle,
            mask_polygon: state.pendingMask.mask_polygon,
            bbox: state.pendingMask.bbox,
            area: state.pendingMask.area,
            click_x: state.pendingMask.click_x,
            click_y: state.pendingMask.click_y,
            approved: false,
            corrected: false,
        });

        showToast(`Annotation saved: ${label}`, 'success');
        clearPendingMask();
        await Promise.all([loadAnnotations(state.currentImage.id), loadImages()]);
    } catch (err) {
        showToast(`Save failed: ${err.message}`, 'error');
    }
}

async function approveAnnotation(annId) {
    try {
        await api.put(`/api/annotations/${annId}`, { approved: true });
        showToast('Annotation approved ✓', 'success');
        await loadAnnotations(state.currentImage.id);
    } catch (err) {
        showToast(`Approve failed: ${err.message}`, 'error');
    }
}

async function correctAnnotation(annId, newLabel) {
    try {
        await api.put(`/api/annotations/${annId}`, { label: newLabel, corrected: true });
        showToast(`Label corrected to: ${newLabel}`, 'info');
        await loadAnnotations(state.currentImage.id);
    } catch (err) {
        showToast(`Correction failed: ${err.message}`, 'error');
    }
}

async function deleteAnnotation(annId) {
    try {
        await api.delete(`/api/annotations/${annId}`);
        showToast('Annotation deleted.', 'info');
        await Promise.all([loadAnnotations(state.currentImage.id), loadImages()]);
    } catch (err) {
        showToast(`Delete failed: ${err.message}`, 'error');
    }
}


// ==========================================================================
// Load & Render Annotations
// ==========================================================================
async function loadAnnotations(imageId) {
    try {
        const data = await api.get(`/api/annotations/image/${imageId}`);
        state.annotations = data.annotations || [];
        renderAnnotationList();
        redrawOverlay();
    } catch {
        state.annotations = [];
        renderAnnotationList();
    }
}

function renderAnnotationList() {
    const count = state.annotations.length;
    dom.annotationCount.textContent = count;

    if (count === 0) {
        dom.annotationsEmpty.style.display = '';
        dom.annotationList.querySelectorAll('.annotation-card').forEach(c => c.remove());
        return;
    }

    dom.annotationsEmpty.style.display = 'none';

    // Build HTML
    const html = state.annotations.map(ann => {
        const color = ORGAN_COLORS[ann.label] || ORGAN_COLORS.unknown;
        const approved = !!ann.approved;
        const confPct = (typeof ann.confidence === 'number' ? ann.confidence * 100 : 0).toFixed(0);
        const confClass = ann.confidence >= 0.7 ? 'high' : ann.confidence >= 0.4 ? 'medium' : 'low';

        // Parse bbox for display
        let bboxStr = '—';
        try {
            const bbox = typeof ann.bbox === 'string' ? JSON.parse(ann.bbox) : ann.bbox;
            if (bbox) bboxStr = `${bbox[0]},${bbox[1]} ${bbox[2]}×${bbox[3]}`;
        } catch {}

        const areaStr = ann.area ? Math.round(ann.area).toLocaleString() + 'px²' : '—';

        return `
            <div class="annotation-card ${approved ? 'approved' : 'pending'}" data-id="${ann.id}">
                <div class="annotation-card-header">
                    <span class="organ-label">
                        <span class="organ-dot" style="background: ${color}"></span>
                        ${ann.label}
                    </span>
                    <span class="confidence-badge confidence-${confClass}">${confPct}%</span>
                </div>

                <div class="annotation-card-meta">
                    <span>📐 ${areaStr}</span>
                    <span>📍 ${bboxStr}</span>
                </div>

                <!-- Label correction -->
                <select class="label-select" data-ann-id="${ann.id}" title="Correct label">
                    ${ORGAN_LIST.map(o =>
                        `<option value="${o}" ${o === ann.label ? 'selected' : ''}>${o.replace(/_/g, ' ')}</option>`
                    ).join('')}
                </select>

                <div class="annotation-card-actions">
                    ${!approved ? `
                        <button class="btn btn-success btn-sm" onclick="approveAnnotation('${ann.id}')">
                            ✓ Approve
                        </button>
                    ` : `
                        <button class="btn btn-ghost btn-sm" disabled>
                            ✓ Approved
                        </button>
                    `}
                    <button class="btn btn-danger btn-sm" onclick="deleteAnnotation('${ann.id}')">
                        🗑
                    </button>
                </div>
            </div>
        `;
    }).join('');

    // Preserve the empty state element, replace cards
    dom.annotationList.querySelectorAll('.annotation-card').forEach(c => c.remove());
    dom.annotationList.insertAdjacentHTML('beforeend', html);

    // Attach label correction handlers
    dom.annotationList.querySelectorAll('.label-select').forEach(sel => {
        sel.addEventListener('change', (e) => {
            correctAnnotation(e.target.dataset.annId, e.target.value);
        });
    });
}


// ==========================================================================
// Clear pending mask
// ==========================================================================
function clearPendingMask() {
    state.pendingMask = null;
    state.pendingLabel = null;
    dom.btnIdentify.disabled = true;
    dom.btnSaveAnn.disabled = true;

    // Clear click markers
    dom.clickMarkers.innerHTML = '';

    // Redraw overlay (just annotation boxes)
    if (state.currentImage) {
        redrawOverlay();
    }
}


// ==========================================================================
// Loading overlay
// ==========================================================================
function showLoading(text = 'Processing...') {
    dom.loadingText.textContent = text;
    dom.loadingOverlay.classList.add('visible');
}

function hideLoading() {
    dom.loadingOverlay.classList.remove('visible');
}


// ==========================================================================
// Upload handling
// ==========================================================================
async function handleFiles(files) {
    for (const file of files) {
        try {
            showToast(`Uploading ${file.name}...`, 'info', 2000);
            await api.upload(file);
            showToast(`Uploaded: ${file.name}`, 'success');
        } catch (err) {
            showToast(`Upload failed: ${err.message}`, 'error');
        }
    }
    await loadImages();
}

// Drag & drop
dom.uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dom.uploadZone.classList.add('drag-over');
});

dom.uploadZone.addEventListener('dragleave', () => {
    dom.uploadZone.classList.remove('drag-over');
});

dom.uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dom.uploadZone.classList.remove('drag-over');
    handleFiles(e.dataTransfer.files);
});

// Click to upload
dom.uploadZone.addEventListener('click', () => dom.fileInput.click());
dom.fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) handleFiles(e.target.files);
    e.target.value = '';  // Reset so same file can be re-uploaded
});


// ==========================================================================
// COCO Export
// ==========================================================================
dom.btnExport.addEventListener('click', async () => {
    try {
        showToast('Generating COCO export...', 'info', 2000);
        const result = await api.post('/api/export', {
            description: 'Surgical annotation dataset',
            version: '1.0',
        });

        const stats = result.stats;
        dom.exportModalBody.textContent =
            `Exported ${stats.annotations} annotation(s) across ${stats.images} image(s) in ${stats.categories} categories.`;
        dom.exportModalDownload.href = `/api/export/${result.filename}`;
        dom.exportModal.classList.add('visible');
    } catch (err) {
        showToast(`Export failed: ${err.message}`, 'error');
    }
});

dom.exportModalClose.addEventListener('click', () => {
    dom.exportModal.classList.remove('visible');
});


// ==========================================================================
// Toolbar buttons
// ==========================================================================
dom.btnClearMask.addEventListener('click', clearPendingMask);
dom.btnIdentify.addEventListener('click', runIdentification);
dom.btnSaveAnn.addEventListener('click', saveAnnotation);


// ==========================================================================
// Keyboard shortcuts
// ==========================================================================
document.addEventListener('keydown', (e) => {
    // Ignore if typing in an input
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA') return;

    switch (e.key) {
        case 'Escape':
            clearPendingMask();
            dom.exportModal.classList.remove('visible');
            break;
        case 'i':
        case 'I':
            if (state.pendingMask && !state.isIdentifying) runIdentification();
            break;
        case 's':
        case 'S':
            if (state.pendingMask) {
                e.preventDefault();
                saveAnnotation();
            }
            break;
    }
});


// ==========================================================================
// Window resize — refit overlay
// ==========================================================================
window.addEventListener('resize', () => {
    if (state.currentImage && dom.canvasImage.complete) {
        fitOverlay();
    }
});


// ==========================================================================
// Init
// ==========================================================================
async function init() {
    await checkHealth();
    await loadImages();

    // Poll health every 30s
    setInterval(checkHealth, 30000);
}

init();
