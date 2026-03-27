/**
 * script.js — Dashboard Logic
 * ─────────────────────────────
 * Handles auth, API calls, stream control, and status polling.
 */

// ── Auth ──
const TOKEN = localStorage.getItem('token');
const USERNAME = localStorage.getItem('username');

if (!TOKEN) {
    window.location.href = '/login';
}

// Set username in navbar
const navUser = document.getElementById('nav-user');
if (navUser && USERNAME) {
    navUser.textContent = `👤 ${USERNAME}`;
}

function logout() {
    localStorage.removeItem('token');
    localStorage.removeItem('username');
    window.location.href = '/login';
}

// ── API Helper ──
async function api(endpoint, method = 'POST', body = null) {
    const opts = {
        method,
        headers: {
            'Authorization': `Bearer ${TOKEN}`,
        },
    };
    if (body && !(body instanceof FormData)) {
        opts.headers['Content-Type'] = 'application/json';
        opts.body = JSON.stringify(body);
    } else if (body instanceof FormData) {
        opts.body = body;
    }

    const res = await fetch(endpoint, opts);
    if (res.status === 401) {
        logout();
        return null;
    }
    return await res.json();
}

// ── Elements ──
const videoStream = document.getElementById('video-stream');
const videoPlaceholder = document.getElementById('video-placeholder');
const btnStart = document.getElementById('btn-start');
const btnStopTrack = document.getElementById('btn-stop-track');
const btnStop = document.getElementById('btn-stop');
const uploadArea = document.getElementById('upload-area');
const statusDot = document.getElementById('status-dot');
const statusState = document.getElementById('status-state');
const statusFps = document.getElementById('status-fps');
const statusSkip = document.getElementById('status-skip');

// ── Stream ──
function startStream() {
    videoStream.src = `/stream?token=${encodeURIComponent(TOKEN)}`;
    videoStream.style.display = 'block';
    videoPlaceholder.style.display = 'none';
    btnStop.disabled = false;
    btnStart.disabled = false;
}

function stopStream() {
    videoStream.src = '';
    videoStream.style.display = 'none';
    videoPlaceholder.style.display = 'block';
    btnStart.disabled = true;
    btnStopTrack.disabled = true;
    btnStop.disabled = true;
}

// ── Actions ──
function showUpload() {
    uploadArea.classList.toggle('active');
}

async function uploadVideo(input) {
    const file = input.files[0];
    if (!file) return;

    uploadArea.classList.remove('active');

    const formData = new FormData();
    formData.append('file', file);

    const data = await api('/api/upload-video', 'POST', formData);
    if (data) {
        startStream();
    }
}

async function startWebcam() {
    uploadArea.classList.remove('active');
    const data = await api('/api/start-webcam');
    if (data) {
        startStream();
    }
}

async function startTracking() {
    const data = await api('/api/start-tracking');
    if (data) {
        btnStart.disabled = true;
        btnStopTrack.disabled = false;
    }
}

async function stopTracking() {
    const data = await api('/api/stop-tracking');
    if (data) {
        btnStart.disabled = false;
        btnStopTrack.disabled = true;
    }
}

async function stopVideo() {
    await api('/api/stop-video');
    stopStream();
}

async function setQuality() {
    const skip = parseInt(document.getElementById('quality-select').value);
    await api('/api/set-quality', 'POST', { frame_skip: skip });
}

// ── Status Polling ──
async function pollStatus() {
    try {
        const data = await api('/api/status', 'GET');
        if (!data) return;

        statusState.textContent = data.state.charAt(0).toUpperCase() + data.state.slice(1);
        statusFps.textContent = data.fps.toFixed(1);
        statusSkip.textContent = data.frame_skip;

        // Update dot
        statusDot.className = 'dot';
        if (data.state === 'tracking') statusDot.classList.add('tracking');
        else if (data.state === 'playing') statusDot.classList.add('active');

        // Update button states
        if (data.state === 'idle') {
            btnStart.disabled = true;
            btnStopTrack.disabled = true;
            btnStop.disabled = true;
        } else if (data.state === 'playing') {
            btnStart.disabled = false;
            btnStopTrack.disabled = true;
            btnStop.disabled = false;
        } else if (data.state === 'tracking') {
            btnStart.disabled = true;
            btnStopTrack.disabled = false;
            btnStop.disabled = false;
        }
    } catch (e) {
        // ignore polling errors
    }
}

// Poll every 1s
setInterval(pollStatus, 1000);
pollStatus();
