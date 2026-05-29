document.addEventListener("DOMContentLoaded", () => {
    // -----------------------------------------------------------------------
    // UI ELEMENTS
    // -----------------------------------------------------------------------
    const systemStatusInd = document.getElementById("system-status-indicator");
    
    // KPI Cards
    const kpiTotalVal = document.querySelector("#kpi-total-nodes .kpi-val");
    const kpiOnlineVal = document.querySelector("#kpi-online-nodes .kpi-val");
    const kpiPlayingVal = document.querySelector("#kpi-playing-nodes .kpi-val");
    
    // Master Actions
    const btnMasterBuffer = document.getElementById("btn-master-buffer");
    const btnMasterPlay = document.getElementById("btn-master-play");
    const btnMasterStop = document.getElementById("btn-master-stop");
    const masterStatusAlert = document.getElementById("master-status-alert");
    const masterStatusText = document.getElementById("master-status-text");

    // Upload & Library
    const dropZone = document.getElementById("drop-zone");
    const audioInput = document.getElementById("audio-file-input");
    const uploadProgressContainer = document.getElementById("upload-progress-container");
    const uploadProgressBar = document.getElementById("upload-progress-bar");
    const uploadProgressText = document.getElementById("upload-progress-text");
    const audioLibraryList = document.getElementById("audio-library-list");
    const btnRefreshLibrary = document.getElementById("btn-refresh-library");

    // Client Nodes
    const addNodeForm = document.getElementById("add-node-form");
    const nodeUrlInput = document.getElementById("node-url-input");
    const btnAddNode = document.getElementById("btn-add-node");
    const nodesGrid = document.getElementById("nodes-grid");
    const btnRefreshNodes = document.getElementById("btn-refresh-nodes");

    // Classification & Logs
    const classificationCard = document.getElementById("classification-results-card");
    const btnCloseClassification = document.getElementById("btn-close-classification");
    const clsFilenameText = document.getElementById("cls-filename");
    const clsStemTargetText = document.getElementById("cls-stem-target");
    const clsPresentBadge = document.getElementById("cls-present-badge");
    const clsConfidenceVal = document.getElementById("cls-confidence-val");
    
    const logFeed = document.getElementById("system-log-feed");
    const btnClearLogs = document.getElementById("btn-clear-logs");

    // -----------------------------------------------------------------------
    // GLOBAL STATE
    // -----------------------------------------------------------------------
    let selectedTrack = null;
    let registeredNodes = [];      // simple array of strings: urls
    let nodesStatusList = [];      // array of status objects from server
    let isPolling = false;
    let pollInterval = null;
    let _audioState = {};          // persists browser audio state across grid re-renders

    // -----------------------------------------------------------------------
    // INITIALIZATION & EVENT LISTENERS
    // -----------------------------------------------------------------------
    function init() {
        addLog("Dashboard initialized. Orchestrator mode active.", "system");
        
        // Load initial data
        fetchAudioLibrary();
        fetchModelLibrary();
        fetchNodesList().then(() => {
            refreshNodesStatus();
            fetchTrainingStatus();
        });

        // Setup polling (every 5 seconds)
        startStatusPolling();

        // Bind listeners
        btnRefreshLibrary.addEventListener("click", fetchAudioLibrary);
        btnRefreshNodes.addEventListener("click", refreshNodesStatus);
        btnClearLogs.addEventListener("click", () => {
            logFeed.innerHTML = "";
            addLog("Logs cleared.", "system");
        });

        // Add Node form
        addNodeForm.addEventListener("submit", handleAddNode);

        // Upload handlers
        setupUploadHandlers();

        // Master playback controls
        btnMasterBuffer.addEventListener("click", triggerMasterBuffer);
        btnMasterPlay.addEventListener("click", triggerMasterPlay);
        btnMasterStop.addEventListener("click", triggerMasterStop);

        // Training panel refresh button
        document.getElementById("btn-refresh-training").addEventListener("click", fetchTrainingStatus);

        // Model library handlers
        setupModelLibraryHandlers();

        // Close Classification Results
        btnCloseClassification.addEventListener("click", () => {
            classificationCard.classList.add("hidden");
        });

        // Update states initially
        updateControlStates();
    }

    // -----------------------------------------------------------------------
    // LOGGING UTILITY
    // -----------------------------------------------------------------------
    function addLog(message, type = "info") {
        const timestamp = new Date().toLocaleTimeString();
        const line = document.createElement("div");
        line.className = `log-line log-${type}`;
        line.innerText = `[${timestamp}] ${message}`;
        
        logFeed.appendChild(line);
        logFeed.scrollTop = logFeed.scrollHeight;
    }

    // -----------------------------------------------------------------------
    // UI CONTROL STATE ENFORCEMENT
    // -----------------------------------------------------------------------
    function updateControlStates() {
        // Master Buffer button enabled if a track is selected
        btnMasterBuffer.disabled = !selectedTrack;

        // Master Play button enabled if any node has buffered audio
        btnMasterPlay.disabled = !hasAnyBufferedNode();

        // Master Stop button enabled if any node is currently playing
        btnMasterStop.disabled = !hasAnyPlayingNode();

        // Update node card classify buttons depending on if a track is selected
        document.querySelectorAll(".btn-node-classify").forEach(btn => {
            btn.disabled = !selectedTrack;
        });
    }

    function hasAnyBufferedNode() {
        return nodesStatusList.some(n => n.online && n.buffered);
    }

    function hasAnyPlayingNode() {
        return nodesStatusList.some(n => n.online && n.playing);
    }

    // -----------------------------------------------------------------------
    // REST API REQUESTS
    // -----------------------------------------------------------------------

    // GET /api/nodes -> Fetch list of URL strings
    async function fetchNodesList() {
        try {
            const resp = await fetch("/api/nodes");
            if (!resp.ok) throw new Error("Failed to load nodes config");
            registeredNodes = await resp.json();
            kpiTotalVal.innerText = registeredNodes.length;
        } catch (e) {
            addLog(`Error fetching registered nodes list: ${e.message}`, "error");
        }
    }

    // GET /api/nodes/status -> poll endpoints of each node in parallel
    async function refreshNodesStatus() {
        if (isPolling) return;
        isPolling = true;
        btnRefreshNodes.disabled = true;
        btnRefreshNodes.innerText = "Querying...";
        
        try {
            const resp = await fetch("/api/nodes/status");
            if (!resp.ok) throw new Error("Failed to retrieve statuses");
            nodesStatusList = await resp.json();
            
            updateNodesGrid();
            updateKPIs();
            
            // Check overall status
            const onlineCount = nodesStatusList.filter(n => n.online).length;
            if (onlineCount === 0 && registeredNodes.length > 0) {
                systemStatusInd.innerText = "All Offline";
                systemStatusInd.className = "status-badge status-offline";
            } else if (onlineCount < registeredNodes.length) {
                systemStatusInd.innerText = "Degraded";
                systemStatusInd.className = "status-badge status-busy";
            } else {
                systemStatusInd.innerText = "Healthy";
                systemStatusInd.className = "status-badge status-online";
            }
        } catch (e) {
            addLog(`Error checking nodes status: ${e.message}`, "error");
            systemStatusInd.innerText = "Offline";
            systemStatusInd.className = "status-badge status-offline";
        } finally {
            isPolling = false;
            btnRefreshNodes.disabled = false;
            btnRefreshNodes.innerText = "Refresh Status";
            updateControlStates();
        }
    }

    function startStatusPolling() {
        if (pollInterval) clearInterval(pollInterval);
        pollInterval = setInterval(() => {
            refreshNodesStatus();
            fetchTrainingStatus();
        }, 5000);
    }

    // -----------------------------------------------------------------------
    // UI GRID UPDATING
    // -----------------------------------------------------------------------
    function updateKPIs() {
        kpiTotalVal.innerText = registeredNodes.length;
        kpiOnlineVal.innerText = nodesStatusList.filter(n => n.online).length;
        kpiPlayingVal.innerText = nodesStatusList.filter(n => n.online && n.playing).length;
    }

    function updateNodesGrid() {
        _saveAudioState();
        _clearAllVisualizers();
        nodesGrid.innerHTML = "";

        if (nodesStatusList.length === 0) {
            nodesGrid.innerHTML = `<div class="empty-state"><svg class="empty-icon" viewBox="0 0 24 24"><path d="M4 6h16v2H4zm0 5h16v2H4zm0 5h16v2H4z" opacity=".3"/><circle cx="19" cy="5" r="3" fill="currentColor"/></svg><span>No client nodes registered.<br>Use the form above to add one.</span></div>`;
            return;
        }

        nodesStatusList.forEach(node => {
            const card = document.createElement("div");
            const isOnline = node.online;
            const isBusy = node.busy;
            const isPlaying = node.playing;
            const hasBuffer = node.buffered;
            const safeId = btoa(node.url).replace(/=/g, "");

            let statusClass = "offline";
            let statusText = "Offline";
            if (isOnline) {
                if (isBusy)         { statusClass = "busy";    statusText = "Training"; }
                else if (isPlaying) { statusClass = "playing"; statusText = `Playing — ${node.buffered_stem || ""}`; }
                else if (hasBuffer) { statusClass = "online";  statusText = `Ready — ${node.buffered_stem || ""}`; }
                else                { statusClass = "online";  statusText = "Online — Idle"; }
            }

            card.className = `node-card ${statusClass}`;
            
            // Generate visual components
            const cardHeader = `
                <div class="node-card-header">
                    <div class="node-title-info">
                        <span class="node-url-label" title="${node.url}">${node.url.replace("http://", "")}</span>
                        ${isOnline ? `<span class="node-stem-badge">${node.target_stem || 'Unassigned Stem'}</span>` : ''}
                    </div>
                    <div class="node-status-badge">
                        <span class="status-dot"></span>
                        <span>${statusText}</span>
                    </div>
                </div>
            `;
            
            const cardMeta = isOnline ? `
                <div class="node-meta-grid">
                    <div class="meta-item">
                        <span class="meta-label">Model:</span>
                        <span class="meta-val">${node.model_name || 'N/A'}</span>
                    </div>
                    <div class="meta-item">
                        <span class="meta-label">Sample Rate:</span>
                        <span class="meta-val">${node.sample_rate ? node.sample_rate + ' Hz' : 'N/A'}</span>
                    </div>
                </div>
            ` : `<div class="empty-state font-mono" style="font-size:10px; padding:10px;">Unreachable/Connection Refused</div>`;

            const cardStates = isOnline ? `
                <div class="node-states-row">
                    <div class="state-badge">
                        <span class="state-badge-label">Buffered:</span>
                        <span class="state-badge-val ${hasBuffer ? 'active' : 'inactive'}">
                            ${hasBuffer ? `YES (${node.buffered_stem})` : 'NO'}
                        </span>
                    </div>
                    <div class="state-badge">
                        <span class="state-badge-label">Audio Output:</span>
                        ${isPlaying ? `
                            <div class="playing-equalizer">
                                <span class="eq-bar"></span>
                                <span class="eq-bar"></span>
                                <span class="eq-bar"></span>
                                <span class="eq-bar"></span>
                                <span class="state-badge-val active" style="margin-left: 6px;">PLAYING</span>
                            </div>
                        ` : `
                            <span class="state-badge-val inactive">IDLE</span>
                        `}
                    </div>
                </div>
            ` : '';

            // Card controls
            const cardActions = isOnline ? `
                <div class="node-card-actions">
                    <button class="btn btn-secondary btn-sm btn-node-play" ${isPlaying ? 'disabled' : ''} data-url="${node.url}">Play</button>
                    <button class="btn btn-danger btn-sm btn-node-stop" ${!isPlaying ? 'disabled' : ''} data-url="${node.url}">Stop</button>
                    <button class="btn btn-secondary btn-sm btn-node-classify" ${!selectedTrack ? 'disabled' : ''} data-url="${node.url}">Classify</button>
                </div>
            ` : '';

            const deleteBtn = `
                <button class="node-card-delete btn-node-delete" data-url="${node.url}" title="Remove this node">
                    <svg style="width:14px;height:14px;" viewBox="0 0 24 24">
                        <path fill="currentColor" d="M19,4H15.5L14.5,3H9.5L8.5,4H5V6H19M6,19A2,2 0 0,0 8,21H16A2,2 0 0,0 18,19V7H6V19Z"/>
                    </svg>
                </button>
            `;

            const encUrl = encodeURIComponent(node.url);
            const cardAudio = isOnline && hasBuffer ? `
                <div class="node-audio-player">
                    <audio class="node-audio-elem" id="audio-elem-${safeId}"
                           data-url="${node.url}" data-sid="${safeId}" preload="none">
                        <source src="/api/audio/stream?node_url=${encUrl}" type="audio/wav">
                    </audio>
                    <canvas class="waveform-canvas hidden" id="waveform-${safeId}" width="220" height="32"></canvas>
                    <div class="audio-controls-bar">
                        <button class="btn btn-sm btn-browser-play" id="btn-play-${safeId}" data-sid="${safeId}">
                            <svg class="icon" viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>
                            Listen
                        </button>
                        <span class="audio-stem-chip">${node.buffered_stem || "stem"}</span>
                        <div class="audio-volume-group">
                            <button class="btn-mute" id="btn-mute-${safeId}" data-sid="${safeId}">🔊</button>
                            <input type="range" class="volume-slider" id="vol-${safeId}"
                                   data-sid="${safeId}" min="0" max="100" value="80">
                        </div>
                    </div>
                </div>
            ` : '';

            card.innerHTML = `
                ${deleteBtn}
                ${cardHeader}
                ${cardMeta}
                ${cardStates}
                ${cardAudio}
                ${cardActions}
                <div class="node-loader-overlay hidden" id="loader-${btoa(node.url).replace(/=/g, "")}">
                    <div class="buf-steps">
                        <div class="buf-step" id="step-upload-${btoa(node.url).replace(/=/g, "")}">
                            <span class="buf-step-dot"></span>
                            <span class="buf-step-label">Uploading</span>
                        </div>
                        <div class="buf-step-line"></div>
                        <div class="buf-step" id="step-sep-${btoa(node.url).replace(/=/g, "")}">
                            <span class="buf-step-dot"></span>
                            <span class="buf-step-label">Separating</span>
                        </div>
                        <div class="buf-step-line"></div>
                        <div class="buf-step" id="step-ready-${btoa(node.url).replace(/=/g, "")}">
                            <span class="buf-step-dot"></span>
                            <span class="buf-step-label">Buffered</span>
                        </div>
                    </div>
                    <span class="loader-text" id="loader-text-${btoa(node.url).replace(/=/g, "")}">Uploading…</span>
                </div>
            `;
            
            nodesGrid.appendChild(card);
        });

        // Re-attach card button listeners
        document.querySelectorAll(".btn-node-play").forEach(btn => {
            btn.addEventListener("click", () => triggerSingleNodePlayback(btn.dataset.url));
        });
        document.querySelectorAll(".btn-node-stop").forEach(btn => {
            btn.addEventListener("click", () => stopSingleNodePlayback(btn.dataset.url));
        });
        document.querySelectorAll(".btn-node-classify").forEach(btn => {
            btn.addEventListener("click", () => triggerSingleNodeClassification(btn.dataset.url));
        });
        document.querySelectorAll(".node-card-delete").forEach(btn => {
            btn.addEventListener("click", () => removeNode(btn.dataset.url));
        });

        // Browser audio player listeners
        document.querySelectorAll(".btn-browser-play").forEach(btn => {
            btn.addEventListener("click", () => toggleBrowserPlayback(btn.dataset.sid));
        });
        document.querySelectorAll(".btn-mute").forEach(btn => {
            btn.addEventListener("click", () => toggleMute(btn.dataset.sid));
        });
        document.querySelectorAll(".volume-slider").forEach(slider => {
            slider.addEventListener("input", () => setVolume(slider.dataset.sid, slider.value));
        });

        _restoreAudioState();
    }

    // Show/Hide loader overlay on node cards during async events (separate/buffer)
    // step: 0=uploading, 1=separating, 2=ready  (omit to keep current step)
    function toggleNodeCardLoader(url, show, text = "Processing Audio...", step = 0) {
        const sid = btoa(url).replace(/=/g, "");
        const loader = document.getElementById(`loader-${sid}`);
        if (!loader) return;
        const textSpan = document.getElementById(`loader-text-${sid}`);
        if (textSpan) textSpan.innerText = text;
        if (show) {
            loader.classList.remove("hidden");
            const stepIds = [`step-upload-${sid}`, `step-sep-${sid}`, `step-ready-${sid}`];
            stepIds.forEach((id, i) => {
                const el = document.getElementById(id);
                if (!el) return;
                el.classList.remove("active", "done");
                if (i < step)  el.classList.add("done");
                if (i === step) el.classList.add("active");
            });
        } else {
            loader.classList.add("hidden");
        }
    }

    // -----------------------------------------------------------------------
    // ADD & REMOVE NODES
    // -----------------------------------------------------------------------
    async function handleAddNode(e) {
        e.preventDefault();
        const url = nodeUrlInput.value.trim().replace(/\/$/, "");
        if (!url) return;

        try {
            const resp = await fetch("/api/nodes", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ url })
            });

            const data = await resp.json();
            if (resp.status === 200) {
                addLog(`Node added: ${url}`, "success");
                nodeUrlInput.value = "";
                registeredNodes = data.nodes;
                updateKPIs();
                refreshNodesStatus();
            } else {
                throw new Error(data.detail || "Error adding node");
            }
        } catch (err) {
            addLog(`Failed to add node: ${err.message}`, "error");
        }
    }

    async function removeNode(url) {
        if (!confirm(`Are you sure you want to remove node ${url}?`)) return;

        try {
            const resp = await fetch("/api/nodes", {
                method: "DELETE",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ url })
            });

            const data = await resp.json();
            if (resp.status === 200) {
                addLog(`Removed node: ${url}`, "info");
                registeredNodes = data.nodes;
                updateKPIs();
                refreshNodesStatus();
            } else {
                throw new Error(data.detail || "Error deleting node");
            }
        } catch (err) {
            addLog(`Failed to delete node: ${err.message}`, "error");
        }
    }

    // -----------------------------------------------------------------------
    // AUDIO LIBRARY LOADING & UPLOADING
    // -----------------------------------------------------------------------
    async function fetchAudioLibrary() {
        try {
            const resp = await fetch("/api/audio");
            if (!resp.ok) throw new Error("Failed to query library");
            const files = await resp.json();
            renderLibraryList(files);
        } catch (e) {
            addLog(`Error loading audio library: ${e.message}`, "error");
        }
    }

    function renderLibraryList(files) {
        audioLibraryList.innerHTML = "";
        
        if (files.length === 0) {
            audioLibraryList.innerHTML = `<div class="empty-state"><svg class="empty-icon" viewBox="0 0 24 24"><path d="M12 3v10.55A4 4 0 1 0 14 17V7h4V3h-6z" opacity=".35"/><path d="M12 3v10.55A4 4 0 1 0 14 17V7h4V3h-6z" fill="none" stroke="currentColor" stroke-width="1"/></svg><span>No audio tracks uploaded yet.<br>Drop a file above.</span></div>`;
            return;
        }

        files.forEach(file => {
            const item = document.createElement("div");
            item.className = `audio-item ${selectedTrack === file.filename ? 'selected' : ''}`;
            
            const meta = `
                <div class="audio-meta">
                    <span class="audio-name" title="${file.filename}">${file.filename}</span>
                    <span class="audio-size">${file.size_mb} MB</span>
                </div>
            `;
            
            const actions = `
                <div class="audio-actions">
                    <button class="btn-select-track" data-name="${file.filename}">
                        ${selectedTrack === file.filename ? 'Selected' : 'Select'}
                    </button>
                    <button class="btn-icon btn-delete-track" data-name="${file.filename}" title="Delete file">
                        <svg style="width:14px;height:14px;" viewBox="0 0 24 24">
                            <path fill="currentColor" d="M19,4H15.5L14.5,3H9.5L8.5,4H5V6H19M6,19A2,2 0 0,0 8,21H16A2,2 0 0,0 18,19V7H6V19Z"/>
                        </svg>
                    </button>
                </div>
            `;

            item.innerHTML = meta + actions;
            audioLibraryList.appendChild(item);
        });

        // Add selection listener
        document.querySelectorAll(".btn-select-track").forEach(btn => {
            btn.addEventListener("click", () => {
                selectTrack(btn.dataset.name);
            });
        });

        // Add delete listener
        document.querySelectorAll(".btn-delete-track").forEach(btn => {
            btn.addEventListener("click", () => {
                deleteTrack(btn.dataset.name);
            });
        });
    }

    function selectTrack(filename) {
        if (selectedTrack === filename) {
            selectedTrack = null;
            addLog("Track deselected.");
        } else {
            selectedTrack = filename;
            addLog(`Selected track: "${filename}"`);
        }
        
        fetchAudioLibrary(); // re-render layout selected classes
        updateControlStates(); // toggle buffer button enabling
    }

    async function deleteTrack(filename) {
        if (!confirm(`Delete "${filename}" from the library?`)) return;

        try {
            const resp = await fetch(`/api/audio/${filename}`, { method: "DELETE" });
            if (!resp.ok) throw new Error("Delete request failed");
            
            addLog(`Deleted track "${filename}" from library.`);
            if (selectedTrack === filename) {
                selectedTrack = null;
            }
            fetchAudioLibrary();
            updateControlStates();
        } catch (e) {
            addLog(`Error deleting track: ${e.message}`, "error");
        }
    }

    // Setup drag and drop / click upload logic
    function setupUploadHandlers() {
        // Drop zone drag events
        ["dragenter", "dragover"].forEach(eventName => {
            dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropZone.classList.add("dragover");
            }, false);
        });

        ["dragleave", "drop"].forEach(eventName => {
            dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropZone.classList.remove("dragover");
            }, false);
        });

        dropZone.addEventListener("drop", (e) => {
            const dt = e.dataTransfer;
            const files = dt.files;
            if (files.length > 0) {
                uploadFile(files[0]);
            }
        });

        audioInput.addEventListener("change", () => {
            if (audioInput.files.length > 0) {
                uploadFile(audioInput.files[0]);
            }
        });
    }

    // Upload using XMLHttpRequest to get accurate progress updates
    function uploadFile(file) {
        // Validate format
        const ext = file.name.split('.').pop().toLowerCase();
        if (!["wav", "mp3", "flac", "ogg"].includes(ext)) {
            addLog(`Unsupported file format: .${ext}. Must be WAV, MP3, FLAC or OGG.`, "error");
            return;
        }

        addLog(`Starting upload: ${file.name} (${(file.size / (1024 * 1024)).toFixed(2)} MB)`);
        
        // Setup UI
        uploadProgressContainer.classList.remove("hidden");
        uploadProgressBar.style.width = "0%";
        uploadProgressText.innerText = "Uploading: 0%";
        
        const fd = new FormData();
        fd.append("file", file);

        const xhr = new XMLHttpRequest();
        xhr.open("POST", "/api/audio/upload", true);

        // Upload progress
        xhr.upload.addEventListener("progress", (e) => {
            if (e.lengthComputable) {
                const percent = Math.round((e.loaded / e.total) * 100);
                uploadProgressBar.style.width = percent + "%";
                uploadProgressText.innerText = `Uploading: ${percent}%`;
            }
        });

        xhr.onload = function() {
            uploadProgressContainer.classList.add("hidden");
            try {
                if (xhr.status === 200) {
                    const response = JSON.parse(xhr.responseText);
                    addLog(`Upload completed successfully: "${response.filename}"`, "success");
                    fetchAudioLibrary();
                } else {
                    let detail = "Server error";
                    try {
                        const response = JSON.parse(xhr.responseText);
                        detail = response.detail || detail;
                    } catch (e) {}
                    addLog(`Upload failed: ${detail} (HTTP ${xhr.status})`, "error");
                }
            } catch (err) {
                addLog(`Upload processed with error: ${err.message}`, "error");
            }
        };

        xhr.onerror = function() {
            uploadProgressContainer.classList.add("hidden");
            addLog("Upload failed: Connection lost or refused.", "error");
        };

        xhr.send(fd);
    }

    // -----------------------------------------------------------------------
    // BUFFERING (SEPARATION) ORCHESTRATION
    // -----------------------------------------------------------------------
    async function triggerMasterBuffer() {
        if (!selectedTrack) return;

        const onlineNodes = nodesStatusList.filter(n => n.online);
        if (onlineNodes.length === 0) {
            addLog("Buffer failed: No online client nodes detected.", "warning");
            return;
        }

        const onlineUrls = onlineNodes.map(n => n.url);
        addLog(`Distributing track "${selectedTrack}" to ${onlineUrls.length} online nodes. Initiating separation...`, "info");
        
        // Update alerts and button loaders
        btnMasterBuffer.disabled = true;
        btnMasterBuffer.innerText = "Buffering...";
        masterStatusAlert.classList.remove("hidden");
        masterStatusText.innerText = "Distributing and separating audio mix in parallel. Please stand by...";

        // Show step 0 (Uploading) immediately on all nodes
        onlineUrls.forEach(url => {
            toggleNodeCardLoader(url, true, "Uploading track…", 0);
        });

        try {
            const resp = await fetch("/api/audio/buffer", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    filename: selectedTrack,
                    nodes: onlineUrls
                })
            });

            if (!resp.ok) {
                throw new Error(`Server error ${resp.status}`);
            }

            // Stream NDJSON lines as the backend yields per-node results
            const reader = resp.body.getReader();
            const decoder = new TextDecoder();
            let buf = "";

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                buf += decoder.decode(value, { stream: true });
                const lines = buf.split("\n");
                buf = lines.pop(); // hold incomplete trailing line
                for (const line of lines) {
                    if (!line.trim()) continue;
                    const event = JSON.parse(line);
                    if (event.done) {
                        addLog("Buffer transaction complete.");
                    } else if (event.step === "separating") {
                        // Backend confirmed upload received — advance to step 1
                        toggleNodeCardLoader(event.url, true, "Separating stem…", 1);
                    } else if (event.success) {
                        toggleNodeCardLoader(event.url, true, "Buffered ✓", 2);
                        addLog(`  ✓ ${event.url} — stem "${event.data.stem}" buffered and ready.`, "success");
                    } else {
                        addLog(`  ✗ ${event.url} — failed: ${event.error}`, "error");
                    }
                }
            }

            await new Promise(r => setTimeout(r, 700));
            await refreshNodesStatus();

        } catch (e) {
            addLog(`Failed to run parallel buffering: ${e.message}`, "error");
        } finally {
            // Clean up states
            onlineUrls.forEach(url => {
                toggleNodeCardLoader(url, false);
            });
            btnMasterBuffer.disabled = false;
            btnMasterBuffer.innerText = "Buffer Selected Track";
            masterStatusText.innerText = "Buffering transaction finalized.";
            updateControlStates();
        }
    }

    // -----------------------------------------------------------------------
    // PLAYBACK ORCHESTRATION
    // -----------------------------------------------------------------------
    async function triggerMasterPlay() {
        const bufferedNodes = nodesStatusList.filter(n => n.online && n.buffered);
        if (bufferedNodes.length === 0) {
            addLog("Playback aborted: No nodes have audio buffered in memory.", "warning");
            return;
        }

        const urls = bufferedNodes.map(n => n.url);
        addLog(`Triggering synchronized playback on ${urls.length} buffered nodes...`, "info");
        
        btnMasterPlay.disabled = true;

        try {
            const resp = await fetch("/api/audio/playback/start", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ nodes: urls })
            });

            const result = await resp.json();
            
            for (const url in result.results) {
                const res = result.results[url];
                if (res.success) {
                    addLog(`  ✓ ${url} — playing stem "${res.data.stem}"`, "success");
                    // Start in-browser playback on the node card audio element
                    const sid = btoa(url).replace(/=/g, "");
                    const audio = document.getElementById(`audio-elem-${sid}`);
                    if (audio && audio.paused) {
                        audio.src = `/api/audio/stream?node_url=${encodeURIComponent(url)}&t=${Date.now()}`;
                        audio.play().then(() => {
                            const playBtn = document.getElementById(`btn-play-${sid}`);
                            if (playBtn) {
                                playBtn.innerHTML = `<svg class="icon" viewBox="0 0 24 24"><path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/></svg> Pause`;
                                playBtn.classList.add("active");
                            }
                            _startVisualizer(sid, audio);
                        }).catch(() => {});
                    }
                } else {
                    addLog(`  ✗ ${url} — failed: ${res.error}`, "error");
                }
            }

            // Query state to show visual equalizers
            setTimeout(refreshNodesStatus, 800);

        } catch (e) {
            addLog(`Failed to run synchronized playback: ${e.message}`, "error");
        } finally {
            btnMasterPlay.disabled = false;
            updateControlStates();
        }
    }

    async function triggerMasterStop() {
        const playingNodes = nodesStatusList.filter(n => n.online && n.playing);
        // Fallback to all online nodes if none are flagged as playing (in case polling is out of sync)
        const targetNodes = playingNodes.length > 0 ? playingNodes : nodesStatusList.filter(n => n.online);
        
        if (targetNodes.length === 0) {
            addLog("Stop aborted: No online client nodes available.", "warning");
            return;
        }

        const urls = targetNodes.map(n => n.url);
        addLog(`Sending stop playback command to ${urls.length} nodes...`, "info");
        
        btnMasterStop.disabled = true;

        try {
            const resp = await fetch("/api/audio/playback/stop", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ nodes: urls })
            });

            await resp.json();
            addLog("Stop playback command completed.");

            // Stop in-browser playback on all targeted node cards
            urls.forEach(url => {
                const sid = btoa(url).replace(/=/g, "");
                const audio = document.getElementById(`audio-elem-${sid}`);
                if (audio && !audio.paused) {
                    audio.pause();
                    audio.currentTime = 0;
                    const playBtn = document.getElementById(`btn-play-${sid}`);
                    if (playBtn) {
                        playBtn.innerHTML = `<svg class="icon" viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg> Listen`;
                        playBtn.classList.remove("active");
                    }
                    _stopVisualizer(sid);
                }
            });

            // Force state refresh
            setTimeout(refreshNodesStatus, 800);

        } catch (e) {
            addLog(`Failed to send stop command: ${e.message}`, "error");
        } finally {
            btnMasterStop.disabled = false;
            updateControlStates();
        }
    }

    // Single card play/stop triggers
    async function triggerSingleNodePlayback(url) {
        addLog(`Starting playback on node ${url}...`);
        
        try {
            const resp = await fetch("/api/audio/playback/start", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ nodes: [url] })
            });
            const data = await resp.json();
            const res = data.results[url];
            if (res.success) {
                addLog(`Node ${url} started playing stem: ${res.data.stem}`, "success");
            } else {
                addLog(`Node ${url} playback failure: ${res.error}`, "error");
            }
            refreshNodesStatus();
        } catch (e) {
            addLog(`Failed: ${e.message}`, "error");
        }
    }

    async function stopSingleNodePlayback(url) {
        addLog(`Stopping playback on node ${url}...`);
        
        try {
            const resp = await fetch("/api/audio/playback/stop", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ nodes: [url] })
            });
            const data = await resp.json();
            const res = data.results[url];
            if (res.success) {
                addLog(`Node ${url} playback stopped.`, "info");
            } else {
                addLog(`Node ${url} stop failure: ${res.error}`, "error");
            }
            refreshNodesStatus();
        } catch (e) {
            addLog(`Failed: ${e.message}`, "error");
        }
    }

    // -----------------------------------------------------------------------
    // BROWSER AUDIO PLAYER
    // -----------------------------------------------------------------------
    function _saveAudioState() {
        document.querySelectorAll(".node-audio-elem").forEach(audio => {
            _audioState[audio.dataset.url] = {
                playing: !audio.paused,
                currentTime: audio.currentTime,
                volume: audio.volume,
                muted: audio.muted,
            };
        });
    }

    function _restoreAudioState() {
        document.querySelectorAll(".node-audio-elem").forEach(audio => {
            const s = _audioState[audio.dataset.url];
            if (!s) return;
            const sid = audio.dataset.sid;
            audio.volume = s.volume;
            audio.muted = s.muted;
            const volSlider = document.getElementById(`vol-${sid}`);
            if (volSlider) volSlider.value = Math.round(s.volume * 100);
            const muteBtn = document.getElementById(`btn-mute-${sid}`);
            if (muteBtn) muteBtn.innerText = s.muted ? "🔇" : "🔊";
            if (s.playing) {
                const pos = s.currentTime;
                audio.src = `/api/audio/stream?node_url=${encodeURIComponent(audio.dataset.url)}&t=${Date.now()}`;
                audio.play().then(() => {
                    audio.currentTime = pos;
                    const playBtn = document.getElementById(`btn-play-${sid}`);
                    if (playBtn) {
                        playBtn.innerHTML = `<svg class="icon" viewBox="0 0 24 24"><path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/></svg> Pause`;
                        playBtn.classList.add("active");
                    }
                    _startVisualizer(sid, audio);
                }).catch(() => {});
            }
        });
    }

    // Per-node Web Audio visualizer contexts keyed by sid
    const _vizContexts = {};

    function _startVisualizer(sid, audio) {
        const canvas = document.getElementById(`waveform-${sid}`);
        if (!canvas) return;
        if (_vizContexts[sid]) return; // already running

        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        const src = ctx.createMediaElementSource(audio);
        const analyser = ctx.createAnalyser();
        analyser.fftSize = 64;
        src.connect(analyser);
        analyser.connect(ctx.destination);

        const bufLen = analyser.frequencyBinCount;
        const data   = new Uint8Array(bufLen);
        const c      = canvas.getContext("2d");
        let   raf;

        function draw() {
            raf = requestAnimationFrame(draw);
            analyser.getByteFrequencyData(data);
            c.clearRect(0, 0, canvas.width, canvas.height);
            const barW = (canvas.width / bufLen) - 1;
            data.forEach((v, i) => {
                const h = (v / 255) * canvas.height;
                const hue = 200 + (i / bufLen) * 60; // indigo→amber sweep
                c.fillStyle = `hsla(${hue}, 70%, 65%, 0.85)`;
                c.fillRect(i * (barW + 1), canvas.height - h, barW, h);
            });
        }

        audio.addEventListener("ended", () => _stopVisualizer(sid), { once: true });
        _vizContexts[sid] = { ctx, raf: null, draw, cancel: () => cancelAnimationFrame(raf) };
        canvas.classList.remove("hidden");
        draw();
    }

    function _stopVisualizer(sid) {
        const v = _vizContexts[sid];
        if (!v) return;
        v.cancel();
        delete _vizContexts[sid];
        const canvas = document.getElementById(`waveform-${sid}`);
        if (canvas) {
            const c = canvas.getContext("2d");
            c.clearRect(0, 0, canvas.width, canvas.height);
            canvas.classList.add("hidden");
        }
    }

    function _clearAllVisualizers() {
        Object.keys(_vizContexts).forEach(sid => _stopVisualizer(sid));
    }

    function toggleBrowserPlayback(sid) {
        const audio = document.getElementById(`audio-elem-${sid}`);
        const btn   = document.getElementById(`btn-play-${sid}`);
        if (!audio || !btn) return;
        if (audio.paused) {
            audio.src = `/api/audio/stream?node_url=${encodeURIComponent(audio.dataset.url)}&t=${Date.now()}`;
            audio.play()
                .then(() => {
                    btn.innerHTML = `<svg class="icon" viewBox="0 0 24 24"><path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/></svg> Pause`;
                    btn.classList.add("active");
                    _startVisualizer(sid, audio);
                    addLog(`Browser playback started for ${audio.dataset.url.replace("http://", "")}`, "success");
                })
                .catch(e => addLog(`Browser playback error: ${e.message}`, "error"));
        } else {
            audio.pause();
            btn.innerHTML = `<svg class="icon" viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg> Listen`;
            btn.classList.remove("active");
            _stopVisualizer(sid);
        }
    }

    function toggleMute(sid) {
        const audio = document.getElementById(`audio-elem-${sid}`);
        const btn   = document.getElementById(`btn-mute-${sid}`);
        if (!audio || !btn) return;
        audio.muted = !audio.muted;
        btn.innerText = audio.muted ? "🔇" : "🔊";
    }

    function setVolume(sid, value) {
        const audio = document.getElementById(`audio-elem-${sid}`);
        if (!audio) return;
        audio.volume = value / 100;
        const muteBtn = document.getElementById(`btn-mute-${sid}`);
        if (muteBtn) muteBtn.innerText = (Number(value) === 0 || audio.muted) ? "🔇" : "🔊";
    }

    // -----------------------------------------------------------------------
    // TRAINING MONITOR
    // -----------------------------------------------------------------------
    async function fetchTrainingStatus() {
        try {
            const resp = await fetch("/api/training/status");
            if (!resp.ok) return;
            const data = await resp.json();
            renderTrainingPanel(data);
        } catch (e) {
            // non-critical — silently skip
        }
    }

    function renderTrainingPanel(nodesData) {
        const container = document.getElementById("training-nodes-row");
        const globalBadge = document.getElementById("training-global-badge");

        if (!nodesData || nodesData.length === 0) {
            container.innerHTML = `<div class="empty-state"><svg class="empty-icon" viewBox="0 0 24 24"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12" fill="none" stroke="currentColor" stroke-width="1.2" stroke-linecap="round" stroke-linejoin="round" opacity=".45"/></svg><span>No training data yet.<br>Start federated training to see metrics.</span></div>`;
            return;
        }

        const anyBusy    = nodesData.some(n => n.ok && n.busy);
        const anyHistory = nodesData.some(n => n.ok && n.history && n.history.length > 0);

        if (anyBusy) {
            globalBadge.innerText = "Training";
            globalBadge.className = "status-badge status-busy";
        } else if (anyHistory) {
            globalBadge.innerText = "Completed";
            globalBadge.className = "status-badge status-online";
        } else {
            globalBadge.innerText = "Idle";
            globalBadge.className = "status-badge status-offline";
        }

        container.innerHTML = "";

        nodesData.forEach(node => {
            const card = document.createElement("div");
            card.className = "training-node-card";

            if (!node.ok) {
                card.innerHTML = `
                    <div class="training-node-header">
                        <div class="training-node-id">
                            <span class="training-node-url">${node.url.replace("http://", "")}</span>
                        </div>
                        <span class="training-round-badge unreachable">Unreachable</span>
                    </div>`;
                container.appendChild(card);
                return;
            }

            const history = node.history || [];
            const isBusy  = node.busy;
            const round   = node.current_round || 0;

            let roundBadgeClass = "training-round-badge";
            let roundText;
            if (isBusy) {
                roundBadgeClass += " busy";
                roundText = `Round ${round} — Training…`;
            } else if (round > 0) {
                roundText = `Round ${round} — Done`;
            } else {
                roundText = "No rounds yet";
            }

            const stemBadge = node.target_stem
                ? `<span class="node-stem-badge">${node.target_stem}</span>`
                : "";

            // Build loss sparkline from fit-phase history
            function buildSparkline(history) {
                const fitRecs = history.filter(r => r.phase === "fit" && r.loss !== undefined);
                if (fitRecs.length < 2) return "";
                const losses = fitRecs.map(r => r.loss);
                const W = 80, H = 24, PAD = 2;
                const minL = Math.min(...losses), maxL = Math.max(...losses);
                const rangeL = maxL - minL || 1;
                const pts = losses.map((l, i) => {
                    const x = PAD + (i / (losses.length - 1)) * (W - PAD * 2);
                    const y = PAD + (1 - (l - minL) / rangeL) * (H - PAD * 2);
                    return `${x.toFixed(1)},${y.toFixed(1)}`;
                }).join(" ");
                const lastLoss = losses[losses.length - 1].toFixed(3);
                const trend = losses[losses.length - 1] < losses[0] ? "down" : "up";
                return `<div class="sparkline-wrap" title="Loss over rounds">
                    <svg class="sparkline" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}">
                        <polyline points="${pts}" fill="none" stroke="var(--indigo)" stroke-width="1.5" stroke-linejoin="round" stroke-linecap="round"/>
                        <circle cx="${pts.split(" ").pop().split(",")[0]}" cy="${pts.split(" ").pop().split(",")[1]}" r="2" fill="var(--indigo)"/>
                    </svg>
                    <span class="sparkline-val sparkline-${trend}">${lastLoss}</span>
                </div>`;
            }

            let tableHtml;
            if (history.length === 0) {
                tableHtml = `<div class="training-empty">No rounds completed yet.</div>`;
            } else {
                const rows = history.map(rec => {
                    const isFit   = rec.phase === "fit";
                    const loss    = rec.loss    !== undefined ? rec.loss.toFixed(3)   : "—";
                    const clsAcc  = rec.cls_accuracy !== undefined ? `${(rec.cls_accuracy * 100).toFixed(0)}%` : "—";
                    const siSdr   = !isFit && rec.mean_si_sdr_improvement !== undefined
                        ? `${rec.mean_si_sdr_improvement.toFixed(1)} dB`
                        : `<span class="cell-muted">—</span>`;
                    const dur = rec.duration_s !== undefined ? `${rec.duration_s}s` : "—";
                    return `<tr class="round-row-${rec.phase}">
                        <td>${rec.round}</td>
                        <td><span class="phase-badge ${rec.phase}">${rec.phase}</span></td>
                        <td>${loss}</td>
                        <td>${siSdr}</td>
                        <td>${clsAcc}</td>
                        <td class="cell-muted">${dur}</td>
                    </tr>`;
                }).join("");

                tableHtml = `<table class="rounds-table">
                    <thead><tr>
                        <th>Rd</th><th>Phase</th><th>Loss</th>
                        <th>SI-SDR↑</th><th>CLS Acc</th><th>Time</th>
                    </tr></thead>
                    <tbody>${rows}</tbody>
                </table>`;
            }

            const sparkline = buildSparkline(history);

            card.innerHTML = `
                <div class="training-node-header">
                    <div class="training-node-id">
                        <span class="training-node-url">${node.url.replace("http://", "")}</span>
                        ${stemBadge}
                    </div>
                    <div class="training-node-header-right">
                        ${sparkline}
                        <span class="${roundBadgeClass}">${roundText}</span>
                    </div>
                </div>
                <div class="rounds-table-container scrollable">${tableHtml}</div>`;

            container.appendChild(card);
        });
    }

    // -----------------------------------------------------------------------
    // STEM CLASSIFICATION PROXYING
    // -----------------------------------------------------------------------
    async function triggerSingleNodeClassification(url) {
        if (!selectedTrack) return;

        const node = nodesStatusList.find(n => n.url === url);
        
        addLog(`Requesting stem presence classification for "${selectedTrack}" on client ${url}...`, "info");
        
        toggleNodeCardLoader(url, true, "Running Classifier Head...");
        classificationCard.classList.add("hidden");

        try {
            const resp = await fetch("/api/audio/classify", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    filename: selectedTrack,
                    node_url: url
                })
            });

            if (!resp.ok) {
                const errData = await resp.json();
                throw new Error(errData.detail || "Server error");
            }

            const data = await resp.json();
            addLog(`Classification complete for node ${url}`, "success");

            // Format results
            clsFilenameText.innerText = selectedTrack;
            clsStemTargetText.innerText = `${url.replace("http://", "")} (${data.stem})`;
            
            if (data.present) {
                clsPresentBadge.innerText = "PRESENT";
                clsPresentBadge.className = "cls-result-badge present";
                addLog(`  → Stem "${data.stem}" is PRESENT (confidence: ${(data.confidence * 100).toFixed(1)}%)`, "success");
            } else {
                clsPresentBadge.innerText = "ABSENT";
                clsPresentBadge.className = "cls-result-badge absent";
                addLog(`  → Stem "${data.stem}" is ABSENT (confidence: ${(data.confidence * 100).toFixed(1)}%)`, "warning");
            }
            clsConfidenceVal.innerText = `${(data.confidence * 100).toFixed(1)}% Confidence`;
            
            // Expand card
            classificationCard.classList.remove("hidden");

        } catch (e) {
            addLog(`Classification failed: ${e.message}`, "error");
        } finally {
            toggleNodeCardLoader(url, false);
        }
    }

    // -----------------------------------------------------------------------
    // MODEL LIBRARY
    // -----------------------------------------------------------------------
    let _modelPushTarget = null;

    async function fetchModelLibrary() {
        try {
            const resp = await fetch("/api/models");
            if (!resp.ok) throw new Error("Failed to query model library");
            renderModelLibrary(await resp.json());
        } catch (e) {
            addLog(`Error loading model library: ${e.message}`, "error");
        }
    }

    function renderModelLibrary(files) {
        const list = document.getElementById("model-library-list");
        list.innerHTML = "";
        if (files.length === 0) {
            list.innerHTML = `<div class="empty-state"><svg class="empty-icon" viewBox="0 0 24 24"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" fill="none" stroke="currentColor" stroke-width="1.2" stroke-linecap="round" stroke-linejoin="round" opacity=".5"/></svg><span>No models uploaded yet.<br>Use Upload .pt above.</span></div>`;
            return;
        }
        files.forEach(file => {
            const item = document.createElement("div");
            item.className = "audio-item model-item";
            item.innerHTML = `
                <div class="audio-meta">
                    <span class="audio-name" title="${file.filename}">${file.filename}</span>
                    <span class="audio-size">${file.size_mb} MB</span>
                </div>
                <div class="audio-actions">
                    <button class="btn-select-track btn-model-push" data-name="${file.filename}">Push</button>
                    <button class="btn-icon btn-delete-model" data-name="${file.filename}" title="Delete model">
                        <svg style="width:14px;height:14px;" viewBox="0 0 24 24">
                            <path fill="currentColor" d="M19,4H15.5L14.5,3H9.5L8.5,4H5V6H19M6,19A2,2 0 0,0 8,21H16A2,2 0 0,0 18,19V7H6V19Z"/>
                        </svg>
                    </button>
                </div>`;
            list.appendChild(item);
        });

        document.querySelectorAll(".btn-model-push").forEach(btn => {
            btn.addEventListener("click", () => showModelPushPanel(btn.dataset.name));
        });
        document.querySelectorAll(".btn-delete-model").forEach(btn => {
            btn.addEventListener("click", () => deleteModel(btn.dataset.name));
        });
    }

    function showModelPushPanel(filename) {
        _modelPushTarget = filename;
        document.getElementById("model-push-filename").innerText = filename;

        const nodeListEl = document.getElementById("model-push-node-list");
        nodeListEl.innerHTML = "";
        const online = nodesStatusList.filter(n => n.online);
        if (online.length === 0) {
            nodeListEl.innerHTML = `<div class="empty-state">No online nodes available.</div>`;
        } else {
            online.forEach(n => {
                const row = document.createElement("label");
                row.className = "model-node-select-row";
                row.innerHTML = `
                    <input type="checkbox" class="model-node-check" value="${n.url}" checked>
                    <span class="model-node-url">${n.url.replace("http://", "")}</span>
                    ${n.target_stem ? `<span class="node-stem-badge">${n.target_stem}</span>` : ""}`;
                nodeListEl.appendChild(row);
            });
        }

        document.getElementById("model-push-panel").classList.remove("hidden");
    }

    async function deleteModel(filename) {
        if (!confirm(`Delete model "${filename}"?`)) return;
        try {
            const resp = await fetch(`/api/models/${encodeURIComponent(filename)}`, { method: "DELETE" });
            if (!resp.ok) throw new Error((await resp.json()).detail || "Delete failed");
            addLog(`Deleted model "${filename}"`, "info");
            fetchModelLibrary();
        } catch (e) {
            addLog(`Failed to delete model: ${e.message}`, "error");
        }
    }

    async function pushModel() {
        const checks = document.querySelectorAll(".model-node-check:checked");
        const nodes = Array.from(checks).map(c => c.value);
        if (!_modelPushTarget || nodes.length === 0) return;

        const btn = document.getElementById("btn-model-push-confirm");
        btn.disabled = true;
        btn.innerText = "Pushing…";
        addLog(`Pushing model "${_modelPushTarget}" to ${nodes.length} node(s)…`, "info");

        try {
            const resp = await fetch("/api/models/push", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ filename: _modelPushTarget, nodes })
            });
            const result = await resp.json();
            for (const url in result.results) {
                const r = result.results[url];
                if (r.success) {
                    addLog(`  ✓ ${url} — model loaded`, "success");
                } else {
                    addLog(`  ✗ ${url} — ${r.error}`, "error");
                }
            }
            document.getElementById("model-push-panel").classList.add("hidden");
            _modelPushTarget = null;
        } catch (e) {
            addLog(`Push failed: ${e.message}`, "error");
        } finally {
            btn.disabled = false;
            btn.innerText = "Push to Selected Nodes";
        }
    }

    function setupModelLibraryHandlers() {
        const modelInput = document.getElementById("model-file-input");
        modelInput.addEventListener("change", async () => {
            const file = modelInput.files[0];
            if (!file) return;
            addLog(`Uploading model: ${file.name}…`);
            const fd = new FormData();
            fd.append("file", file);
            try {
                const resp = await fetch("/api/models/upload", { method: "POST", body: fd });
                if (!resp.ok) throw new Error((await resp.json()).detail || "Upload failed");
                addLog(`Model "${file.name}" uploaded successfully`, "success");
                fetchModelLibrary();
            } catch (e) {
                addLog(`Model upload failed: ${e.message}`, "error");
            }
            modelInput.value = "";
        });

        document.getElementById("btn-model-push-close").addEventListener("click", () => {
            document.getElementById("model-push-panel").classList.add("hidden");
            _modelPushTarget = null;
        });

        document.getElementById("btn-model-push-confirm").addEventListener("click", pushModel);
    }

    // -----------------------------------------------------------------------
    // KEYBOARD SHORTCUTS
    // -----------------------------------------------------------------------
    function setupKeyboardShortcuts() {
        document.addEventListener("keydown", e => {
            // Ignore when typing in an input/textarea
            const tag = document.activeElement.tagName;
            if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;

            if (e.code === "Space") {
                e.preventDefault();
                if (!btnMasterPlay.disabled) {
                    triggerMasterPlay();
                } else if (!btnMasterStop.disabled) {
                    triggerMasterStop();
                }
            }
            if (e.code === "KeyR" && !e.ctrlKey && !e.metaKey) {
                e.preventDefault();
                refreshNodesStatus();
            }
            if (e.code === "KeyB" && !e.ctrlKey && !e.metaKey) {
                e.preventDefault();
                if (!btnMasterBuffer.disabled) triggerMasterBuffer();
            }
        });

        // Attach kbd hint badges to buttons
        const hints = [
            ["btn-master-play",  "Space"],
            ["btn-master-stop",  "Space"],
            ["btn-master-buffer","B"],
            ["btn-refresh-nodes","R"],
        ];
        hints.forEach(([id, key]) => {
            const btn = document.getElementById(id);
            if (!btn) return;
            const kbd = document.createElement("kbd");
            kbd.className = "kbd-hint";
            kbd.textContent = key;
            btn.appendChild(kbd);
        });
    }

    // Run initialization
    init();
    setupKeyboardShortcuts();
});
