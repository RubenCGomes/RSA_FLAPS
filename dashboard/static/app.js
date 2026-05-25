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

    // -----------------------------------------------------------------------
    // INITIALIZATION & EVENT LISTENERS
    // -----------------------------------------------------------------------
    function init() {
        addLog("Dashboard initialized. Orchestrator mode active.", "system");
        
        // Load initial data
        fetchAudioLibrary();
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
        nodesGrid.innerHTML = "";
        
        if (nodesStatusList.length === 0) {
            nodesGrid.innerHTML = `<div class="empty-state">No client nodes registered or responding.</div>`;
            return;
        }

        nodesStatusList.forEach(node => {
            const card = document.createElement("div");
            const isOnline = node.online;
            const isBusy = node.busy;
            const isPlaying = node.playing;
            const hasBuffer = node.buffered;
            
            let statusClass = "offline";
            let statusText = "Offline";
            if (isOnline) {
                statusClass = isBusy ? "busy" : "online";
                statusText = isBusy ? "Busy (FL)" : "Online";
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

            card.innerHTML = `
                ${deleteBtn}
                ${cardHeader}
                ${cardMeta}
                ${cardStates}
                ${cardActions}
                <!-- Loader screen -->
                <div class="node-loader-overlay hidden" id="loader-${btoa(node.url).replace(/=/g, "")}">
                    <div class="spinner"></div>
                    <span class="loader-text">Processing Audio...</span>
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
    }

    // Show/Hide loader overlay on node cards during async events (separate/buffer)
    function toggleNodeCardLoader(url, show, text = "Processing Audio...") {
        const id = `loader-${btoa(url).replace(/=/g, "")}`;
        const loader = document.getElementById(id);
        if (loader) {
            const textSpan = loader.querySelector(".loader-text");
            if (textSpan) textSpan.innerText = text;
            if (show) {
                loader.classList.remove("hidden");
            } else {
                loader.classList.add("hidden");
            }
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
            audioLibraryList.innerHTML = `<div class="empty-state">No audio tracks uploaded yet.</div>`;
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

        // Show individual loaders on card nodes
        onlineUrls.forEach(url => {
            toggleNodeCardLoader(url, true, "Separating & Buffering Stem...");
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

            const result = await resp.json();
            addLog(`Buffer transaction complete. Inspection results:`);
            
            // Loop through results
            for (const url in result.results) {
                const res = result.results[url];
                if (res.success) {
                    addLog(`  ✓ ${url} — stem "${res.data.stem}" is buffered and ready.`, "success");
                } else {
                    addLog(`  ✗ ${url} — failed: ${res.error}`, "error");
                }
            }

            // Immediately query node status updates
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

            const result = await resp.json();
            addLog("Stop playback command completed.");
            
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
            container.innerHTML = `<div class="empty-state">No client nodes registered.</div>`;
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

            card.innerHTML = `
                <div class="training-node-header">
                    <div class="training-node-id">
                        <span class="training-node-url">${node.url.replace("http://", "")}</span>
                        ${stemBadge}
                    </div>
                    <span class="${roundBadgeClass}">${roundText}</span>
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

    // Run initialization
    init();
});
