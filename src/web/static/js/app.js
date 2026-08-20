(() => {
    "use strict";

    const SLOT_COLORS = {
        regular: "#475569",
        ev_charging: "#0e7490",
        reserved: "#6d28d9",
        handicapped: "#b45309",
    };
    const SLOT_FREE_COLORS = {
        regular: "#64748b",
        ev_charging: "#22d3ee",
        reserved: "#a78bfa",
        handicapped: "#fbbf24",
    };
    const VEHICLE_COLORS = {
        regular: "#3b82f6",
        ev: "#10b981",
        reserved: "#8b5cf6",
        handicapped: "#f59e0b",
    };
    const SLOT_OCCUPIED_COLOR = "#991b1b";

    let ws = null;
    let state = null;
    let history = [];
    let logs = [];
    let running = false;
    let config = { slot_colors: SLOT_FREE_COLORS, vehicle_colors: VEHICLE_COLORS };

    const canvas = document.getElementById("sim-canvas");
    const ctx = canvas.getContext("2d");

    const chartRewardCanvas = document.getElementById("chart-reward");
    const chartRewardCtx = chartRewardCanvas.getContext("2d");
    const chartOccCanvas = document.getElementById("chart-occupancy");
    const chartOccCtx = chartOccCanvas.getContext("2d");

    function resizeCanvas() {
        const wrap = canvas.parentElement;
        const dpr = window.devicePixelRatio || 1;
        canvas.width = wrap.clientWidth * dpr;
        canvas.height = wrap.clientHeight * dpr;
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

        for (const c of [chartRewardCanvas, chartOccCanvas]) {
            const p = c.parentElement;
            c.width = p.clientWidth * dpr;
            c.height = 140 * dpr;
            c.style.width = p.clientWidth + "px";
            c.style.height = "140px";
        }
    }
    window.addEventListener("resize", () => { resizeCanvas(); draw(); });

    function connect() {
        const proto = location.protocol === "https:" ? "wss" : "ws";
        ws = new WebSocket(`${proto}://${location.host}/ws/sim`);
        ws.onmessage = (e) => {
            const msg = JSON.parse(e.data);
            if (msg.type === "state") {
                state = msg.data;
                if (state.config) config = state.config;
                history.push({ step: state.step, reward: state.stats.reward, occ: state.stats.occupancy });
                if (history.length > 500) history.shift();
                addLog(state);
                updateUI();
                draw();
            } else if (msg.type === "done") {
                running = false;
                document.getElementById("btn-run").textContent = "Run Day";
            }
        };
        ws.onclose = () => setTimeout(connect, 1000);
    }

    function send(msg) {
        if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(msg));
    }

    document.getElementById("btn-reset").onclick = () => {
        running = false;
        history = [];
        logs = [];
        document.getElementById("log-list").innerHTML = "";
        document.getElementById("btn-run").textContent = "Run Day";
        send({
            cmd: "reset",
            slots: +document.getElementById("sel-slots").value,
            agent: document.getElementById("sel-agent").value,
            steps: 500,
            seed: 42,
        });
    };

    document.getElementById("btn-step").onclick = () => {
        if (!running) send({ cmd: "step" });
    };

    document.getElementById("btn-run").onclick = () => {
        if (running) {
            running = false;
            document.getElementById("btn-run").textContent = "Run Day";
            return;
        }
        running = true;
        document.getElementById("btn-run").textContent = "Stop";
        send({ cmd: "run_day" });
    };

    document.getElementById("sel-slots").onchange = document.getElementById("sel-agent").onchange = () => {
        running = false;
        history = [];
        logs = [];
        document.getElementById("log-list").innerHTML = "";
        document.getElementById("btn-run").textContent = "Run Day";
        send({
            cmd: "init",
            slots: +document.getElementById("sel-slots").value,
            agent: document.getElementById("sel-agent").value,
            steps: 500,
            seed: 42,
        });
    };

    function addLog(s) {
        if (!s || s.step === undefined) return;
        const entry = document.getElementById("log-list");
        let cls = "";
        let text = "";
        if (s.action !== null && s.action !== undefined && s.incoming) {
            if (s.action === (s.grid ? s.grid.slots.length : 12)) {
                cls = "log-reject";
                text = `[${s.step}] REJECT ${s.incoming.id} (${s.incoming.type})`;
            } else {
                cls = "log-alloc";
                text = `[${s.step}] ALLOC ${s.incoming.id} -> Slot #${s.action}`;
            }
        } else {
            text = `[${s.step}] tick`;
        }
        const div = document.createElement("div");
        div.className = "log-entry " + cls;
        div.textContent = text;
        entry.appendChild(div);
        entry.scrollTop = entry.scrollHeight;
        if (entry.children.length > 200) entry.removeChild(entry.firstChild);
    }

    function updateUI() {
        if (!state) return;
        const s = state.stats;
        document.getElementById("st-step").textContent = `${state.step} / ${state.max_steps}`;
        document.getElementById("st-reward").textContent = s.reward.toFixed(1);
        document.getElementById("st-allocated").textContent = s.allocated;
        document.getElementById("st-rejected").textContent = s.rejected;
        document.getElementById("st-queue").textContent = s.queue_length;
        document.getElementById("st-occupancy").textContent = (s.occupancy * 100).toFixed(0) + "%";
        document.getElementById("occ-fill").style.width = (s.occupancy * 100) + "%";

        const vbox = document.getElementById("vehicle-box");
        if (state.incoming) {
            const v = state.incoming;
            vbox.innerHTML = `<div class="vehicle-info">
                <span class="vid">${v.id}</span> &mdash;
                <span class="vtype">${v.type.toUpperCase()}</span><br>
                Wait: ${v.wait_time} &nbsp;|&nbsp; EV: ${v.requires_ev ? "Yes" : "No"} &nbsp;|&nbsp; HC: ${v.requires_handicapped ? "Yes" : "No"}
            </div>`;
        } else {
            vbox.innerHTML = '<div class="no-vehicle">No vehicle waiting</div>';
        }

        const abox = document.getElementById("action-box");
        if (state.last_action !== null && state.last_action !== undefined) {
            const numSlots = state.grid ? state.grid.slots.length : 12;
            if (state.last_action === numSlots) {
                abox.innerHTML = `<span class="action-badge reject">REJECT</span>`;
            } else {
                const slot = state.slots[state.last_action];
                abox.innerHTML = `Allocate to Slot #${state.last_action}
                    <span class="action-badge alloc">DIST ${slot ? slot.distance.toFixed(1) : "?"}</span>`;
            }
        }
    }

    function draw() {
        const W = canvas.width / (window.devicePixelRatio || 1);
        const H = canvas.height / (window.devicePixelRatio || 1);
        ctx.clearRect(0, 0, W, H);
        ctx.fillStyle = "#0f172a";
        ctx.fillRect(0, 0, W, H);

        if (!state) {
            ctx.fillStyle = "#64748b";
            ctx.font = "16px Inter, sans-serif";
            ctx.textAlign = "center";
            ctx.fillText("Connecting to simulation...", W / 2, H / 2);
            drawChart(chartRewardCtx, chartRewardCanvas, [], "Reward", "#3b82f6");
            drawChart(chartOccCtx, chartOccCanvas, [], "Occupancy", "#f59e0b");
            return;
        }

        const grid = state.grid;
        const cols = grid.cols;
        const rows = grid.rows;
        const slotW = Math.min(80, (W - 80) / cols);
        const slotH = Math.min(60, (H - 120) / rows);
        const lotW = cols * slotW;
        const lotH = rows * slotH;
        const ox = (W - lotW) / 2;
        const oy = (H - lotH) / 2 + 20;

        ctx.fillStyle = "#f1f5f9";
        ctx.font = "bold 13px Inter, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText(
            `Parking Lot  |  ${state.grid.slots.length} Slots  |  Agent: ${document.getElementById("sel-agent").value}`,
            W / 2, oy - 30
        );

        const ex = ox + grid.entrance[0] * slotW + slotW / 2;
        const ey = oy + grid.entrance[1] * slotH + slotH / 2 + slotH;
        ctx.fillStyle = "#22d3ee";
        ctx.font = "bold 11px Inter, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("ENTRANCE", ex, ey - slotH * 0.3);
        ctx.strokeStyle = "#22d3ee";
        ctx.lineWidth = 2;
        ctx.globalAlpha = 0.5;
        ctx.beginPath();
        ctx.moveTo(ex - slotW * 0.4, ey - slotH * 0.1);
        ctx.lineTo(ex + slotW * 0.4, ey - slotH * 0.1);
        ctx.stroke();
        ctx.globalAlpha = 1;

        for (const slot of state.slots) {
            const [gx, gy] = slot.pos;
            const x = ox + gx * slotW;
            const y = oy + (rows - 1 - gy) * slotH;
            const r = 6;

            ctx.beginPath();
            ctx.roundRect(x + 3, y + 3, slotW - 6, slotH - 6, r);

            if (slot.occupied) {
                ctx.fillStyle = SLOT_OCCUPIED_COLOR;
                ctx.fill();
                ctx.strokeStyle = "#facc15";
                ctx.lineWidth = 2;
            } else {
                ctx.fillStyle = SLOT_FREE_COLORS[slot.type] || "#64748b";
                ctx.fill();
                ctx.strokeStyle = "#334155";
                ctx.lineWidth = 1;
            }
            ctx.stroke();

            ctx.fillStyle = slot.occupied ? "#fef2f2" : "#0f172a";
            ctx.font = "bold 12px Inter, sans-serif";
            ctx.textAlign = "center";
            ctx.fillText(`#${slot.id}`, x + slotW / 2, y + slotH / 2 - 4);

            ctx.font = "9px Inter, sans-serif";
            ctx.globalAlpha = 0.7;
            ctx.fillText(
                slot.type.replace("_", " ").substring(0, 8),
                x + slotW / 2,
                y + slotH / 2 + 10
            );
            ctx.globalAlpha = 1;
        }

        if (state.incoming) {
            const vx = ex;
            const vy = ey + slotH * 0.4;
            const vw = slotW * 0.5;
            const vh = slotH * 0.35;
            ctx.beginPath();
            ctx.roundRect(vx - vw / 2, vy - vh / 2, vw, vh, 4);
            ctx.fillStyle = VEHICLE_COLORS[state.incoming.type] || "#3b82f6";
            ctx.fill();
            ctx.strokeStyle = "white";
            ctx.lineWidth = 1.5;
            ctx.stroke();
            ctx.fillStyle = "white";
            ctx.font = "bold 8px JetBrains Mono, monospace";
            ctx.textAlign = "center";
            ctx.fillText(state.incoming.id.substring(0, 6), vx, vy + 3);
        }

        drawChart(chartRewardCtx, chartRewardCanvas,
            history.map(h => h.reward), "Cumulative Reward", "#3b82f6");
        drawChart(chartOccCtx, chartOccCanvas,
            history.map(h => h.occ * 100), "Occupancy %", "#f59e0b");
    }

    function drawChart(cctx, ccanvas, data, label, color) {
        const dpr = window.devicePixelRatio || 1;
        const w = ccanvas.width / dpr;
        const h = ccanvas.height / dpr;
        cctx.save();
        cctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        cctx.clearRect(0, 0, w, h);
        cctx.fillStyle = "#0f172a";
        cctx.fillRect(0, 0, w, h);

        cctx.fillStyle = "#94a3b8";
        cctx.font = "10px Inter, sans-serif";
        cctx.textAlign = "left";
        cctx.fillText(label, 8, 14);

        if (data.length < 2) {
            cctx.fillStyle = "#475569";
            cctx.textAlign = "center";
            cctx.fillText("Collecting data...", w / 2, h / 2 + 5);
            cctx.restore();
            return;
        }

        const pad = { l: 35, r: 10, t: 22, b: 18 };
        const cw = w - pad.l - pad.r;
        const ch = h - pad.t - pad.b;
        const maxVal = Math.max(...data, 1);
        const minVal = Math.min(...data, 0);
        const range = maxVal - minVal || 1;

        cctx.strokeStyle = "#334155";
        cctx.lineWidth = 0.5;
        for (let i = 0; i <= 4; i++) {
            const yy = pad.t + ch - (i / 4) * ch;
            cctx.beginPath();
            cctx.moveTo(pad.l, yy);
            cctx.lineTo(pad.l + cw, yy);
            cctx.stroke();
            cctx.fillStyle = "#64748b";
            cctx.font = "8px JetBrains Mono, monospace";
            cctx.textAlign = "right";
            cctx.fillText((minVal + (i / 4) * range).toFixed(0), pad.l - 4, yy + 3);
        }

        cctx.beginPath();
        for (let i = 0; i < data.length; i++) {
            const x = pad.l + (i / (data.length - 1)) * cw;
            const y = pad.t + ch - ((data[i] - minVal) / range) * ch;
            if (i === 0) cctx.moveTo(x, y);
            else cctx.lineTo(x, y);
        }
        cctx.strokeStyle = color;
        cctx.lineWidth = 1.5;
        cctx.stroke();

        const last = data[data.length - 1];
        const lx = pad.l + cw;
        const ly = pad.t + ch - ((last - minVal) / range) * ch;
        cctx.beginPath();
        cctx.arc(lx, ly, 3, 0, Math.PI * 2);
        cctx.fillStyle = color;
        cctx.fill();

        cctx.restore();
    }

    resizeCanvas();
    connect();
    send({
        cmd: "init",
        slots: +document.getElementById("sel-slots").value,
        agent: document.getElementById("sel-agent").value,
        steps: 500,
        seed: 42,
    });
})();
