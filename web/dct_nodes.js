/**
 * ComfyUI DCT Nodes – frontend extension
 *
 * Adds an "Edit Mask" button to the DCT Spectrum Mask Editor node.
 * Clicking the button opens a full-screen modal paint editor where the
 * user can draw directly on the DCT spectrum:
 *
 *   Left-click  / drag  → paint black  (zero those DCT frequencies)
 *   Right-click / drag  → paint white  (restore those frequencies)
 *   Scroll wheel        → brush size
 *   "Clear All"         → fill white (keep all)
 *   "Invert"            → invert the whole mask
 *   "Apply Mask"        → save & close
 *   "Cancel"            → discard changes
 *
 * NOTE: File is intentionally placed at web/dct_nodes.js (not web/js/)
 * so that "../../scripts/app.js" resolves to ComfyUI's /scripts/app.js.
 */

import { app } from "../../scripts/app.js";

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/** Follow the dct_spectrum input back to its source node and return the
 *  first HTMLImageElement stored in that node's imgs[] array.
 *  (DCT2D has one IMAGE output – spectrum_preview – so it lands at imgs[0].) */
function getSpectrumImg(node) {
    const input = node.inputs?.find(i => i.name === "dct_spectrum");
    if (!input?.link) return null;
    const link = app.graph.links[input.link];
    if (!link) return null;
    const src = app.graph.getNodeById(link.origin_id);
    if (!src?.imgs?.length) return null;
    const el = Array.isArray(src.imgs[0]) ? src.imgs[0][0] : src.imgs[0];
    return (el instanceof HTMLImageElement && el.complete && el.naturalWidth > 0) ? el : null;
}

function makeButton(text, bg, onClick) {
    const btn = document.createElement("button");
    btn.textContent = text;
    Object.assign(btn.style, {
        background: bg, color: "#eee", border: "none",
        padding: "6px 16px", borderRadius: "4px",
        cursor: "pointer", fontSize: "12px", fontFamily: "sans-serif",
    });
    btn.addEventListener("mousedown", e => e.stopPropagation());
    btn.addEventListener("click", onClick);
    return btn;
}

// ─────────────────────────────────────────────────────────────────────────────
// Modal mask editor
// ─────────────────────────────────────────────────────────────────────────────
function openMaskEditor(node) {
    const specImg   = getSpectrumImg(node);
    const maskWidget = node.widgets?.find(w => w.name === "mask_data");

    // ── Decide canvas dimensions ──────────────────────────────────────────
    const maxW = Math.floor(window.innerWidth  * 0.78);
    const maxH = Math.floor(window.innerHeight * 0.72);
    let cW, cH;
    if (specImg) {
        const aspect = specImg.naturalWidth / specImg.naturalHeight;
        if (aspect >= 1) { cW = Math.min(maxW, specImg.naturalWidth);  cH = Math.round(cW / aspect); }
        else             { cH = Math.min(maxH, specImg.naturalHeight); cW = Math.round(cH * aspect); }
    } else {
        cW = Math.min(maxW, 512);
        cH = Math.min(maxH, 512);
    }

    // ── Off-screen mask canvas (paint target) ─────────────────────────────
    const maskC  = document.createElement("canvas");
    maskC.width  = cW;
    maskC.height = cH;
    const mctx   = maskC.getContext("2d");
    mctx.fillStyle = "white";
    mctx.fillRect(0, 0, cW, cH);

    // Restore previously saved mask
    const saved = maskWidget?.value ?? "";
    if (saved && saved.length > 50) {
        const img = new Image();
        img.onload = () => { mctx.drawImage(img, 0, 0, cW, cH); render(); };
        img.src = saved;
    }

    // ── Display canvas (composite: spectrum bg + mask overlay) ────────────
    const dispC  = document.createElement("canvas");
    dispC.width  = cW;
    dispC.height = cH;
    dispC.style.cssText = "display:block; border:1px solid #555; cursor:crosshair;";
    const dctx   = dispC.getContext("2d");

    function render() {
        dctx.clearRect(0, 0, cW, cH);
        if (specImg) {
            dctx.globalAlpha = 1.0;
            dctx.drawImage(specImg, 0, 0, cW, cH);
        } else {
            dctx.fillStyle = "#1a1a1a";
            dctx.fillRect(0, 0, cW, cH);
            dctx.fillStyle = "#555";
            dctx.font = "14px sans-serif";
            dctx.textAlign = "center";
            dctx.fillText("Run the DCT 2D Transform node first to load the spectrum background.",
                cW / 2, cH / 2);
            dctx.textAlign = "left";
        }
        // Mask overlay
        dctx.globalAlpha = 0.55;
        dctx.drawImage(maskC, 0, 0);
        dctx.globalAlpha = 1.0;
    }
    render();

    // ── Paint logic ───────────────────────────────────────────────────────
    let painting  = false;
    let erasing   = false;
    let brushSize = 20;
    let lastPos   = null;

    function paint(x, y, erase) {
        mctx.beginPath();
        mctx.arc(x, y, brushSize, 0, Math.PI * 2);
        mctx.fillStyle = erase ? "white" : "black";
        mctx.fill();
        if (lastPos) {
            mctx.beginPath();
            mctx.moveTo(lastPos[0], lastPos[1]);
            mctx.lineTo(x, y);
            mctx.strokeStyle  = erase ? "white" : "black";
            mctx.lineWidth    = brushSize * 2;
            mctx.lineCap      = "round";
            mctx.stroke();
        }
        lastPos = [x, y];
        render();
    }

    function evPos(e) {
        const r = dispC.getBoundingClientRect();
        return [e.clientX - r.left, e.clientY - r.top];
    }

    dispC.addEventListener("mousedown",  e => {
        e.preventDefault();
        painting = true;
        erasing  = e.button === 2;
        lastPos  = null;
        paint(...evPos(e), erasing);
    });
    dispC.addEventListener("mousemove",  e => { if (painting) paint(...evPos(e), erasing); });
    dispC.addEventListener("contextmenu",e => e.preventDefault());

    const globalUp = () => { painting = false; lastPos = null; };
    window.addEventListener("mouseup", globalUp);

    dispC.addEventListener("wheel", e => {
        e.preventDefault();
        brushSize = Math.max(2, Math.min(120, brushSize + (e.deltaY > 0 ? -2 : 2)));
        brushLbl.textContent = `Brush: ${brushSize} px`;
    }, { passive: false });

    // ── DOM layout ────────────────────────────────────────────────────────
    const overlay = document.createElement("div");
    Object.assign(overlay.style, {
        position: "fixed", top: "0", left: "0", width: "100%", height: "100%",
        background: "rgba(0,0,0,0.82)", zIndex: "10000",
        display: "flex", flexDirection: "column",
        alignItems: "center", justifyContent: "center",
        fontFamily: "sans-serif",
    });

    const panel = document.createElement("div");
    Object.assign(panel.style, {
        background: "#252525", borderRadius: "8px", padding: "14px",
        display: "flex", flexDirection: "column", gap: "10px",
        boxShadow: "0 8px 32px rgba(0,0,0,0.7)",
        maxWidth: "92vw",
    });

    // Title row
    const titleRow = document.createElement("div");
    Object.assign(titleRow.style, {
        display: "flex", justifyContent: "space-between", alignItems: "center",
    });
    const titleTxt = document.createElement("span");
    titleTxt.textContent = "DCT Spectrum Mask Editor";
    Object.assign(titleTxt.style, { color: "#ccc", fontSize: "13px", fontWeight: "bold" });

    const closeX = makeButton("✕", "#555", close);
    titleRow.append(titleTxt, closeX);

    // Hint
    const hint = document.createElement("div");
    hint.textContent = "LMB: remove frequencies (black) · RMB: restore (white) · Scroll: brush size";
    Object.assign(hint.style, { color: "#666", fontSize: "11px" });

    // Brush label
    const brushLbl = document.createElement("span");
    brushLbl.textContent = `Brush: ${brushSize} px`;
    Object.assign(brushLbl.style, { color: "#aaa", fontSize: "12px" });

    // Control buttons
    const ctrlRow = document.createElement("div");
    Object.assign(ctrlRow.style, { display: "flex", gap: "8px", alignItems: "center", flexWrap: "wrap" });

    const clearBtn  = makeButton("Clear All", "#444", () => {
        mctx.fillStyle = "white";
        mctx.fillRect(0, 0, cW, cH);
        render();
    });
    const invertBtn = makeButton("Invert", "#444", () => {
        const d = mctx.getImageData(0, 0, cW, cH);
        for (let i = 0; i < d.data.length; i += 4) {
            d.data[i]   = 255 - d.data[i];
            d.data[i+1] = 255 - d.data[i+1];
            d.data[i+2] = 255 - d.data[i+2];
        }
        mctx.putImageData(d, 0, 0);
        render();
    });
    const applyBtn  = makeButton("✔ Apply Mask", "#2a7a40", () => {
        if (maskWidget) maskWidget.value = maskC.toDataURL("image/png");
        close();
    });
    const cancelBtn = makeButton("✕ Cancel", "#7a2a2a", close);

    ctrlRow.append(brushLbl, clearBtn, invertBtn, applyBtn, cancelBtn);

    panel.append(titleRow, dispC, ctrlRow, hint);
    overlay.appendChild(panel);
    document.body.appendChild(overlay);

    // Click outside panel → cancel
    overlay.addEventListener("mousedown", e => { if (e.target === overlay) close(); });

    function close() {
        window.removeEventListener("mouseup", globalUp);
        if (overlay.parentNode) document.body.removeChild(overlay);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Extension registration
// ─────────────────────────────────────────────────────────────────────────────
app.registerExtension({
    name: "ComfyUI.DCT.MaskEditor",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "DCTMask") return;

        const origCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            origCreated?.apply(this, arguments);

            // Hide the raw mask_data STRING widget – it is managed by the modal.
            const mw = this.widgets?.find(w => w.name === "mask_data");
            if (mw) mw.computeSize = () => [0, -4];

            // "Edit Mask" button
            const self = this;
            this.addWidget("button", "✏  Edit Mask", null, () => openMaskEditor(self));

            if (this.size[0] < 300) this.size[0] = 300;
        };
    },
});
