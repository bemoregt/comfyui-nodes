/**
 * ComfyUI DCT Nodes – frontend extension
 *
 * Replaces the auto-generated STRING widget for "mask_data" on the
 * DCTMask node with an interactive paint-canvas widget.
 *
 * Controls
 * ─────────
 * Left-click  / drag   Paint black  →  zero those DCT frequencies
 * Right-click / drag   Paint white  →  restore those frequencies
 * Scroll wheel         Change brush size
 * Double-click         Clear canvas (all white = keep all frequencies)
 *
 * After the upstream DCT2D node runs, its spectrum_preview image is
 * automatically shown as the canvas background.
 */

import { app } from "../../scripts/app.js";

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────
const INTERNAL_SIZE = 512;   // off-screen mask canvas resolution
const DISPLAY_H     = 300;   // pixel height of the widget inside the node

// ─────────────────────────────────────────────────────────────────────────────
// Factory – creates one paint-canvas widget per node instance
// ─────────────────────────────────────────────────────────────────────────────
function createMaskCanvasWidget(initialValue) {

    /* ── Off-screen mask canvas ─────────────────────────────────────────── */
    const offscreen = document.createElement("canvas");
    offscreen.width  = INTERNAL_SIZE;
    offscreen.height = INTERNAL_SIZE;
    const octx = offscreen.getContext("2d");

    // Start all-white → keep all frequencies by default
    octx.fillStyle = "white";
    octx.fillRect(0, 0, INTERNAL_SIZE, INTERNAL_SIZE);

    /* ── State ──────────────────────────────────────────────────────────── */
    let bgImage      = null;    // HTMLImageElement – spectrum preview background
    let isPainting   = false;
    let eraseMode    = false;   // false = black (remove), true = white (restore)
    let brushSize    = 20;      // radius in internal canvas pixels
    let lastPos      = null;    // [x, y] for smooth line segments
    let widgetY      = 0;       // last y given to draw()
    let widgetH      = DISPLAY_H;
    let cachedValue  = null;    // memoised serialisation
    let isDirty      = true;

    /* ── Serialise / restore ────────────────────────────────────────────── */
    function serialise() {
        if (isDirty || !cachedValue) {
            cachedValue = offscreen.toDataURL("image/png");
            isDirty = false;
        }
        return cachedValue;
    }

    function restoreFromDataURL(url) {
        if (!url || url.length < 50) return;
        const img = new Image();
        img.onload = () => {
            octx.clearRect(0, 0, INTERNAL_SIZE, INTERNAL_SIZE);
            octx.drawImage(img, 0, 0, INTERNAL_SIZE, INTERNAL_SIZE);
            cachedValue = url;
            isDirty     = false;
        };
        img.src = url;
    }

    if (initialValue) restoreFromDataURL(initialValue);

    /* ── Painting ───────────────────────────────────────────────────────── */
    function paintAt(cx, cy, erase) {
        const color = erase ? "white" : "black";

        // Filled circle
        octx.beginPath();
        octx.arc(cx, cy, brushSize, 0, Math.PI * 2);
        octx.fillStyle = color;
        octx.fill();

        // Line segment back to last position for smooth strokes
        if (lastPos) {
            octx.beginPath();
            octx.moveTo(lastPos[0], lastPos[1]);
            octx.lineTo(cx, cy);
            octx.strokeStyle  = color;
            octx.lineWidth    = brushSize * 2;
            octx.lineCap      = "round";
            octx.stroke();
        }
        lastPos = [cx, cy];
        isDirty = true;
    }

    /* ── Coordinate helper ──────────────────────────────────────────────── */
    // Convert node-local [nx, ny] → internal canvas [cx, cy]
    function toCanvas(nx, ny, nodeW) {
        return [
            Math.round((nx / nodeW)              * INTERNAL_SIZE),
            Math.round(((ny - widgetY) / widgetH) * INTERNAL_SIZE),
        ];
    }

    /* ── Background refresh ─────────────────────────────────────────────── */
    function refreshBackground(node) {
        // Follow the dct_spectrum input (slot 0) back to the source node.
        // DCT2D outputs [dct_spectrum(0), spectrum_preview(1)].
        // ComfyUI stores IMAGE outputs in node.imgs[] (only IMAGE type outputs).
        // Because dct_spectrum is DCT_SPECTRUM type (not IMAGE), spectrum_preview
        // is the ONLY IMAGE output → it lands at node.imgs[0].
        const input = node.inputs?.[0];
        if (!input?.link) return;

        const linkData = app.graph.links[input.link];
        if (!linkData) return;

        const srcNode = app.graph.getNodeById(linkData.origin_id);
        if (!srcNode?.imgs) return;

        // imgs can be a flat array or nested; grab the first element
        const raw   = srcNode.imgs;
        const imgEl = Array.isArray(raw) ? raw[0] : raw;

        if (imgEl instanceof HTMLImageElement && imgEl.complete && imgEl.naturalWidth > 0) {
            bgImage = imgEl;
        }
    }

    /* ── LiteGraph widget object ────────────────────────────────────────── */
    const widget = {
        type  : "DCT_MASK_CANVAS",
        name  : "mask_data",         // must match Python INPUT_TYPES key

        get value() { return serialise(); },
        set value(v) {
            // Called by LiteGraph when restoring a saved workflow
            if (v && v.length > 50) restoreFromDataURL(v);
        },

        computeSize(width) {
            return [width, DISPLAY_H];
        },

        /* ── draw ─────────────────────────────────────────────────────── */
        draw(lgctx, node, widget_width, y, H) {
            widgetY = y;
            widgetH = H;

            const x = 0;

            // Background – spectrum preview when available, else dark fill
            if (bgImage) {
                lgctx.drawImage(bgImage, x, y, widget_width, H);
            } else {
                lgctx.fillStyle = "#141414";
                lgctx.fillRect(x, y, widget_width, H);

                // Hint text when no background is loaded
                lgctx.fillStyle = "#555";
                lgctx.font = "13px sans-serif";
                lgctx.textAlign = "center";
                lgctx.fillText(
                    "Run DCT2D node first to see spectrum background",
                    widget_width / 2, y + H / 2
                );
                lgctx.textAlign = "left";
            }

            // Mask overlay (semi-transparent)
            lgctx.globalAlpha = 0.55;
            lgctx.drawImage(offscreen, x, y, widget_width, H);
            lgctx.globalAlpha = 1.0;

            // Border
            lgctx.strokeStyle = "#555";
            lgctx.lineWidth   = 1;
            lgctx.strokeRect(x, y, widget_width, H);

            // Status bar at the bottom
            const barH = 20;
            lgctx.fillStyle = "rgba(0,0,0,0.65)";
            lgctx.fillRect(x + 1, y + H - barH, widget_width - 2, barH - 1);

            lgctx.fillStyle = "#bbb";
            lgctx.font      = "11px monospace";
            lgctx.fillText(
                `LMB: remove freq  |  RMB: restore  |  Scroll: brush ${brushSize}px  |  Dblclick: clear`,
                x + 6, y + H - 6
            );
        },

        /* ── mouse ────────────────────────────────────────────────────── */
        mouse(event, pos, node) {
            const [nx, ny] = pos;

            // Guard: only handle events inside the widget rectangle
            if (ny < widgetY || ny > widgetY + widgetH) return false;

            const nodeW       = node.size[0];
            const [cx, cy]    = toCanvas(nx, ny, nodeW);

            switch (event.type) {
                case "pointerdown":
                case "mousedown": {
                    if (event.button === 2) {
                        // Right-click – suppress context menu, paint white
                        event.preventDefault?.();
                        eraseMode = true;
                    } else {
                        eraseMode = false;
                    }
                    isPainting = true;
                    lastPos    = null;
                    paintAt(cx, cy, eraseMode);
                    node.setDirtyCanvas(true, true);
                    return true;
                }

                case "pointermove":
                case "mousemove": {
                    if (!isPainting) return false;
                    paintAt(cx, cy, eraseMode);
                    node.setDirtyCanvas(true, true);
                    return true;
                }

                case "pointerup":
                case "mouseup": {
                    isPainting = false;
                    lastPos    = null;
                    return true;
                }

                case "dblclick": {
                    // Clear → fill white (keep all frequencies)
                    octx.fillStyle = "white";
                    octx.fillRect(0, 0, INTERNAL_SIZE, INTERNAL_SIZE);
                    isDirty = true;
                    node.setDirtyCanvas(true, true);
                    return true;
                }

                case "wheel": {
                    const delta = event.deltaY > 0 ? -2 : 2;
                    brushSize   = Math.max(2, Math.min(120, brushSize + delta));
                    node.setDirtyCanvas(true, true);
                    event.preventDefault?.();
                    return true;
                }

                case "contextmenu": {
                    // Prevent LiteGraph context menu when right-clicking the canvas
                    return true;
                }
            }

            return false;
        },

        /* ── called after node executes ──────────────────────────────── */
        onExecuted(node) {
            refreshBackground(node);
        },
    };

    return widget;
}


// ─────────────────────────────────────────────────────────────────────────────
// ComfyUI Extension Registration
// ─────────────────────────────────────────────────────────────────────────────
app.registerExtension({
    name: "ComfyUI.DCT.MaskEditor",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "DCTMask") return;

        /* ── onNodeCreated ── inject paint widget ────────────────────── */
        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            origOnNodeCreated?.apply(this, arguments);

            // Find the auto-created STRING widget for mask_data
            const idx     = this.widgets?.findIndex(w => w.name === "mask_data") ?? -1;
            const saved   = idx >= 0 ? this.widgets[idx].value : "";

            const canvasW = createMaskCanvasWidget(saved);

            if (idx >= 0) {
                // Replace in-place so serialisation index stays consistent
                this.widgets[idx] = canvasW;
            } else {
                this.addCustomWidget(canvasW);
            }

            // Make the node wide enough to be useful by default
            if (this.size[0] < 340) {
                this.size[0] = 340;
            }
        };

        /* ── onExecuted ── refresh spectrum background ───────────────── */
        const origOnExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            origOnExecuted?.apply(this, arguments);

            const cw = this.widgets?.find(w => w.type === "DCT_MASK_CANVAS");
            cw?.onExecuted(this);
        };
    },
});
