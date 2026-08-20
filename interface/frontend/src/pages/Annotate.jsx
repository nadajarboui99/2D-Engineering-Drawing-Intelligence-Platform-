import { useState, useEffect, useRef } from "react";
import { api, saveAnnotation, API_BASE } from "../api/client";
import { Panel, FormRow, Select, Btn, Badge, InfoBox } from "../components/ui";

const META_DEFAULT = { standard: "unknown", source_type: "unknown", clutter: "med", has_gdt: false, difficulty: "med" };
const CLASS_COLOR = { dimension: "#6366F1", table: "#F59E0B" };

// in-progress annotation, kept outside React so it survives tab switches
let SESSION = null;
const DRAFT_IMG = "annotate_draft_img";   // {stem,w,h,isNew,dataUrl}, written on image load
const DRAFT_ANN = "annotate_draft_ann";   // {regions,feat,meta,drawClass,scale}, written on edit

function coerce(v, type) {
  if (v === "" || v == null) return null;
  if (type === "list") {
    // Separate items with ";" so a decimal comma (12,50) isn't mistaken for a separator.
    const arr = String(v).split(/[;\n]/).map(s => s.trim()).filter(Boolean).map(s => {
      const cleaned = s.replace(/,/g, ".").replace(/[^0-9.\-]/g, "");   // "Ø12,50" → "12.50"
      const num = parseFloat(cleaned);
      return Number.isNaN(num) ? s : num;
    });
    return arr.length ? arr : null;
  }
  if (type === "number" || type === "integer") {
    const n = Number(v);
    if (Number.isNaN(n)) return v;
    return type === "integer" ? Math.round(n) : n;
  }
  return v;
}

export default function AnnotatePage() {
  const [features, setFeatures] = useState([]);
  const [existing, setExisting] = useState([]);
  const [pending, setPending]   = useState(() => SESSION?.pending ?? []);     // {file, url, stem}
  const [img, setImg]           = useState(() => SESSION?.img ?? null);       // {url, stem, w, h, file|null}
  const [regions, setRegions]   = useState(() => SESSION?.regions ?? []);     // {class, bbox:[x,y,w,h], text, cells?}
  const [feat, setFeat]         = useState(() => SESSION?.feat ?? {});
  const [meta, setMeta]         = useState(() => SESSION?.meta ?? META_DEFAULT);
  const [drawClass, setDrawClass] = useState(() => SESSION?.drawClass ?? "dimension");
  const [scale, setScale]       = useState(() => SESSION?.scale ?? 1);
  const [sel, setSel]           = useState(() => SESSION?.sel ?? null);
  const [draft, setDraft]       = useState(null);
  const [saving, setSaving]     = useState(false);
  const [dirty, setDirty]       = useState(false);
  const [msg, setMsg]           = useState("");

  const svgRef  = useRef();
  const drawing = useRef(false);
  const saveRef = useRef(() => {});

  // Cmd/Ctrl+S saves (prevents the browser's save dialog).
  useEffect(() => {
    const h = (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "s") { e.preventDefault(); saveRef.current(); }
    };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, []);

  useEffect(() => { api.getFeatures().then(setFeatures).catch(() => {}); refreshExisting(); }, []);

  // Restore an unsaved draft after a full page reload (SESSION is null then).
  useEffect(() => {
    if (SESSION || img) return;
    try {
      const im = JSON.parse(localStorage.getItem(DRAFT_IMG) || "null");
      if (!im) return;
      const an = JSON.parse(localStorage.getItem(DRAFT_ANN) || "null") || {};
      setImg({ url: im.isNew ? im.dataUrl : `${API_BASE}/annotation/image/${encodeURIComponent(im.stem)}`,
               dataUrl: im.isNew ? im.dataUrl : null, stem: im.stem, w: im.w, h: im.h, isNew: im.isNew });
      setRegions(an.regions || []); setFeat(an.feat || {}); setMeta(an.meta || META_DEFAULT);
      setDrawClass(an.drawClass || "dimension"); setScale(an.scale || fitScale(im.w));
      setMsg("Restored your unsaved draft — save it to keep it.");
    } catch { /* ignore */ }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    SESSION = { pending, img, regions, feat, meta, drawClass, scale, sel };
  }, [pending, img, regions, feat, meta, drawClass, scale, sel]);

  useEffect(() => {
    if (!img) return;
    try { localStorage.setItem(DRAFT_ANN, JSON.stringify({ regions, feat, meta, drawClass, scale })); }
    catch { /* quota — ignore */ }
  }, [img, regions, feat, meta, drawClass, scale]);

  useEffect(() => {
    if (!dirty) return;
    const h = (e) => { e.preventDefault(); e.returnValue = ""; };
    window.addEventListener("beforeunload", h);
    return () => window.removeEventListener("beforeunload", h);
  }, [dirty]);

  function refreshExisting() { api.listAnnotations().then(setExisting).catch(() => setExisting([])); }
  function blankFeatures(list) { const o = {}; list.forEach(f => o[f.name] = { value: "", text: "" }); return o; }
  function fitScale(w) { return w ? Math.min(1, 760 / w) : 1; }
  function clearDraftStorage() { try { localStorage.removeItem(DRAFT_IMG); localStorage.removeItem(DRAFT_ANN); } catch { /* ignore */ } }

  // loading images
  async function onFiles(e) {
    const files = [...e.target.files];
    e.target.value = "";
    for (const f of files) {
      const stem = f.name.replace(/\.[^.]+$/, "");
      const isPdf = f.type === "application/pdf" || /\.pdf$/i.test(f.name);
      if (isPdf) {
        setMsg(`Converting ${f.name}…`);
        try {
          const blob = await api.renderPdf(f);                 // backend → PNG
          const png = new File([blob], stem + ".png", { type: "image/png" });
          setPending(p => [...p, { file: png, stem }]);
          setMsg("");
        } catch (err) {
          setMsg(`Couldn't read ${f.name}: ${err.message}`);
        }
      } else {
        setPending(p => [...p, { file: f, stem }]);
      }
    }
  }

  function removePending(item) {
    setPending(p => p.filter(it => it.stem !== item.stem));
    if (img?.stem === item.stem && img?.isNew) {   // loaded but not saved → clear canvas + draft
      setImg(null); setRegions([]); setSel(null); setDraft(null); setDirty(false); clearDraftStorage();
    }
  }

  function loadPending(item) {
    const reader = new FileReader();
    reader.onload = () => {
      const dataUrl = reader.result;
      const image = new Image();
      image.onload = () => {
        const w = image.naturalWidth, h = image.naturalHeight;
        setImg({ url: dataUrl, dataUrl, stem: item.stem, w, h, isNew: true });
        setRegions([]); setFeat(blankFeatures(features)); setMeta(META_DEFAULT);
        setSel(null); setDraft(null); setScale(fitScale(w)); setMsg(""); setDirty(false);
        try { localStorage.setItem(DRAFT_IMG, JSON.stringify({ stem: item.stem, w, h, isNew: true, dataUrl })); localStorage.removeItem(DRAFT_ANN); }
        catch { setMsg("Note: image too large to auto-save across reloads — save often."); }
      };
      image.src = dataUrl;
    };
    reader.readAsDataURL(item.file);
  }

  async function deleteExisting(stem) {
    if (!window.confirm(`Delete the annotation and image for “${stem}”? This cannot be undone.`)) return;
    try {
      await api.deleteAnnotation(stem);
      if (img?.stem === stem) { setImg(null); setRegions([]); setSel(null); setDirty(false); clearDraftStorage(); }
      refreshExisting();
    } catch (e) { setMsg("Error: " + e.message); }
  }

  async function loadExisting(stem) {
    setMsg("");
    const rec = await api.getMaster(stem);
    const url = `${API_BASE}/annotation/image/${encodeURIComponent(stem)}`;
    const image = new Image();
    image.onload = () => {
      const w = rec.width || image.naturalWidth, h = rec.height || image.naturalHeight;
      setImg({ url, dataUrl: null, stem, w, h, isNew: false });
      setRegions((rec.regions || []).map(r => ({ class: r.class || "dimension", bbox: r.bbox || [0, 0, 0, 0], text: r.text || "", cells: r.cells || [] })));
      const o = {};
      features.forEach(f => {
        const raw = (rec.features || {})[f.name]?.value;
        o[f.name] = { value: Array.isArray(raw) ? raw.join("; ") : (raw ?? ""), text: (rec.features || {})[f.name]?.text ?? "" };
      });
      setFeat(o); setMeta(rec.meta || META_DEFAULT);
      setSel(null); setDraft(null); setScale(fitScale(w)); setDirty(false);
      try { localStorage.setItem(DRAFT_IMG, JSON.stringify({ stem, w, h, isNew: false, dataUrl: null })); localStorage.removeItem(DRAFT_ANN); } catch { /* ignore */ }
    };
    image.src = url;
  }

  // drawing (SVG viewBox = natural image coords)
  function toNat(e) {
    const r = svgRef.current.getBoundingClientRect();
    const x = (e.clientX - r.left) * (img.w / r.width);
    const y = (e.clientY - r.top) * (img.h / r.height);
    return [Math.max(0, Math.min(img.w, x)), Math.max(0, Math.min(img.h, y))];
  }
  function onDown(e) {
    if (!img) return;
    const [x, y] = toNat(e);
    drawing.current = true;
    setDraft({ x0: x, y0: y, x1: x, y1: y });
    setSel(null);
    e.currentTarget.setPointerCapture(e.pointerId);
  }
  function onMove(e) {
    if (!drawing.current) return;
    const [x, y] = toNat(e);
    setDraft(d => (d ? { ...d, x1: x, y1: y } : d));
  }
  function onUp() {
    if (!drawing.current) return;
    drawing.current = false;
    const d = draft;
    setDraft(null);
    if (!d) return;
    const x = Math.min(d.x0, d.x1), y = Math.min(d.y0, d.y1);
    const w = Math.abs(d.x1 - d.x0), h = Math.abs(d.y1 - d.y0);
    if (w > 3 && h > 3) {
      setRegions(rs => [...rs, { class: drawClass, bbox: [Math.round(x), Math.round(y), Math.round(w), Math.round(h)], text: "", cells: [] }]);
      setSel(regions.length);
      setDirty(true);
    }
  }

  function updateRegion(i, patch) { setRegions(rs => rs.map((r, j) => j === i ? { ...r, ...patch } : r)); setDirty(true); }
  function deleteRegion(i) { setRegions(rs => rs.filter((_, j) => j !== i)); setSel(null); setDirty(true); }

  // save
  async function save(status = "draft") {
    if (!img) return;
    setSaving(true); setMsg("");
    try {
      const payload = {
        image: img.stem, width: img.w, height: img.h, meta, status,
        regions: regions.map((r, i) => ({
          id: i, class: r.class, bbox: r.bbox, text: r.text,
          ...(r.class === "table" ? { cells: r.cells || [] } : {}),
        })),
        features: Object.fromEntries(features.map(f => [f.name, {
          value: coerce(feat[f.name]?.value, f.type), text: feat[f.name]?.text || "",
        }])),
      };
      let file = null;
      if (img.isNew && img.dataUrl) {                       // new image → upload its bytes
        const blob = await (await fetch(img.dataUrl)).blob();
        const ext = blob.type === "image/jpeg" ? ".jpg" : ".png";
        file = new File([blob], img.stem + ext, { type: blob.type || "image/png" });
      }
      const res = await saveAnnotation(payload, file);
      setMsg(status === "complete" ? `Completed ✓ — now in the evaluation set (${res.regions} regions)`
                                   : `Draft saved ✓ — not in the evaluation set yet (${res.regions} regions)`);
      setPending(p => p.filter(it => it.stem !== img.stem));
      setImg(im => im ? { ...im, isNew: false, dataUrl: null, status, url: `${API_BASE}/annotation/image/${encodeURIComponent(im.stem)}` } : im);
      setDirty(false); clearDraftStorage();
      refreshExisting();
    } catch (e) {
      setMsg("Error: " + e.message);
    } finally { setSaving(false); }
  }
  // Ctrl/Cmd+S = quick draft save (never accidentally promotes to the eval set).
  saveRef.current = () => { if (img && !saving) save("draft"); };

  const fs = 11 / (scale || 1);   // label ~11px on screen at any zoom (viewBox scales it back down)

  return (
    <div>
      <InfoBox>
        Add images, draw a box for each dimension/table (drag on the image), type its text, then fill the
        feature values. <strong>Save</strong> writes the annotation and copies the image into <code>dataset/selected_images/</code>.
      </InfoBox>

      {/* Load images */}
      <Panel title="Images">
        <label className="inline-flex items-center gap-2 text-sm cursor-pointer border border-gray-200 rounded-lg px-3 py-1.5 hover:bg-gray-50">
          + Add drawings (image or PDF)
          <input type="file" accept=".jpg,.jpeg,.png,.pdf" multiple onChange={onFiles} className="hidden" />
        </label>
        {pending.length > 0 && (
          <div className="mt-3">
            <p className="text-xs text-gray-400 mb-1">To annotate ({pending.length})</p>
            <div className="flex flex-wrap gap-2">
              {pending.map(it => (
                <div key={it.stem}
                  className={`text-xs rounded-lg border flex items-center ${img?.stem === it.stem ? "border-gray-900 bg-gray-50" : "border-gray-200"}`}>
                  <button onClick={() => loadPending(it)} className="pl-2.5 pr-1.5 py-1 hover:bg-gray-50 rounded-l-lg">
                    {it.stem.slice(0, 24)}
                  </button>
                  <button onClick={() => removePending(it)} title="Remove from queue (won't be annotated)"
                    className="px-1.5 py-1 text-gray-300 hover:text-red-500 border-l border-gray-100">✕</button>
                </div>
              ))}
            </div>
          </div>
        )}
        {existing.length > 0 && (
          <div className="mt-3">
            <p className="text-xs text-gray-400 mb-1">Saved — click to edit ({existing.length})</p>
            <div className="flex flex-wrap gap-2">
              {existing.map(it => (
                <div key={it.image}
                  className={`text-xs rounded-lg border flex items-center ${img?.stem === it.image ? "border-gray-900 bg-gray-50" : "border-gray-200"}`}>
                  <button onClick={() => loadExisting(it.image)} className="pl-2.5 pr-1.5 py-1 flex items-center gap-1.5 hover:bg-gray-50 rounded-l-lg">
                    {it.image.slice(0, 20)}
                    {it.status === "draft" && <Badge variant="amber">draft</Badge>}
                    <Badge variant={it.with_text === it.regions && it.regions > 0 ? "green" : "gray"}>{it.with_text}/{it.regions}</Badge>
                  </button>
                  <button onClick={() => deleteExisting(it.image)} title="Delete annotation"
                    className="px-1.5 py-1 text-gray-300 hover:text-red-500 border-l border-gray-100">✕</button>
                </div>
              ))}
            </div>
          </div>
        )}
      </Panel>

      {!img && <div className="text-center py-16 text-gray-300 text-sm">Add images above, then click one to start annotating.</div>}

      {img && (
        <div className="flex gap-4 items-start">
          {/* Canvas */}
          <div className="flex-1 min-w-0">
            <Panel title={img.stem.slice(0, 40)}
              action={
                <div className="flex items-center gap-2">
                  {["dimension", "table"].map(c => (
                    <button key={c} onClick={() => setDrawClass(c)}
                      className={`text-xs px-2 py-0.5 rounded-full border ${drawClass === c ? "text-white border-transparent" : "text-gray-600 border-gray-200"}`}
                      style={drawClass === c ? { background: CLASS_COLOR[c] } : {}}>
                      {c}
                    </button>
                  ))}
                  <span className="text-gray-200">|</span>
                  <button onClick={() => setScale(s => Math.max(0.1, s - 0.15))} className="text-xs px-1.5 border border-gray-200 rounded">−</button>
                  <span className="text-xs text-gray-400 w-10 text-center">{Math.round(scale * 100)}%</span>
                  <button onClick={() => setScale(s => Math.min(4, s + 0.15))} className="text-xs px-1.5 border border-gray-200 rounded">+</button>
                  <button onClick={() => setScale(fitScale(img.w))} className="text-xs px-1.5 border border-gray-200 rounded">fit</button>
                </div>
              }>
              <div className="overflow-auto border border-gray-100 rounded-lg bg-gray-50" style={{ maxHeight: "70vh" }}>
                <div style={{ position: "relative", width: img.w * scale, height: img.h * scale }}>
                  <img src={img.url} width={img.w * scale} height={img.h * scale} draggable={false} style={{ display: "block", userSelect: "none" }} />
                  <svg ref={svgRef} viewBox={`0 0 ${img.w} ${img.h}`} width={img.w * scale} height={img.h * scale}
                    onPointerDown={onDown} onPointerMove={onMove} onPointerUp={onUp} onPointerLeave={onUp}
                    style={{ position: "absolute", top: 0, left: 0, cursor: "crosshair", touchAction: "none" }}>
                    {regions.map((r, i) => {
                      const [bx, by, bw, bh] = r.bbox;
                      const color = CLASS_COLOR[r.class] || "#6366F1";
                      const label = `${i}`;
                      const above = by > fs * 1.4;                       // room above the box?
                      const ty = above ? by - fs * 0.35 : by + bh + fs;
                      return (
                        <g key={i} onPointerDown={(e) => { e.stopPropagation(); setSel(i); }} style={{ cursor: "pointer" }}>
                          <rect x={bx} y={by} width={bw} height={bh}
                            fill={sel === i ? "rgba(29,158,117,0.12)" : "rgba(99,102,241,0.04)"}
                            stroke={color} strokeWidth={sel === i ? 1.75 : 1} vectorEffect="non-scaling-stroke" />
                          <text x={bx} y={ty} fontSize={fs} fill={color} stroke="#fff" strokeWidth={fs * 0.14}
                            paintOrder="stroke" style={{ userSelect: "none", fontWeight: 600 }}>{label}</text>
                        </g>
                      );
                    })}
                    {draft && (
                      <rect x={Math.min(draft.x0, draft.x1)} y={Math.min(draft.y0, draft.y1)}
                        width={Math.abs(draft.x1 - draft.x0)} height={Math.abs(draft.y1 - draft.y0)}
                        fill="rgba(0,0,0,0.05)" stroke="#111" strokeDasharray="4" vectorEffect="non-scaling-stroke" />
                    )}
                  </svg>
                </div>
              </div>
              <p className="text-xs text-gray-400 mt-2">Drag to draw a “{drawClass}” box · click a box to select · scroll to pan</p>
            </Panel>
          </div>

          {/* Side panels — pinned so the image stays in view while you scroll regions */}
          <div className="w-80 shrink-0 sticky top-0 self-start">
            <Panel title="Regions" action={<span className="text-xs text-gray-400">{regions.length}</span>}>
              {regions.length === 0 && <p className="text-xs text-gray-400">No boxes yet — drag on the image.</p>}
              <div className="overflow-y-auto -mr-2 pr-2" style={{ maxHeight: "32vh" }}>
              {regions.map((r, i) => (
                <div key={i} className={`py-2 border-b border-gray-50 last:border-0 ${sel === i ? "bg-gray-50 -mx-2 px-2 rounded" : ""}`}
                  onClick={() => setSel(i)}>
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-xs font-mono text-gray-400 w-4">{i}</span>
                    <Select value={r.class} onChange={v => updateRegion(i, { class: v })}
                      options={[{ value: "dimension", label: "dimension" }, { value: "table", label: "table" }]} />
                    <button onClick={(e) => { e.stopPropagation(); deleteRegion(i); }} className="text-xs text-gray-300 hover:text-red-400">✕</button>
                  </div>
                  <input value={r.text} onChange={e => updateRegion(i, { text: e.target.value })}
                    placeholder={r.class === "table" ? "table text…" : "e.g. ⌀25±0.1"}
                    className="w-full text-sm border border-gray-200 rounded px-2 py-1 focus:outline-none focus:ring-1 focus:ring-gray-300" />
                </div>
              ))}
              </div>
            </Panel>

            <Panel title="Features">
              {features.length === 0 && <p className="text-xs text-gray-400">No feature schema loaded.</p>}
              <div className="overflow-y-auto -mr-2 pr-2" style={{ maxHeight: "32vh" }}>
              {features.map(f => (
                <div key={f.name} className="mb-2.5">
                  <label className="block text-xs text-gray-600 mb-0.5 break-all">
                    {f.name}
                    <span className="text-gray-400">{f.unit ? ` (${f.unit})` : f.type === "list" ? " (list)" : ""}</span>
                  </label>
                  <input value={feat[f.name]?.value ?? ""} onChange={e => { setFeat(s => ({ ...s, [f.name]: { ...s[f.name], value: e.target.value } })); setDirty(true); }}
                    placeholder={f.type === "list" ? "separate holes with ;  ·  e.g. 12,50; 8; 8" : "—"}
                    className="w-full text-sm border border-gray-200 rounded px-2 py-1 focus:outline-none focus:ring-1 focus:ring-gray-300" />
                </div>
              ))}
              </div>
            </Panel>

            <div className="flex items-center gap-3 flex-wrap">
              <Btn onClick={() => save("draft")} loading={saving} disabled={saving}>💾 Save draft</Btn>
              <Btn primary onClick={() => save("complete")} loading={saving} disabled={saving}>✓ Save &amp; mark complete</Btn>
              {msg && <span className={`text-xs ${msg.startsWith("Error") ? "text-red-500" : "text-emerald-600"}`}>{msg}</span>}
            </div>
            <p className="text-xs text-gray-400 mt-2">
              <strong>Draft</strong> keeps your work but stays out of evaluation. <strong>Mark complete</strong> adds it to the evaluation set and ground truth. (Ctrl/Cmd+S saves a draft.)
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
