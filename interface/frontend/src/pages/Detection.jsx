import { useState, useEffect, useRef } from "react";
import { api, pollJob, API_BASE } from "../api/client";
import { Panel, FormRow, Select, Btn, Tabs, JobLog, InfoBox, MetricCard, Collapsible, Badge } from "../components/ui";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts";

function shuffle(arr) {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
  return a;
}
function iou(a, b) {
  const ix1 = Math.max(a[0], b[0]), iy1 = Math.max(a[1], b[1]);
  const ix2 = Math.min(a[2], b[2]), iy2 = Math.min(a[3], b[3]);
  const iw = Math.max(0, ix2 - ix1), ih = Math.max(0, iy2 - iy1);
  const inter = iw * ih;
  const ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter;
  return ua > 0 ? inter / ua : 0;
}

// Draws the drawing with GT boxes (green=found / red=missed) and predicted boxes,
// class-aware: table preds purple, dimension preds blue.
const asBox  = g => Array.isArray(g) ? { box: g, cls: null } : g;
const asPred = p => Array.isArray(p) ? { box: p, score: 1, cls: null } : p;

function DetOverlay({ rec, conf, onZoom }) {
  const W = rec.width || 1000, H = rec.height || 1000;
  const gt    = rec.gt.map(asBox);
  const preds = rec.pred.map(asPred).filter(p => p.score >= conf);
  const gtHit = gt.map(g => preds.some(p => (p.cls == null || g.cls == null || p.cls === g.cls) && iou(g.box, p.box) >= 0.5));
  const missed = gtHit.filter(h => !h).length;
  const sw = Math.max(W, H) / 350;
  const predColor = c => c === "table" ? "#7C3AED" : "#2563EB";
  return (
    <div>
      <div className="relative cursor-zoom-in group" onClick={() => onZoom?.(rec)}>
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto rounded-lg border border-gray-100 bg-gray-50">
          <image href={`${API_BASE}/annotation/image/${encodeURIComponent(rec.image)}`} x="0" y="0" width={W} height={H} />
          {preds.map((p, i) => (
            <rect key={"p"+i} x={p.box[0]} y={p.box[1]} width={p.box[2]-p.box[0]} height={p.box[3]-p.box[1]}
              fill="none" stroke={predColor(p.cls)} strokeWidth={sw} opacity="0.7" />
          ))}
          {gt.map((g, i) => (
            <g key={"g"+i}>
              <rect x={g.box[0]} y={g.box[1]} width={g.box[2]-g.box[0]} height={g.box[3]-g.box[1]}
                fill="none" stroke={gtHit[i] ? "#10B981" : "#EF4444"} strokeWidth={sw*1.6}
                strokeDasharray={gtHit[i] ? "none" : `${sw*3},${sw*2}`} />
              {g.cls && <text x={g.box[0]} y={g.box[1] - sw} fontSize={sw*5} fill={gtHit[i] ? "#10B981" : "#EF4444"}>{g.cls}</text>}
            </g>
          ))}
        </svg>
        {onZoom && <span className="absolute bottom-1 right-1 text-[10px] bg-black/60 text-white px-1.5 py-0.5 rounded opacity-0 group-hover:opacity-100">🔍 enlarge</span>}
      </div>
      <p className="text-xs text-gray-400 mt-1.5">
        <span className="text-blue-600">{preds.length} predicted</span> (of {rec.n_pred_total} raw) ·
        <span className="text-emerald-600"> {gt.length - missed}/{gt.length} annotated found</span> ·
        {missed > 0 ? <span className="text-red-500"> {missed} missed</span> : <span className="text-emerald-600"> none missed</span>}
      </p>
    </div>
  );
}

export default function DetectionPage() {
  const [task, setTask]           = useState("tables");
  const [weights, setWeights]     = useState([]);
  const [selectedW, setSelectedW] = useState("");
  const [imgsz, setImgsz]         = useState(640);
  const [results, setResults]     = useState([]);
  const [logs, setLogs]           = useState([]);
  const [annotated, setAnnotated] = useState(null);   // metrics + per-image boxes
  const [annRunning, setAnnRunning] = useState(false);
  // Box-overlay viewer
  const [conf, setConf]           = useState(0.25);
  const [numImages, setNumImages] = useState(0);
  const [order, setOrder]         = useState([]);
  const [zoomRec, setZoomRec]     = useState(null);

  // Upload form
  const [uploadName, setUploadName] = useState("");
  const [uploadFile, setUploadFile] = useState(null);
  const fileRef = useRef();

  useEffect(() => { loadAll(); }, [task]);

  async function loadAll() {
    try {
      const w = await api.getWeightsList(task);
      setWeights(w);
      if (w.length > 0 && !selectedW) setSelectedW(w[0].id);
      const r = await api.getDetectionResults(task);
      setResults(r);
      // Auto-load the last saved annotated eval (metrics + box overlay) for this task.
      const saved = await api.getDetectionAnnotated(task);
      if (saved && saved.images?.length) {
        setAnnotated(saved);
        const imgs = saved.images.map(im => im.image);
        setOrder(shuffle(imgs));
        setNumImages(imgs.length);
      } else {
        setAnnotated(null);
      }
    } catch {}
  }

  async function uploadWeights() {
    if (!uploadFile || !uploadName) return;
    const form = new FormData();
    form.append("file", uploadFile);
    form.append("name", uploadName);
    form.append("task", task);
    await fetch("http://localhost:8000/weights/upload", { method:"POST", body:form });
    setUploadName(""); setUploadFile(null);
    if (fileRef.current) fileRef.current.value = "";
    await loadAll();
  }

  async function runAnnotatedEval() {
    const w = weights.find(w => w.id === selectedW);
    if (task !== "both" && !w) return;   // "both" auto-uses the best of each detector
    if (w && w.source === "custom-arch" && !w.available) {
      setLogs([{ ts: new Date().toISOString(), msg: `"${w.name}" needs its dependencies. Install: ${w.install_cmd}` }]);
      return;
    }
    setAnnRunning(true); setLogs([]); setAnnotated(null);
    try {
      const archId = (w && w.source === "custom-arch") ? w.arch_id : null;
      const { job_id } = await api.evalDetectionAnnotated(task, task === "both" ? null : w?.path, imgsz, task === "both" ? null : archId);
      const result = await pollJob(job_id, setLogs);
      setAnnotated(result);
      const imgs = (result?.images || []).map(im => im.image);
      setOrder(shuffle(imgs));
      setNumImages(imgs.length);
    } catch (e) {
      setLogs(l => [...l, { ts: new Date().toISOString(), msg: `Error: ${e.message}` }]);
    } finally { setAnnRunning(false); }
  }

  async function removeWeights(id) {
    await api.deleteWeights(id);
    setSelectedW("");
    await loadAll();
  }

  const chartData = results.map(r => ({
    name: r.run?.slice(0,14),
    mAP:  parseFloat(r.map50?.toFixed(3)     || 0),
    P:    parseFloat(r.precision?.toFixed(3)  || 0),
    R:    parseFloat(r.recall?.toFixed(3)     || 0),
  }));
  const best = results[0] ?? null;

  return (
    <div>
      <Tabs tabs={[{id:"tables",label:"Tables"},{id:"dimensions",label:"Dimensions"},{id:"both",label:"Both"}]}
        active={task} onChange={t => { setTask(t); setSelectedW(""); setLogs([]); setAnnotated(null); }} />

      {/* Model selection — pick from trained runs found on disk (auto for "both") */}
      <Panel title="Model">
        {task === "both" ? (
          <InfoBox>Runs the best <strong>table</strong> and best <strong>dimension</strong> detector together and scores/overlays both classes on each drawing.</InfoBox>
        ) : (
          <>
            <FormRow label="Detection model" hint="Trained runs found on disk">
              {weights.length === 0
                ? <InfoBox>No trained weights found for {task} under Table_dimensions_detection/runs/detect.</InfoBox>
                : <Select value={selectedW} onChange={setSelectedW}
                    options={weights.map(w => ({ value: w.id, label: w.name }))} />}
            </FormRow>
            {selectedW && (
              <p className="text-xs text-gray-400 font-mono truncate pl-[8.75rem]">
                {weights.find(w => w.id === selectedW)?.path}
              </p>
            )}
          </>
        )}

        {task !== "both" && (
          <Collapsible title="Add a model">
            <p className="text-xs text-gray-500 mb-2">Upload YOLO / RT-DETR weights (<code className="bg-gray-100 px-1 rounded">.pt</code>) for <strong>{task}</strong>:</p>
            <div className="flex gap-3 flex-wrap">
              <input type="text" placeholder="Model name (e.g. yolov11n_tables)" value={uploadName}
                onChange={e => setUploadName(e.target.value)}
                className="flex-1 text-sm border border-gray-200 rounded-lg px-3 py-1.5" />
              <input type="file" accept=".pt" ref={fileRef} onChange={e => setUploadFile(e.target.files[0])}
                className="text-sm text-gray-500 file:mr-3 file:text-xs file:border-0 file:bg-gray-100 file:rounded-lg file:px-3 file:py-1.5" />
              <Btn onClick={uploadWeights} disabled={!uploadFile || !uploadName}>+ Upload</Btn>
            </div>
            <p className="text-xs text-gray-400 mt-3">
              <strong>Other architectures</strong> (e.g. Table Transformer) appear in the model dropdown above with "(install required)".
              Run their install command, then select and evaluate them — no code needed. To add a brand-new architecture,
              drop a wrapper in <code className="bg-gray-100 px-1 rounded">Table_dimensions_detection/models/</code> and register it (see <code className="bg-gray-100 px-1 rounded">detection_registry.py</code>).
            </p>
          </Collapsible>
        )}
      </Panel>

      {/* Eval config */}
      <Panel title="Evaluation">
        <FormRow label="Image size" hint="Must match the size the model was trained on">
          <Select value={imgsz} onChange={v => setImgsz(Number(v))}
            options={[{value:640,label:"640px"},{value:1280,label:"1280px"}]} />
        </FormRow>
        <Btn primary onClick={runAnnotatedEval} loading={annRunning} disabled={annRunning || (task !== "both" && !selectedW)}>
          ▶ Run evaluation
        </Btn>
        {task !== "both" && !selectedW && <p className="text-xs text-red-400 mt-2">Select a model first.</p>}
        <p className="text-xs text-gray-400 mt-2">Evaluates on the images you labelled in the Annotate tab — metrics plus the predicted-vs-annotated box overlay appear below.</p>
      </Panel>

      {(annRunning || logs.length > 0) && <Panel title="Live logs"><JobLog logs={logs} /></Panel>}

      {annotated?.available && (
        <Panel title="Results — on YOUR annotated set" badge="real test">
          <div className="grid grid-cols-4 gap-3">
            <MetricCard label="mAP@0.5"   value={annotated.map50?.toFixed(3)}     good={annotated.map50 > 0.8} />
            <MetricCard label="Precision" value={annotated.precision?.toFixed(3)} good={annotated.precision > 0.8} sub="of detections correct" />
            <MetricCard label="Recall"    value={annotated.recall?.toFixed(3)}    good={annotated.recall > 0.8}  sub="of real boxes found" />
            <MetricCard label="F1"        value={annotated.f1?.toFixed(3)}        good={annotated.f1 > 0.8} />
          </div>
          <p className="text-xs text-gray-400 mt-2">
            {annotated.n_gt} annotated boxes · TP {annotated.tp} · FP {annotated.fp} · FN {annotated.fn}
            {" "}· conf ≥ {annotated.conf_threshold} · IoU ≥ {annotated.iou_threshold}
          </p>
          {annotated.per_class && Object.keys(annotated.per_class).length > 1 && (
            <div className="grid grid-cols-2 gap-3 mt-3">
              {Object.entries(annotated.per_class).map(([cls, m]) => (
                <div key={cls} className="border border-gray-100 rounded-lg p-3">
                  <p className="text-xs font-medium text-gray-600 mb-1 capitalize">{cls}</p>
                  <p className="text-xs text-gray-500 font-mono">mAP {m.map50?.toFixed(3)} · R {m.recall?.toFixed(3)} · P {m.precision?.toFixed(3)} · TP {m.tp}/FN {m.fn} of {m.n_gt}</p>
                </div>
              ))}
            </div>
          )}
        </Panel>
      )}
      {annotated && annotated.available === false && (
        <InfoBox>{annotated.note || "No annotated ground truth for this task yet — label some images in the Annotate tab first."}</InfoBox>
      )}

      {/* ── Detected boxes overlaid on the drawings (auto after eval) ────────── */}
      {annotated?.images?.length > 0 && (() => {
        const total = annotated.images.length;
        const n = Math.max(1, Math.min(numImages || total, total));
        const byImage = Object.fromEntries(annotated.images.map(im => [im.image, im]));
        const selected = order.slice(0, n).filter(Boolean);
        return (
          <Panel title="Detected boxes on your drawings" badge={task === "both" ? "tables + dimensions" : task}>
            <div className="flex flex-wrap items-center gap-4 mb-3 text-sm text-gray-500">
              <div className="flex items-center gap-2">
                Confidence ≥ <strong className="text-gray-700 w-8">{conf.toFixed(2)}</strong>
                <input type="range" min="0.05" max="0.9" step="0.05" value={conf}
                  onChange={e => setConf(parseFloat(e.target.value))} className="w-40 accent-gray-900" />
              </div>
              <div className="flex items-center gap-2">
                Show <input type="number" min={1} max={total} value={numImages}
                  onChange={e => setNumImages(Math.max(1, Math.min(total, parseInt(e.target.value) || 1)))}
                  className="w-16 border border-gray-200 rounded-lg px-2 py-1 text-center" /> of {total}
                <Btn small onClick={() => setOrder(shuffle(annotated.images.map(im => im.image)))}>↻ shuffle</Btn>
              </div>
            </div>
            <div className="flex flex-wrap gap-4 mb-4 text-xs text-gray-400">
              <span className="flex items-center gap-1"><span className="w-3 h-2 border-2 border-blue-600" />predicted (dimension)</span>
              {task === "both" && <span className="flex items-center gap-1"><span className="w-3 h-2 border-2 border-purple-600" />predicted (table)</span>}
              <span className="flex items-center gap-1"><span className="w-3 h-2 border-2 border-emerald-500" />annotated · found</span>
              <span className="flex items-center gap-1"><span className="w-3 h-2 border-2 border-red-500 border-dashed" />annotated · missed</span>
            </div>
            <div className="grid grid-cols-2 gap-4">
              {selected.map(img => byImage[img] && (
                <div key={img}>
                  <p className="font-mono text-xs text-gray-600 mb-1">{img}</p>
                  <DetOverlay rec={byImage[img]} conf={conf} onZoom={setZoomRec} />
                </div>
              ))}
            </div>
          </Panel>
        );
      })()}

      {zoomRec && (
        <div className="fixed inset-0 z-50 bg-black/85 flex items-center justify-center p-4" onClick={() => setZoomRec(null)}>
          <div className="max-w-[95vw] max-h-[95vh] overflow-auto bg-white rounded-xl p-3" onClick={e => e.stopPropagation()}>
            <div className="flex items-center justify-between mb-2">
              <p className="font-mono text-xs text-gray-600">{zoomRec.image}</p>
              <button onClick={() => setZoomRec(null)} className="text-gray-400 hover:text-gray-800 text-xl leading-none">✕</button>
            </div>
            <div className="w-[80vw] max-w-[1100px]"><DetOverlay rec={zoomRec} conf={conf} /></div>
          </div>
        </div>
      )}

      {best && (
        <>
          <p className="text-xs text-gray-400 mb-1.5 mt-1">Training-time metrics (original val set — not your annotated data)</p>
          <div className="grid grid-cols-4 gap-3 mb-4">
            <MetricCard label="mAP@0.5"   value={best.map50?.toFixed(3)}     good={best.map50 > 0.8}  sub="main detection metric" />
            <MetricCard label="Precision" value={best.precision?.toFixed(3)} good={best.precision > 0.8} sub="% of detections correct" />
            <MetricCard label="Recall"    value={best.recall?.toFixed(3)}    good={best.recall > 0.8}  sub="% of real objects found" />
            <MetricCard label="F1"        value={best.f1?.toFixed(3)}        good={best.f1 > 0.8}  sub="harmonic mean of P & R" />
          </div>

          <Collapsible title="How to read these metrics">
            <div className="space-y-2 text-xs text-gray-500">
              <p><strong className="text-gray-700">mAP@0.5</strong> — A detection is correct if the predicted box overlaps the real box by ≥50%. mAP averages this across all images. Above 0.85 is good.</p>
              <p><strong className="text-gray-700">Low precision</strong> → model detects too much noise. Try higher confidence threshold.</p>
              <p><strong className="text-gray-700">Low recall</strong> → model misses real objects. Try lower confidence threshold or a larger model size.</p>
            </div>
          </Collapsible>
        </>
      )}

      {chartData.length > 0 && (
        <Panel title="Runs comparison">
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={chartData} barGap={2}>
              <XAxis dataKey="name" tick={{fontSize:11}} />
              <YAxis domain={[0,1]} tick={{fontSize:11}} />
              <Tooltip formatter={v => v.toFixed(3)} />
              <Bar dataKey="mAP" name="mAP@0.5"   fill="#1D9E75" radius={[3,3,0,0]} />
              <Bar dataKey="P"   name="Precision"  fill="#6366F1" radius={[3,3,0,0]} />
              <Bar dataKey="R"   name="Recall"     fill="#F59E0B" radius={[3,3,0,0]} />
            </BarChart>
          </ResponsiveContainer>
          <div className="flex gap-4 justify-center mt-2">
            {[["#1D9E75","mAP@0.5"],["#6366F1","Precision"],["#F59E0B","Recall"]].map(([c,l]) => (
              <div key={l} className="flex items-center gap-1.5 text-xs text-gray-500">
                <div className="w-3 h-3 rounded-sm" style={{background:c}} />{l}
              </div>
            ))}
          </div>
        </Panel>
      )}

      {/* ── "Train a new model" hidden for now ───────────────────────────────
      <Collapsible title="Train a new model">
        <InfoBox>Training creates new weights. After training, upload the resulting best.pt above.</InfoBox>
        <p className="text-xs text-gray-500">Run from your terminal inside <code className="bg-gray-100 px-1 rounded">Table_dimensions_detection/</code>:</p>
        <pre className="bg-gray-950 text-green-400 text-xs rounded-lg p-3 mt-2 overflow-x-auto">
{`python train_${task === "tables" ? "tables" : "dimensions"}.py \\
  --config configs/${task === "tables" ? "tables" : "dimensions"}.yaml`}
        </pre>
        <p className="text-xs text-gray-400 mt-2">Then upload <code className="bg-gray-100 px-1 rounded">runs/detect/.../weights/best.pt</code> above.</p>
      </Collapsible>
      ──────────────────────────────────────────────────────────────────────── */}
    </div>
  );
}