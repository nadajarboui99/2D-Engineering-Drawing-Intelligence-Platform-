import { useState, useEffect } from "react";
import { api, pollJob, API_BASE } from "../api/client";
import { Panel, FormRow, Select, Btn, Tabs, JobLog, Badge, InfoBox, MetricCard } from "../components/ui";

const APPROACHES = { full: "Full image", crop: "Cropped patches" };

function shuffle(arr) {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
  return a;
}

function Thumb({ stem, onOpen }) {
  return (
    <div className="relative group cursor-zoom-in" onClick={onOpen} title="Click to enlarge">
      <img src={`${API_BASE}/annotation/image/${encodeURIComponent(stem)}`} alt={stem}
        onError={e => { e.currentTarget.style.display = "none"; }}
        className="rounded-lg border border-gray-100 w-full object-contain bg-gray-50 group-hover:ring-2 group-hover:ring-gray-900/20" />
      <span className="absolute bottom-1 right-1 text-[10px] bg-black/60 text-white px-1.5 py-0.5 rounded opacity-0 group-hover:opacity-100">🔍 enlarge</span>
    </div>
  );
}

// Annotated text shown as word chips, green = OCR found it, red = missed.
function Chips({ tokens, flags }) {
  if (!tokens?.length) return <span className="text-xs text-gray-300">— no annotated text —</span>;
  return (
    <div className="flex flex-wrap gap-1">
      {tokens.map((t, i) => (
        <span key={i} className={`px-1.5 py-0.5 rounded text-xs ${flags[i] ? "bg-emerald-100 text-emerald-700" : "bg-red-100 text-red-600"}`}>{t}</span>
      ))}
    </div>
  );
}

function cov(m) { return m?.word_coverage != null ? `${(m.word_coverage * 100).toFixed(0)}%` : "—"; }

// Shows the detector's crops for one drawing so you can judge patch quality.
function CropStrip({ image, model }) {
  const [data, setData] = useState(null);
  useEffect(() => { api.getOCRCrops(image, model).then(setData).catch(() => setData({ crops: [], count: 0 })); }, [image, model]);
  if (!data) return <p className="text-xs text-gray-300">loading crops…</p>;
  if (!data.count) return <p className="text-xs text-amber-600">Detector produced <strong>0 crops</strong> for this drawing — nothing to read. (Run detection / lower the confidence threshold.)</p>;
  return (
    <div>
      <p className="text-xs text-gray-500 mb-1">{data.count} detected crop(s) — check the boxes are on the right regions and legible:</p>
      <div className="flex gap-2 overflow-x-auto pb-1">
        {data.crops.map((c, i) => (
          <div key={i} className="shrink-0 w-28">
            <img src={`${API_BASE}${c.url}`} alt="" className="rounded border border-gray-100 bg-gray-50 h-16 w-full object-contain" />
            <p className="text-[10px] text-gray-700 truncate mt-0.5" title={c.text}>{c.text || <em className="text-gray-300">empty</em>}</p>
            <p className="text-[10px] text-gray-400">{c.task} · conf {c.confidence?.toFixed(2)}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function OCRPage() {
  const [view, setView]         = useState("full");   // full | crop | compare
  const [cfg, setCfg]           = useState({ ocr_model: "easyocr", task: "both", conf_threshold: 0.25 });
  const [logs, setLogs]         = useState([]);
  const [running, setRunning]   = useState(false);
  const [ocrModels, setOcrModels] = useState([]);

  const [detail, setDetail]     = useState(null);   // {full:[...], crop:[...]}
  const [compare, setCompare]   = useState(null);   // {full:{...}, crop:{...}}
  const [gtImages, setGtImages] = useState([]);
  const [numImages, setNumImages] = useState(0);
  const [order, setOrder]       = useState([]);
  const [zoomImg, setZoomImg]   = useState(null);
  const [zoomFit, setZoomFit]   = useState(true);

  useEffect(() => { loadModels(); }, []);
  useEffect(() => { refreshDetail(); }, [cfg.ocr_model]);
  useEffect(() => { setOrder(shuffle(gtImages)); setNumImages(gtImages.length); }, [gtImages.join(",")]);

  function refreshDetail() {
    api.getOCRDetail(cfg.ocr_model)
      .then(d => { setDetail(d.detail); setCompare(d.compare); setGtImages(d.gt_images || []); })
      .catch(() => { setDetail(null); setCompare(null); setGtImages([]); });
  }
  async function loadModels() {
    try { setOcrModels(await fetch(`${API_BASE}/ocr-models/list`).then(r => r.json())); } catch {}
  }
  async function run() {
    setRunning(true); setLogs([]);
    try {
      const mode = view;  // "full" | "crop"
      const task = view === "full" ? "all" : cfg.task;
      const { job_id } = await api.runOCR({ ...cfg, task, mode });
      await pollJob(job_id, setLogs);
      refreshDetail();
    } catch (e) {
      setLogs(l => [...l, { ts: new Date().toISOString(), msg: `Error: ${e.message}` }]);
    } finally { setRunning(false); }
  }

  const installedModels = ocrModels.filter(m => m.installed).map(m => ({ value: m.id, label: m.label }));
  const total = gtImages.length;
  const n = Math.max(1, Math.min(numImages || total, total));
  const selected = order.slice(0, n).filter(Boolean);
  const recBy = appr => Object.fromEntries((detail?.[appr] || []).map(r => [r.image, r]));

  return (
    <div>
      <Tabs tabs={[{id:"full",label:"Full image"},{id:"crop",label:"Cropped patches"},{id:"compare",label:"⇄ Compare"}]}
        active={view} onChange={setView} />

      {/* ── Run + results for a single approach ─────────────────────────────── */}
      {view !== "compare" && (
        <>
          <Panel title={`Run · ${APPROACHES[view]}`}>
            <FormRow label="OCR model">
              <Select value={cfg.ocr_model} onChange={v => setCfg(c=>({...c,ocr_model:v}))}
                options={installedModels.length > 0 ? installedModels : [{value:"easyocr",label:"EasyOCR"}]} />
            </FormRow>
            {view === "crop" && (
              <>
                <FormRow label="Task" hint="Which class to crop">
                  <Select value={cfg.task} onChange={v => setCfg(c=>({...c,task:v}))}
                    options={[{value:"both",label:"Both (tables + dimensions)"},{value:"tables",label:"Tables"},{value:"dimensions",label:"Dimensions"}]} />
                </FormRow>
                <FormRow label="Detector" hint="Uses the best trained model automatically">
                  <div className="text-sm text-gray-500 pt-1.5">Best detector for the task (auto)</div>
                </FormRow>
                <FormRow label="Conf threshold" hint={`Min detection confidence. Current: ${cfg.conf_threshold}`}>
                  <input type="range" min="0.1" max="0.9" step="0.05" value={cfg.conf_threshold}
                    onChange={e => setCfg(c=>({...c,conf_threshold:parseFloat(e.target.value)}))}
                    className="w-full accent-gray-900" />
                </FormRow>
              </>
            )}
            <div className="mt-2"><Btn primary onClick={run} loading={running} disabled={running}>▶ Run OCR ({APPROACHES[view]})</Btn></div>
            <p className="text-xs text-gray-400 mt-2">
              {view === "full"
                ? "Reads the whole drawing, then scores against all your annotated text (word + character coverage)."
                : "The best detector crops regions (each crop 2× upscaled + sharpened), OCRs them, then scores against your annotated text. Task Both = crop everything."}
            </p>
          </Panel>

          {(running || logs.length > 0) && <Panel title="Live logs"><JobLog logs={logs} /></Panel>}

          {compare?.[view]?.evaluated_images ? (
            <Panel title="Metrics" badge={`${compare[view].evaluated_images} images scored`}>
              <div className="grid grid-cols-3 gap-3 mb-3">
                <MetricCard label="Word coverage" value={cov(compare[view])} good={compare[view].word_coverage > 0.7} sub="of annotated words found" />
                <MetricCard label="Char coverage" value={`${(compare[view].char_coverage*100).toFixed(0)}%`} good={compare[view].char_coverage > 0.7} sub="of annotated characters" />
                <MetricCard label="Word precision" value={`${(compare[view].word_precision*100).toFixed(0)}%`} good={compare[view].word_precision > 0.6} sub="of OCR words that were annotated" />
              </div>
              <p className="text-xs text-gray-400">
                Whole-text scoring: all annotated text vs all OCR output (block-aware — correct for table paragraphs).
                {" "}{compare[view].matched_words}/{compare[view].n_gt_words} annotated words found · {compare[view].n_pred_words} words read.
              </p>
            </Panel>
          ) : <InfoBox>Run {APPROACHES[view]} above to see metrics (scored against your annotations).</InfoBox>}
        </>
      )}

      {/* ── Compare tab: full vs crop ───────────────────────────────────────── */}
      {view === "compare" && (
        <Panel title="Which approach reads your annotations better?" badge="full vs crop">
          {compare && (compare.full?.evaluated_images || compare.crop?.evaluated_images) ? (() => {
            const rows = [["full", compare.full], ["crop", compare.crop]];
            const best = Math.max(...rows.map(([,c]) => c?.word_coverage ?? -1));
            const pct = v => v == null ? "—" : `${(v*100).toFixed(0)}%`;
            return (
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead><tr className="text-xs text-gray-400 border-b border-gray-100">
                    <th className="text-left font-medium py-2 pr-3">Approach</th>
                    <th className="text-right font-medium py-2 px-3">Word coverage</th>
                    <th className="text-right font-medium py-2 px-3">Char coverage</th>
                    <th className="text-right font-medium py-2 px-3">Word precision</th>
                    <th className="text-right font-medium py-2 pl-3">Words found</th>
                  </tr></thead>
                  <tbody>
                    {rows.map(([key, c]) => {
                      const isBest = c && c.word_coverage === best && c.evaluated_images > 0;
                      return (
                        <tr key={key} className={`border-b border-gray-50 last:border-0 ${isBest ? "bg-emerald-50/60" : ""}`}>
                          <td className="py-2 pr-3 text-gray-700">{APPROACHES[key]} {isBest && <Badge variant="green">best</Badge>}{!c?.evaluated_images && <span className="text-xs text-gray-300 ml-1">not run</span>}</td>
                          <td className="text-right px-3 font-mono text-gray-800">{pct(c?.word_coverage)}</td>
                          <td className="text-right px-3 font-mono text-gray-500">{pct(c?.char_coverage)}</td>
                          <td className="text-right px-3 font-mono text-gray-500">{pct(c?.word_precision)}</td>
                          <td className="text-right pl-3 font-mono text-gray-500">{c ? `${c.matched_words}/${c.n_gt_words}` : "—"}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
                <p className="text-xs text-gray-400 mt-3">
                  Both scored with <strong>whole-text coverage</strong> against all annotated text. <strong>Coverage</strong> = how much of your annotation the approach captured. The <strong>best</strong> row wins on word coverage.
                </p>
              </div>
            );
          })() : <InfoBox>Run both <strong>Full image</strong> and <strong>Cropped patches</strong> (in their tabs) — the comparison appears here.</InfoBox>}
        </Panel>
      )}

      {/* ── Per-image inspector (all tabs) ─────────────────────────────────── */}
      {detail && total > 0 && (
        <Panel title="Per-image results vs ground truth" badge={view === "compare" ? "full vs crop" : APPROACHES[view]}>
          <div className="flex items-center gap-3 mb-3 text-sm text-gray-500 flex-wrap">
            Show
            <input type="number" min={1} max={total} value={numImages}
              onChange={e => setNumImages(Math.max(1, Math.min(total, parseInt(e.target.value) || 1)))}
              className="w-16 border border-gray-200 rounded-lg px-2 py-1 text-center focus:outline-none focus:ring-1 focus:ring-gray-300" />
            random image(s) of {total}
            <Btn small onClick={() => setOrder(shuffle(gtImages))}>↻ shuffle</Btn>
            {n >= total && <Badge variant="gray">showing all</Badge>}
          </div>
          <div className="flex gap-4 mb-4 text-xs text-gray-400">
            <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 rounded bg-emerald-100 border border-emerald-300" />found</span>
            <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 rounded bg-red-100 border border-red-300" />missed</span>
          </div>

          {selected.map(img => {
            const full = recBy("full")[img], crop = recBy("crop")[img];
            const rec  = view === "compare" ? (full || crop) : recBy(view)[img];
            if (view !== "compare" && !rec) return <p key={img} className="text-xs text-gray-300 mb-2">{img} — not run in this approach</p>;
            if (view === "compare" && !full && !crop) return <p key={img} className="text-xs text-gray-300 mb-2">{img} — not run</p>;
            return (
              <div key={img} className="border border-gray-100 rounded-xl mb-3 p-4 flex gap-4">
                <div className="w-56 shrink-0"><Thumb stem={img} onOpen={() => setZoomImg(img)} /></div>
                <div className="flex-1 min-w-0 space-y-3">
                  <p className="font-mono text-xs text-gray-600">{img}</p>
                  {view === "compare" ? (
                    <>
                      <div><p className="text-xs text-gray-500 mb-1">Full image · coverage {cov(full?.metrics)}</p>
                        <Chips tokens={(full||crop).gt_tokens} flags={full ? full.matched_flags : (full||crop).gt_tokens.map(()=>false)} /></div>
                      <div><p className="text-xs text-gray-500 mb-1">Cropped patches · coverage {cov(crop?.metrics)}</p>
                        <Chips tokens={(full||crop).gt_tokens} flags={crop ? crop.matched_flags : (full||crop).gt_tokens.map(()=>false)} /></div>
                    </>
                  ) : (
                    <>
                      <div><p className="text-xs text-gray-500 mb-1">Annotated text · coverage {cov(rec.metrics)} · char {(rec.metrics.char_coverage*100).toFixed(0)}%</p>
                        <Chips tokens={rec.gt_tokens} flags={rec.matched_flags} /></div>
                      <div><p className="text-xs text-gray-500 mb-1">OCR read</p>
                        <p className="text-xs text-gray-600 bg-gray-50 rounded-lg p-2 max-h-24 overflow-y-auto whitespace-pre-wrap">{rec.pred_text || <em className="text-gray-300">nothing read</em>}</p></div>
                      {view === "crop" && <div><p className="text-xs text-gray-500 mb-1">Detected crops (patch quality)</p><CropStrip image={img} model={cfg.ocr_model} /></div>}
                    </>
                  )}
                </div>
              </div>
            );
          })}
        </Panel>
      )}

      {total === 0 && !running && (
        <div className="text-center py-16 text-gray-300 text-sm">
          <div className="text-3xl mb-2">Aa</div>Annotate images and run OCR to see results
        </div>
      )}

      {/* ── Full-screen viewer ──────────────────────────────────────────────── */}
      {zoomImg && (() => {
        const full = recBy("full")[zoomImg], crop = recBy("crop")[zoomImg];
        const rec  = view === "compare" ? (full || crop) : recBy(view)[zoomImg];
        return (
          <div className="fixed inset-0 z-50 bg-black/85 flex" onClick={() => setZoomImg(null)}>
            <div className="flex-1 overflow-auto p-4 flex items-start justify-center" onClick={e => e.stopPropagation()}>
              <img src={`${API_BASE}/annotation/image/${encodeURIComponent(zoomImg)}`} alt={zoomImg}
                onClick={() => setZoomFit(f => !f)}
                className={zoomFit ? "max-w-full max-h-[92vh] object-contain cursor-zoom-in" : "max-w-none cursor-zoom-out"} />
            </div>
            <div className="w-[40rem] shrink-0 bg-white h-full overflow-y-auto p-5" onClick={e => e.stopPropagation()}>
              <div className="flex items-center justify-between mb-1">
                <p className="font-mono text-xs text-gray-600 truncate">{zoomImg}</p>
                <button onClick={() => setZoomImg(null)} className="text-gray-400 hover:text-gray-800 text-xl leading-none ml-2">✕</button>
              </div>
              <p className="text-xs text-gray-400 mb-3">Click the image to toggle <strong>fit</strong> ↔ <strong>actual size</strong> (scroll to pan). Green = OCR found the word, red = missed.</p>
              {view === "compare" ? (
                <div className="space-y-4">
                  <div><p className="text-xs text-gray-500 mb-1">Full image · coverage {cov(full?.metrics)}</p>
                    <Chips tokens={rec.gt_tokens} flags={full ? full.matched_flags : rec.gt_tokens.map(()=>false)} /></div>
                  <div><p className="text-xs text-gray-500 mb-1">Cropped patches · coverage {cov(crop?.metrics)}</p>
                    <Chips tokens={rec.gt_tokens} flags={crop ? crop.matched_flags : rec.gt_tokens.map(()=>false)} /></div>
                </div>
              ) : rec && (
                <div className="space-y-4">
                  <div><p className="text-xs text-gray-500 mb-1">Annotated · coverage {cov(rec.metrics)}</p>
                    <Chips tokens={rec.gt_tokens} flags={rec.matched_flags} /></div>
                  <div><p className="text-xs text-gray-500 mb-1">OCR read</p>
                    <p className="text-xs text-gray-600 bg-gray-50 rounded-lg p-2 whitespace-pre-wrap">{rec.pred_text || "nothing read"}</p></div>
                </div>
              )}
            </div>
          </div>
        );
      })()}
    </div>
  );
}
