import { useState, useEffect } from "react";
import { api, pollJob, API_BASE } from "../api/client";
import { Panel, FormRow, Select, Btn, JobLog, Badge, InfoBox, RadioGroup, Collapsible, MetricCard } from "../components/ui";

const MODES = [
  { value:"whole_image",     label:"Image only",        hint:"Full drawing image, no text context — the baseline" },
  { value:"whole_image_ocr", label:"Image + page OCR",  hint:"Full image + OCR text read off the whole page" },
  { value:"cropped_ocr",     label:"Image + crop OCR",  hint:"Full image + detector-crop OCR text with box coordinates" },
];
const MODE_KEYS = MODES.map(m => m.value);
const modeLabel = v => MODES.find(m => m.value === v)?.label || v;

// Per-field verdict → colour.
const VERDICT = {
  correct:      { text: "text-emerald-600", dot: "bg-emerald-500", label: "correct" },
  wrong:        { text: "text-amber-600",   dot: "bg-amber-500",   label: "wrong" },
  missed:       { text: "text-gray-400",    dot: "bg-gray-300",    label: "missed" },
  hallucinated: { text: "text-red-500",     dot: "bg-red-500",     label: "hallucinated" },
  abstained:    { text: "text-gray-400",    dot: "bg-gray-200",    label: "blank ✓" },
};
const showVal = v => (v === null || v === undefined || v === "") ? "—" : Array.isArray(v) ? v.join(", ") : String(v);
function shuffle(arr) {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
  return a;
}

const DEFAULT_PROMPTS = {
  whole_image:     "You are analyzing a 2D mechanical engineering drawing.\nExtract the manufacturing features and return ONLY a valid JSON object, no preamble, no markdown.\nAnalyze the provided image directly.",
  whole_image_ocr: "You are analyzing a 2D mechanical engineering drawing.\nYou are also given OCR-extracted text as a hint — the image is the primary source of truth.\nReturn ONLY a valid JSON object.",
  cropped_ocr:     "You are analyzing cropped patches from a 2D mechanical engineering drawing.\nCombine all patches to fill the feature schema. Return ONLY a valid JSON object.",
};

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

function MetricLine({ m }) {
  if (!m) return <p className="text-xs text-gray-300">not run</p>;
  const c = m.counts || {};
  return (
    <p className="text-xs text-gray-400 mt-1 leading-relaxed">
      acc <strong className="text-gray-600">{m.field_accuracy != null ? m.field_accuracy.toFixed(2) : "—"}</strong>
      {m.hallucination_rate != null && <> · halluc <strong className="text-gray-600">{m.hallucination_rate.toFixed(2)}</strong></>}
      {" · "}<span className="text-emerald-600">{c.correct||0} ok</span>
      {" · "}<span className="text-amber-600">{c.wrong||0} wrong</span>
      {" · "}<span className="text-gray-500">{c.missed||0} missed</span>
      {" · "}<span className="text-red-500">{c.hallucinated||0} halluc</span>
    </p>
  );
}

function SingleImageCard({ rec, onZoom }) {
  return (
    <div className="border border-gray-100 rounded-xl mb-3 p-4 flex gap-4">
      <div className="w-56 shrink-0"><Thumb stem={rec.image} onOpen={() => onZoom(rec.image)} /></div>
      <div className="flex-1 min-w-0 overflow-x-auto">
        <p className="font-mono text-xs text-gray-600 mb-2">{rec.image}</p>
        <table className="w-full text-xs">
          <thead><tr className="text-gray-400">
            <th className="text-left py-1 pr-2">Field</th>
            <th className="text-left px-2">Ground truth</th>
            <th className="text-left px-2">Extracted</th>
            <th></th>
          </tr></thead>
          <tbody>
            {rec.fields.map(f => (
              <tr key={f.name} className="border-t border-gray-50">
                <td className="py-1 pr-2 text-gray-600">{f.name}</td>
                <td className="px-2 text-gray-800">{showVal(f.gt)}</td>
                <td className={`px-2 font-medium ${VERDICT[f.verdict]?.text}`}>{showVal(f.pred)}</td>
                <td className="text-right pl-2"><span className={`inline-block w-2 h-2 rounded-full ${VERDICT[f.verdict]?.dot}`} title={VERDICT[f.verdict]?.label} /></td>
              </tr>
            ))}
          </tbody>
        </table>
        <MetricLine m={rec.metrics} />
      </div>
    </div>
  );
}

function CompareImageCard({ stem, perMode, onZoom }) {
  const anyRec = MODE_KEYS.map(k => perMode[k]).find(Boolean);
  if (!anyRec) return null;
  const fieldNames = anyRec.fields.map(f => f.name);
  const gtMap = Object.fromEntries(anyRec.fields.map(f => [f.name, f.gt]));
  const predOf = (mode, name) => perMode[mode]?.fields.find(f => f.name === name);
  return (
    <div className="border border-gray-100 rounded-xl mb-3 p-4">
      <div className="flex items-center gap-4 mb-3">
        <div className="w-48 shrink-0"><Thumb stem={stem} onOpen={() => onZoom(stem)} /></div>
        <p className="font-mono text-xs text-gray-600">{stem}</p>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead><tr className="text-gray-400">
            <th className="text-left py-1 pr-2">Field</th>
            <th className="text-left px-2">Ground truth</th>
            {MODE_KEYS.map(k => <th key={k} className="text-left px-2 whitespace-nowrap">{modeLabel(k)}</th>)}
          </tr></thead>
          <tbody>
            {fieldNames.map(name => (
              <tr key={name} className="border-t border-gray-50">
                <td className="py-1 pr-2 text-gray-600">{name}</td>
                <td className="px-2 text-gray-800">{showVal(gtMap[name])}</td>
                {MODE_KEYS.map(k => { const f = predOf(k, name); return (
                  <td key={k} className={`px-2 font-medium ${f ? VERDICT[f.verdict]?.text : "text-gray-300"}`}>{f ? showVal(f.pred) : "—"}</td>
                ); })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="grid grid-cols-3 gap-3 mt-3">
        {MODE_KEYS.map(k => (
          <div key={k}><p className="text-xs font-medium text-gray-500">{modeLabel(k)}</p><MetricLine m={perMode[k]?.metrics} /></div>
        ))}
      </div>
    </div>
  );
}

export default function VLMPage() {
  const [cfg, setCfg]         = useState({ vlm_model:"claude", mode:"whole_image", task:"both", ocr_model:"easyocr" });
  const [results, setResults] = useState([]);
  const [metrics, setMetrics] = useState(null);
  const [logs, setLogs]       = useState([]);
  const [running, setRunning] = useState(false);
  const [prompt, setPrompt]   = useState(DEFAULT_PROMPTS.whole_image);
  const [promptSaved, setPromptSaved] = useState(false);
  const [compare, setCompare] = useState(null);
  const [mView, setMView]     = useState("whole_image");  // metrics filter — independent of run config
  const [detail, setDetail]   = useState(null);           // {mode: [{image, fields, metrics}]}
  const [gtImages, setGtImages] = useState([]);
  const [numImages, setNumImages] = useState(0);
  const [order, setOrder]     = useState([]);
  const [zoomImg, setZoomImg] = useState(null);   // full-screen image viewer
  const [zoomFit, setZoomFit] = useState(true);   // fit-to-screen vs actual size

  const needsOCR = cfg.mode === "whole_image_ocr" || cfg.mode === "cropped_ocr";

  useEffect(() => {
    api.getVLMResults().then(setResults).catch(() => setResults([]));
  }, []);

  useEffect(() => {
    if (mView !== "compare") api.getVLMMetrics(null, mView).then(setMetrics).catch(() => setMetrics(null));
    api.getVLMCompare().then(setCompare).catch(() => setCompare(null));
    api.getVLMDetail().then(d => { setDetail(d.detail); setGtImages(d.gt_images || []); }).catch(() => { setDetail(null); setGtImages([]); });
  }, [mView, results]);

  // Reset the random ordering/count whenever the evaluated image set changes.
  useEffect(() => {
    setOrder(shuffle(gtImages));
    setNumImages(gtImages.length);
  }, [gtImages.join(",")]);

  // Sync prompt when mode changes
  useEffect(() => {
    api.getPrompt(cfg.mode)
      .then(r => setPrompt(r.text?.trim() ? r.text : DEFAULT_PROMPTS[cfg.mode]))
      .catch(() => setPrompt(DEFAULT_PROMPTS[cfg.mode]));
    setPromptSaved(false);
  }, [cfg.mode]);

  async function run(allModes = false) {
    setRunning(true); setLogs([]);
    try {
      const modes = allModes ? MODES.map(m => m.value) : [cfg.mode];
      const { job_id } = await api.runVLM({ ...cfg, modes });
      await pollJob(job_id, setLogs);
      const fresh = await api.getVLMResults();
      setResults(fresh);
    } catch (e) {
      setLogs(l => [...l, { ts: new Date().toISOString(), msg: `Error: ${e.message}` }]);
    } finally { setRunning(false); }
  }

  async function savePrompt() {
    await api.updatePrompt({ mode: cfg.mode, text: prompt });
    setPromptSaved(true);
    setTimeout(() => setPromptSaved(false), 2000);
  }

  return (
    <div>
      <Panel title="Config">
        <FormRow label="VLM model">
          <input value={cfg.vlm_model} onChange={e => setCfg(c=>({...c,vlm_model:e.target.value}))}
            placeholder="claude-sonnet-4-6"
            className="w-full text-sm border border-gray-200 rounded-lg px-3 py-1.5 font-mono focus:outline-none focus:ring-1 focus:ring-gray-300" />
        </FormRow>
        <FormRow label="Input mode">
          <RadioGroup name="vlm-mode" options={MODES} value={cfg.mode} onChange={v => setCfg(c=>({...c,mode:v}))} />
        </FormRow>
        {/* Task only matters for crop context — it selects which detector crops feed the VLM.
            Whole-image modes read the entire page and fill the unified schema. */}
        {cfg.mode === "cropped_ocr" && (
          <FormRow label="Crop source" hint="Which detector crops to feed as context">
            <Select value={cfg.task} onChange={v => setCfg(c=>({...c,task:v}))}
              options={[{value:"both",label:"Both (tables + dimensions)"},{value:"tables",label:"Tables only"},{value:"dimensions",label:"Dimensions only"}]} />
          </FormRow>
        )}
        {needsOCR && (
          <FormRow label="OCR source" hint="Which OCR run to use as text context">
            <Select value={cfg.ocr_model} onChange={v => setCfg(c=>({...c,ocr_model:v}))} options={["easyocr","tesseract"]} />
          </FormRow>
        )}
        <div className="mt-3 flex flex-wrap gap-2 items-center">
          <Btn primary onClick={() => run(false)} loading={running} disabled={running}>▶ Run this mode</Btn>
          <Btn onClick={() => run(true)} disabled={running}>▶▶ Run all 3 modes &amp; compare</Btn>
          <span className="text-xs text-gray-400">Running all 3 makes 3 API calls per image.</span>
        </div>
      </Panel>

      {/* Prompt editor — title reacts to selected mode */}
      <Collapsible title={`Edit prompt · ${MODES.find(m=>m.value===cfg.mode)?.label}`}>
        <p className="text-xs text-gray-400 mb-3">
          The feature schema is appended automatically — you only need to write the instructions here.
        </p>
        <textarea value={prompt} onChange={e => setPrompt(e.target.value)}
          className="w-full h-36 text-sm font-mono border border-gray-200 rounded-lg p-3 resize-y focus:outline-none focus:ring-1 focus:ring-gray-300 text-gray-800" />
        <div className="flex gap-2 mt-2">
          <Btn primary small onClick={savePrompt}>{promptSaved ? "✓ Saved" : "Save"}</Btn>
          <Btn small onClick={() => { setPrompt(DEFAULT_PROMPTS[cfg.mode]); api.resetPrompt(cfg.mode); }}>Reset to default</Btn>
        </div>
      </Collapsible>

      {(running || logs.length > 0) && <Panel title="Live logs"><JobLog logs={logs} /></Panel>}

      {/* ── Evaluation metrics — own mode/compare filter (independent of run config) ── */}
      <Panel title="Evaluation metrics">
        <div className="flex flex-wrap gap-1.5 mb-4">
          {MODES.map(m => (
            <button key={m.value} onClick={() => setMView(m.value)}
              className={`text-xs px-3 py-1.5 rounded-lg border ${mView===m.value ? "bg-gray-900 text-white border-gray-900" : "bg-white text-gray-600 border-gray-200 hover:border-gray-300"}`}>
              {m.label}
            </button>
          ))}
          <button onClick={() => setMView("compare")}
            className={`text-xs px-3 py-1.5 rounded-lg border ${mView==="compare" ? "bg-gray-900 text-white border-gray-900" : "bg-white text-gray-600 border-gray-200 hover:border-gray-300"}`}>
            ⇄ Compare all 3
          </button>
        </div>

        {mView === "compare" ? (
          compare?.modes?.some(m => m.evaluated_images > 0 || m.resources?.n_calls > 0) ? (() => {
            const scored = compare.modes.filter(m => m.evaluated_images > 0 && m.field_accuracy != null);
            const bestAcc = scored.length ? Math.max(...scored.map(m => m.field_accuracy)) : null;
            const fmt = (v, s="") => v == null ? "—" : `${v}${s}`;
            return (
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-xs text-gray-400 border-b border-gray-100">
                      <th className="text-left font-medium py-2 pr-3">Input mode</th>
                      <th className="text-right font-medium py-2 px-3">Field acc.</th>
                      <th className="text-right font-medium py-2 px-3">Halluc.</th>
                      <th className="text-right font-medium py-2 px-3">Miss</th>
                      <th className="text-right font-medium py-2 px-3">Num %off</th>
                      <th className="text-right font-medium py-2 px-3">Latency</th>
                      <th className="text-right font-medium py-2 px-3">Tok (in/out)</th>
                      <th className="text-right font-medium py-2 pl-3">Cost</th>
                    </tr>
                  </thead>
                  <tbody>
                    {compare.modes.map(m => {
                      const r = m.resources || {};
                      const isBest = bestAcc != null && m.field_accuracy === bestAcc && m.evaluated_images > 0;
                      return (
                        <tr key={m.mode} className={`border-b border-gray-50 last:border-0 ${isBest ? "bg-emerald-50/60" : ""}`}>
                          <td className="py-2 pr-3 text-gray-700">
                            {modeLabel(m.mode)} {isBest && <Badge variant="green">best</Badge>}
                            {m.evaluated_images === 0 && <span className="text-xs text-gray-300 ml-1">not run</span>}
                          </td>
                          <td className="text-right px-3 font-mono text-gray-800">{fmt(m.field_accuracy?.toFixed(3))}</td>
                          <td className="text-right px-3 font-mono text-gray-500">{fmt(m.hallucination_rate?.toFixed(3))}</td>
                          <td className="text-right px-3 font-mono text-gray-500">{fmt(m.miss_rate?.toFixed(3))}</td>
                          <td className="text-right px-3 font-mono text-gray-500">{m.numeric_mape != null ? `${(m.numeric_mape*100).toFixed(1)}%` : "—"}</td>
                          <td className="text-right px-3 font-mono text-gray-500">{fmt(r.avg_latency_s, "s")}</td>
                          <td className="text-right px-3 font-mono text-gray-500">{fmt(r.avg_input_tokens)}/{fmt(r.avg_output_tokens)}</td>
                          <td className="text-right pl-3 font-mono text-gray-500">{r.total_cost_usd != null ? `$${r.total_cost_usd}` : "—"}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
                <p className="text-xs text-gray-400 mt-3">
                  All modes scored against the same unified answer key. The <strong>best</strong> row is the highest field accuracy — weigh it against hallucination rate and cost.
                </p>
              </div>
            );
          })() : <InfoBox>Run the modes (▶▶ Run all 3 modes above) to compare them here.</InfoBox>
        ) : metrics?.available && metrics?.evaluated_images ? (
          <>
            <p className="text-xs text-gray-400 mb-3">{modeLabel(mView)} · {metrics.evaluated_images} image(s) scored</p>
            <div className="grid grid-cols-4 gap-3 mb-3">
              <MetricCard label="Field accuracy"  value={metrics.field_accuracy?.toFixed(3)}     good={metrics.field_accuracy > 0.7}     sub="correct / fields with a value" />
              <MetricCard label="Error rate"      value={metrics.error_rate?.toFixed(3)}         good={metrics.error_rate < 0.15}        sub="wrong values · lower better" />
              <MetricCard label="Miss rate"       value={metrics.miss_rate?.toFixed(3)}          good={metrics.miss_rate < 0.15}         sub="left blank · lower better" />
              <MetricCard label="Hallucination"   value={metrics.hallucination_rate?.toFixed(3)} good={metrics.hallucination_rate < 0.1}  sub="invented on empty · lower better" />
            </div>
            <div className="grid grid-cols-3 gap-3 mb-4">
              <MetricCard label="Overall accuracy" value={metrics.overall_accuracy?.toFixed(3)} good={metrics.overall_accuracy > 0.7} sub="right or correctly blank · all 15" />
              <MetricCard label="Numeric closeness" value={metrics.numeric_mape != null ? `${(metrics.numeric_mape*100).toFixed(1)}%` : "—"} good={metrics.numeric_mape != null && metrics.numeric_mape < 0.05} sub={`mean % off · ${metrics.numeric_scored||0} numeric fields`} />
              <MetricCard label="Exact match"      value={metrics.exact_match?.toFixed(3)}      good={metrics.exact_match > 0.5} sub="images with 0 mistakes" />
            </div>
            {metrics.counts && (
              <p className="text-xs text-gray-400 mb-4">
                Across {metrics.evaluated_images} image(s):
                <strong className="text-emerald-600"> {metrics.counts.correct} correct</strong> ·
                <strong className="text-amber-600"> {metrics.counts.wrong} wrong</strong> ·
                <strong className="text-gray-500"> {metrics.counts.missed} missed</strong> ·
                <strong className="text-red-500"> {metrics.counts.hallucinated} hallucinated</strong> ·
                {metrics.counts.abstained} correctly blank
                {" "}({metrics.counts.present_fields} fields had a value, {metrics.counts.absent_fields} were empty)
              </p>
            )}
            {metrics.resources && metrics.resources.n_calls > 0 && (
              <div className="grid grid-cols-4 gap-3 mb-4">
                <MetricCard label="Avg latency"   value={metrics.resources.avg_latency_s != null ? `${metrics.resources.avg_latency_s}s` : "—"} sub="per image" />
                <MetricCard label="Avg in tokens" value={metrics.resources.avg_input_tokens ?? "—"}  sub="prompt + image" />
                <MetricCard label="Avg out tokens"value={metrics.resources.avg_output_tokens ?? "—"} sub="response" />
                <MetricCard label="Total cost"    value={metrics.resources.total_cost_usd != null ? `$${metrics.resources.total_cost_usd}` : "—"} sub={`${metrics.resources.n_calls} calls`} />
              </div>
            )}
            {metrics.per_field && Object.keys(metrics.per_field).length > 0 && (
              <div>
                <p className="text-xs text-gray-400 mb-2">Per-field accuracy</p>
                {Object.entries(metrics.per_field).map(([k,v]) => (
                  <div key={k} className="flex items-center gap-3 py-1.5 text-sm">
                    <span className="w-40 shrink-0 text-gray-600">{k}</span>
                    <div className="flex-1 h-1.5 bg-gray-100 rounded-full overflow-hidden">
                      <div className={`h-full rounded-full ${v > 0.8 ? "bg-emerald-500" : v > 0.5 ? "bg-amber-400" : "bg-red-400"}`} style={{width:`${v*100}%`}} />
                    </div>
                    <span className="w-12 text-right font-mono text-xs text-gray-700">{v.toFixed(2)}</span>
                  </div>
                ))}
              </div>
            )}
          </>
        ) : metrics?.available ? (
          <InfoBox>
            Ground truth is loaded{metrics?.gt_images ? ` for ${metrics.gt_images} annotated image(s)` : ""}, but <strong>{modeLabel(mView)}</strong> hasn't been run yet. Set the input mode to {modeLabel(mView)} in the config above and click <strong>Run this mode</strong> (or <strong>Run all 3 modes</strong>) — accuracy appears here automatically.
          </InfoBox>
        ) : (
          <InfoBox>
            No ground truth yet. Annotate images in the <strong>Annotate</strong> tab (their feature values become the answer key at <code className="bg-white/60 px-1 rounded">vlm/data/ground_truth/unified.json</code>). One unified answer key scores every input mode. Then reopen this tab to see accuracy — no re-run needed.
          </InfoBox>
        )}
      </Panel>

      {/* ── Per-image inspector: extracted vs ground truth, side by side ────── */}
      {detail && gtImages.length > 0 && (() => {
        const total = gtImages.length;
        const n = Math.max(1, Math.min(numImages || total, total));
        const selected = order.slice(0, n).filter(Boolean);
        // Build image → {mode: rec} for the compare view.
        const byImage = {};
        MODE_KEYS.forEach(k => (detail[k] || []).forEach(rec => { (byImage[rec.image] ??= {})[k] = rec; }));
        const singleByImage = Object.fromEntries((detail[mView] || []).map(r => [r.image, r]));
        const anyForMode = mView !== "compare" && (detail[mView] || []).length > 0;

        return (
          <Panel title="Per-image results vs ground truth"
            badge={mView === "compare" ? "all 3 modes" : modeLabel(mView)}>
            <div className="flex items-center gap-3 mb-3 text-sm text-gray-500 flex-wrap">
              Show
              <input type="number" min={1} max={total} value={numImages}
                onChange={e => setNumImages(Math.max(1, Math.min(total, parseInt(e.target.value) || 1)))}
                className="w-16 border border-gray-200 rounded-lg px-2 py-1 text-center focus:outline-none focus:ring-1 focus:ring-gray-300" />
              random image(s) of {total}
              <Btn small onClick={() => setOrder(shuffle(gtImages))}>↻ shuffle</Btn>
              {n >= total && <Badge variant="gray">showing all</Badge>}
            </div>
            {/* verdict legend */}
            <div className="flex flex-wrap gap-3 mb-4 text-xs text-gray-400">
              {Object.values(VERDICT).map(v => (
                <span key={v.label} className="flex items-center gap-1"><span className={`w-2 h-2 rounded-full ${v.dot}`} />{v.label}</span>
              ))}
            </div>

            {mView === "compare"
              ? selected.map(img => <CompareImageCard key={img} stem={img} perMode={byImage[img] || {}} onZoom={setZoomImg} />)
              : anyForMode
                ? selected.map(img => singleByImage[img]
                    ? <SingleImageCard key={img} rec={singleByImage[img]} onZoom={setZoomImg} />
                    : <p key={img} className="text-xs text-gray-300 mb-2">{img} — not run in this mode</p>)
                : <InfoBox>{modeLabel(mView)} hasn't been run yet — no per-image results to show.</InfoBox>}
          </Panel>
        );
      })()}

      {results.length === 0 && !running && gtImages.length === 0 && (
        <div className="text-center py-16 text-gray-300 text-sm">
          <div className="text-3xl mb-2">◈</div>Run VLM to extract features
        </div>
      )}

      {/* ── Full-screen image viewer: big drawing + values table beside it ──── */}
      {zoomImg && (() => {
        const rec = (detail?.[mView] || []).find(r => r.image === zoomImg);
        const perMode = {};
        MODE_KEYS.forEach(k => { const r = (detail?.[k] || []).find(x => x.image === zoomImg); if (r) perMode[k] = r; });
        const anyRec = rec || MODE_KEYS.map(k => perMode[k]).find(Boolean);
        const fieldNames = anyRec ? anyRec.fields.map(f => f.name) : [];
        const gtMap = anyRec ? Object.fromEntries(anyRec.fields.map(f => [f.name, f.gt])) : {};
        return (
          <div className="fixed inset-0 z-50 bg-black/85 flex" onClick={() => setZoomImg(null)}>
            <div className="flex-1 overflow-auto p-4 flex items-start justify-center" onClick={e => e.stopPropagation()}>
              <img src={`${API_BASE}/annotation/image/${encodeURIComponent(zoomImg)}`} alt={zoomImg}
                onClick={() => setZoomFit(f => !f)}
                className={zoomFit ? "max-w-full max-h-[92vh] object-contain cursor-zoom-in" : "max-w-none cursor-zoom-out"} />
            </div>
            <div className={`${mView === "compare" ? "w-[44rem]" : "w-[28rem]"} shrink-0 bg-white h-full overflow-y-auto p-5`} onClick={e => e.stopPropagation()}>
              <div className="flex items-center justify-between mb-1">
                <p className="font-mono text-xs text-gray-600 truncate">{zoomImg}</p>
                <button onClick={() => setZoomImg(null)} className="text-gray-400 hover:text-gray-800 text-xl leading-none ml-2">✕</button>
              </div>
              <p className="text-xs text-gray-400 mb-3">Click the image to toggle <strong>fit</strong> ↔ <strong>actual size</strong> (then scroll to pan). Compare each value against what you can read on the drawing.</p>
              <table className="w-full text-xs">
                <thead><tr className="text-gray-400">
                  <th className="text-left py-1 pr-2">Field</th>
                  <th className="text-left px-2">GT</th>
                  {mView === "compare"
                    ? MODE_KEYS.map(k => <th key={k} className="text-left px-1">{modeLabel(k).replace("Image ", "").replace("Image", "img")}</th>)
                    : <th className="text-left px-2">Extracted</th>}
                </tr></thead>
                <tbody>
                  {fieldNames.map(name => (
                    <tr key={name} className="border-t border-gray-50">
                      <td className="py-1 pr-2 text-gray-600">{name}</td>
                      <td className="px-2 text-gray-800">{showVal(gtMap[name])}</td>
                      {mView === "compare"
                        ? MODE_KEYS.map(k => { const f = perMode[k]?.fields.find(x => x.name === name); return <td key={k} className={`px-1 font-medium ${f ? VERDICT[f.verdict]?.text : "text-gray-300"}`}>{f ? showVal(f.pred) : "—"}</td>; })
                        : (() => { const f = rec?.fields.find(x => x.name === name); return <td className={`px-2 font-medium ${f ? VERDICT[f.verdict]?.text : ""}`}>{f ? showVal(f.pred) : "—"}</td>; })()}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        );
      })()}
    </div>
  );
}