import { useState, useEffect } from "react";
import { api } from "../api/client";
import { Panel, Badge, Tabs, InfoBox, Btn } from "../components/ui";
import { RunImageCell, runMetricPairs, runImages, ImgThumb } from "../components/runViews";

const STAGE_METRIC_LABEL = { detection: "mAP@0.5", ocr: "Coverage", vlm: "Fill rate" };
const ALL_RUNS   = "__all_runs__";
const TASK_LABEL = { tables: "Tables", dimensions: "Dimensions", both: "Tables + Dimensions", all: "Whole image" };
const taskLabel  = t => t === ALL_RUNS ? "All runs" : (TASK_LABEL[t] || t);

function getMetricValue(run, stage) {
  if (!run || !run.metrics) return 0;
  const key = stage === "detection" ? "map50" : stage === "ocr" ? "coverage" : "fill_rate";
  return run.metrics[key] ?? 0;
}
function fmtMetric(v) {
  if (v == null) return "—";
  if (typeof v !== "number") return String(v);
  return Number.isInteger(v) ? String(v) : v.toFixed(3);
}
const runTime = ts => new Date(ts).toLocaleString(undefined, { month:"short", day:"numeric", hour:"2-digit", minute:"2-digit" });

export default function ResultsPage() {
  const [stage, setStage]     = useState("detection");
  const [task, setTask]       = useState(ALL_RUNS);
  const [allRuns, setAllRuns] = useState([]);
  const [summary, setSummary] = useState(null);

  const [selected, setSelected] = useState(() => new Set());  // run ids ticked for compare
  const [openSnap, setOpenSnap] = useState(null);             // single run being viewed
  const [compareSnaps, setCompareSnaps] = useState(null);     // [snapshot] for side-by-side
  const [conf, setConf] = useState(0.25);

  useEffect(() => { load(); }, [stage]);
  useEffect(() => { setTask(ALL_RUNS); setSelected(new Set()); setOpenSnap(null); setCompareSnaps(null); }, [stage]);

  async function load() {
    try { setAllRuns(await api.getAllResults(stage)); setSummary(await api.getSummary()); } catch {}
  }
  async function remove(id) { await api.deleteRun(id); setSelected(s => { const n = new Set(s); n.delete(id); return n; }); load(); }

  function toggle(id) {
    setSelected(s => { const n = new Set(s); n.has(id) ? n.delete(id) : n.add(id); return n; });
  }
  async function openSingle(id) {
    setCompareSnaps(null);
    const snap = await api.getRunSnapshot(id);
    setOpenSnap(snap && snap.stage ? snap : { _empty: true });
  }
  async function runCompare() {
    setOpenSnap(null);
    const snaps = await Promise.all([...selected].map(id => api.getRunSnapshot(id).catch(() => null)));
    setCompareSnaps(snaps.filter(s => s && s.stage));
  }

  const taskOptions = [ALL_RUNS, "all", "tables", "dimensions", "both"];
  const runs   = task === ALL_RUNS ? allRuns : allRuns.filter(r => r.task === task);
  const sorted = [...runs].sort((a,b) => getMetricValue(b,stage) - getMetricValue(a,stage));
  const anyDetection = compareSnaps?.some(s => s.stage === "detection") || openSnap?.stage === "detection";

  return (
    <div>
      <InfoBox>
        Every evaluation run is logged here. Tick runs to <strong>compare</strong> them side by side, or open one to see its full dashboard again.
      </InfoBox>

      {summary && (
        <Panel title="Best model per stage">
          <div className="grid grid-cols-3 gap-3">
            {["detection","ocr","vlm"].map(s => {
              const best = summary[s]?.best;
              return (
                <div key={s} className="border border-gray-100 rounded-xl p-4">
                  <p className="text-xs text-gray-400 uppercase tracking-wide mb-2">{s} · {STAGE_METRIC_LABEL[s]}</p>
                  {best ? (
                    <div className="flex items-center justify-between">
                      <div><p className="text-sm font-medium text-gray-700 truncate">{best.model}</p><Badge variant="gray">{best.task}</Badge></div>
                      <p className="text-sm text-emerald-600 font-mono">{getMetricValue(best,s).toFixed(3)}</p>
                    </div>
                  ) : <span className="text-xs text-gray-300">no runs yet</span>}
                  <p className="text-xs text-gray-300 mt-2">{summary[s]?.count || 0} runs · tasks: {(summary[s]?.tasks||[]).join(", ") || "—"}</p>
                </div>
              );
            })}
          </div>
        </Panel>
      )}

      <Tabs tabs={[{id:"detection",label:"Detection"},{id:"ocr",label:"OCR"},{id:"vlm",label:"VLM"}]} active={stage} onChange={setStage} />

      <div className="flex items-center gap-3 mb-4 flex-wrap">
        <span className="text-xs text-gray-400">Task</span>
        <select value={task} onChange={e => setTask(e.target.value)}
          className="text-sm border border-gray-200 rounded-lg px-3 py-1.5 bg-white text-gray-700 focus:outline-none focus:ring-1 focus:ring-gray-300">
          {taskOptions.map(t => <option key={t} value={t}>{taskLabel(t)}</option>)}
        </select>
        <div className="flex-1" />
        {selected.size > 0 && (
          <>
            <span className="text-xs text-gray-400">{selected.size} selected</span>
            <Btn small onClick={runCompare} disabled={selected.size < 2}>⇄ Compare selected</Btn>
            <Btn small onClick={() => setSelected(new Set())}>clear</Btn>
          </>
        )}
      </div>

      <Panel title={`${stage} runs · ${taskLabel(task)}`} badge={sorted.length ? `${sorted.length} runs` : undefined}>
        {sorted.length === 0 && <p className="text-sm text-gray-300 text-center py-8">No runs yet for this stage and task.</p>}
        {sorted.map((r, i) => (
          <div key={r.id} className={`flex items-center gap-3 py-3 border-b border-gray-50 last:border-0 ${i === 0 ? "bg-emerald-50/40 -mx-5 px-5" : ""}`}>
            <input type="checkbox" checked={selected.has(r.id)} onChange={() => toggle(r.id)} className="accent-gray-900" />
            {i === 0 && <Badge variant="green">best</Badge>}
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-gray-700 truncate">{r.model}</p>
              <Badge variant="gray">{taskLabel(r.task)}</Badge>
            </div>
            <div className="flex gap-3 flex-wrap justify-end max-w-[45%]">
              {Object.entries(r.metrics).map(([k,v]) => (
                <div key={k} className="text-right"><p className="text-[10px] text-gray-400">{k}</p><p className="text-sm font-mono text-gray-700">{fmtMetric(v)}</p></div>
              ))}
            </div>
            <span className="text-xs text-gray-300 w-20 text-right shrink-0">{runTime(r.timestamp)}</span>
            <button onClick={() => openSingle(r.id)} className="text-xs text-gray-500 hover:text-gray-900 shrink-0">view</button>
            <button onClick={() => remove(r.id)} className="text-xs text-gray-300 hover:text-red-400 shrink-0">remove</button>
          </div>
        ))}
      </Panel>

      {/* Shared conf slider for detection visuals */}
      {anyDetection && (
        <div className="flex items-center gap-2 mb-3 text-sm text-gray-500">
          Confidence ≥ <strong className="text-gray-700 w-8">{conf.toFixed(2)}</strong>
          <input type="range" min="0.05" max="0.9" step="0.05" value={conf} onChange={e => setConf(parseFloat(e.target.value))} className="w-40 accent-gray-900" />
        </div>
      )}

      {/* Single run dashboard */}
      {openSnap && (openSnap._empty ? (
        <InfoBox>This run has no saved snapshot (it predates run-history). Re-run it to capture the full dashboard.</InfoBox>
      ) : (
        <Panel title={`${openSnap.model}`} badge={`${openSnap.stage} · ${taskLabel(openSnap.task)} · ${runTime(openSnap.timestamp)}`}>
          <div className="flex gap-4 flex-wrap mb-4">
            {runMetricPairs(openSnap).map(([k,v]) => (
              <div key={k}><p className="text-xs text-gray-400">{k}</p><p className="text-sm font-mono text-gray-800">{v}</p></div>
            ))}
            <div className="flex-1" /><Btn small onClick={() => setOpenSnap(null)}>close</Btn>
          </div>
          <div className="grid grid-cols-2 gap-4">
            {runImages(openSnap).map(img => (
              <div key={img} className="border border-gray-100 rounded-xl p-3">
                <p className="font-mono text-xs text-gray-500 mb-2 truncate">{img}</p>
                {openSnap.stage !== "detection" && <div className="mb-2"><ImgThumb image={img} /></div>}
                <RunImageCell snap={openSnap} image={img} conf={conf} />
              </div>
            ))}
          </div>
        </Panel>
      ))}

      {/* Side-by-side comparison */}
      {compareSnaps && compareSnaps.length > 0 && (() => {
        const labels = [];
        compareSnaps.forEach(s => runMetricPairs(s).forEach(([k]) => { if (!labels.includes(k)) labels.push(k); }));
        const imgs = [];
        compareSnaps.forEach(s => runImages(s).forEach(im => { if (!imgs.includes(im)) imgs.push(im); }));
        const pairMap = s => Object.fromEntries(runMetricPairs(s));
        return (
          <Panel title="Comparison" badge={`${compareSnaps.length} runs`}>
            <div className="flex justify-end mb-2"><Btn small onClick={() => setCompareSnaps(null)}>close</Btn></div>
            <div className="overflow-x-auto">
              <table className="text-sm border-collapse">
                <thead>
                  <tr>
                    <th className="text-left text-xs text-gray-400 font-medium p-2 sticky left-0 bg-white">Metric</th>
                    {compareSnaps.map((s, i) => (
                      <th key={i} className="text-left text-xs font-medium p-2 min-w-[180px]">
                        <p className="text-gray-700 truncate">{s.model}</p>
                        <p className="text-gray-300 font-normal">{taskLabel(s.task)} · {runTime(s.timestamp)}</p>
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {labels.map(lbl => (
                    <tr key={lbl} className="border-t border-gray-50">
                      <td className="text-gray-500 p-2 sticky left-0 bg-white">{lbl}</td>
                      {compareSnaps.map((s, i) => <td key={i} className="p-2 font-mono text-gray-800">{pairMap(s)[lbl] ?? "—"}</td>)}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <p className="text-xs text-gray-400 mt-4 mb-2">Per-image (same image across the selected runs)</p>
            <div className="space-y-4">
              {imgs.map(img => (
                <div key={img} className="border border-gray-100 rounded-xl p-3">
                  <p className="font-mono text-xs text-gray-500 mb-2 truncate">{img}</p>
                  <div className="grid gap-3" style={{ gridTemplateColumns: `160px repeat(${compareSnaps.length}, minmax(0,1fr))` }}>
                    <div><ImgThumb image={img} /></div>
                    {compareSnaps.map((s, i) => (
                      <div key={i}>
                        <p className="text-[11px] text-gray-400 mb-1 truncate">{s.model}</p>
                        <RunImageCell snap={s} image={img} conf={conf} />
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </Panel>
        );
      })()}
    </div>
  );
}
