// Shared renderers that turn a saved run SNAPSHOT into its visual dashboard.
import { API_BASE } from "../api/client";

const imgUrl = stem => `${API_BASE}/annotation/image/${encodeURIComponent(stem)}`;

function iou(a, b) {
  const ix1 = Math.max(a[0], b[0]), iy1 = Math.max(a[1], b[1]);
  const ix2 = Math.min(a[2], b[2]), iy2 = Math.min(a[3], b[3]);
  const iw = Math.max(0, ix2 - ix1), ih = Math.max(0, iy2 - iy1);
  const inter = iw * ih;
  const ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter;
  return ua > 0 ? inter / ua : 0;
}
const asBox  = g => Array.isArray(g) ? { box: g, cls: null } : g;
const asPred = p => Array.isArray(p) ? { box: p, score: 1, cls: null } : p;

const VERDICT = {
  correct: "text-emerald-600", wrong: "text-amber-600", missed: "text-gray-400",
  hallucinated: "text-red-500", abstained: "text-gray-400",
};
const pct = v => v == null ? "—" : `${(v * 100).toFixed(0)}%`;
const num = v => v == null ? "—" : (Number.isInteger(v) ? String(v) : v.toFixed(3));

export function runMetricPairs(snap) {
  const m = snap?.metrics || {};
  if (snap?.stage === "detection")
    return [["mAP@0.5", num(m.map50)], ["Precision", num(m.precision)], ["Recall", num(m.recall)],
            ["F1", num(m.f1)], ["Best F1", m.best_f1 != null ? num(m.best_f1) : "—"], ["TP/FP/FN", `${m.tp}/${m.fp}/${m.fn}`]];
  if (snap?.stage === "ocr")
    return [["Word cov.", pct(m.word_coverage)], ["Char cov.", pct(m.char_coverage)],
            ["Word prec.", pct(m.word_precision)], ["CER", m.cer != null ? num(m.cer) : "—"],
            ["Words found", `${m.matched_words}/${m.n_gt_words}`]];
  if (snap?.stage === "vlm") {
    const v = snap?.view || {};
    const prov = v.crop_detector ? `${v.crop_detector} → ${v.ocr_model}`
               : (v.ocr_model ? `${v.ocr_model} (page)` : "image only");
    return [["Field acc.", num(m.field_accuracy)], ["Halluc.", num(m.hallucination_rate)],
            ["Miss", num(m.miss_rate)], ["Exact", num(m.exact_match)], ["Context", prov]];
  }
  return Object.entries(m).map(([k, v]) => [k, num(v)]);
}

export function runImages(snap) {
  const v = snap?.view || {};
  if (snap?.stage === "detection") return (v.images || []).map(im => im.image);
  return (v.detail || []).map(d => d.image);
}

export function RunImageCell({ snap, image, conf = 0.25 }) {
  const v = snap?.view || {};
  if (snap?.stage === "detection") {
    const rec = (v.images || []).find(im => im.image === image);
    if (!rec) return <p className="text-xs text-gray-300">not in this run</p>;
    const W = rec.width || 1000, H = rec.height || 1000;
    const gt = rec.gt.map(asBox);
    const preds = rec.pred.map(asPred).filter(p => p.score >= conf);
    const gtHit = gt.map(g => preds.some(p => (p.cls == null || g.cls == null || p.cls === g.cls) && iou(g.box, p.box) >= 0.5));
    const missed = gtHit.filter(h => !h).length;
    const sw = Math.max(W, H) / 350;
    return (
      <div>
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto rounded-lg border border-gray-100 bg-gray-50">
          <image href={imgUrl(image)} x="0" y="0" width={W} height={H} />
          {preds.map((p, i) => <rect key={"p"+i} x={p.box[0]} y={p.box[1]} width={p.box[2]-p.box[0]} height={p.box[3]-p.box[1]}
            fill="none" stroke={p.cls === "table" ? "#7C3AED" : "#2563EB"} strokeWidth={sw} opacity="0.7" />)}
          {gt.map((g, i) => <rect key={"g"+i} x={g.box[0]} y={g.box[1]} width={g.box[2]-g.box[0]} height={g.box[3]-g.box[1]}
            fill="none" stroke={gtHit[i] ? "#10B981" : "#EF4444"} strokeWidth={sw*1.6} strokeDasharray={gtHit[i] ? "none" : `${sw*3},${sw*2}`} />)}
        </svg>
        <p className="text-[11px] text-gray-400 mt-1">
          <span className="text-blue-600">{preds.length} pred</span> · <span className="text-emerald-600">{gt.length - missed}/{gt.length} found</span>
          {missed > 0 && <span className="text-red-500"> · {missed} missed</span>}
        </p>
      </div>
    );
  }
  if (snap?.stage === "ocr") {
    const rec = (v.detail || []).find(d => d.image === image);
    if (!rec) return <p className="text-xs text-gray-300">not in this run</p>;
    return (
      <div>
        <p className="text-[11px] text-gray-500 mb-1">coverage {pct(rec.metrics?.word_coverage)}</p>
        <div className="flex flex-wrap gap-1">
          {rec.gt_tokens.map((t, i) => (
            <span key={i} className={`px-1.5 py-0.5 rounded text-[11px] ${rec.matched_flags[i] ? "bg-emerald-100 text-emerald-700" : "bg-red-100 text-red-600"}`}>{t}</span>
          ))}
          {!rec.gt_tokens.length && <span className="text-xs text-gray-300">— no annotated text —</span>}
        </div>
      </div>
    );
  }
  if (snap?.stage === "vlm") {
    const rec = (v.detail || []).find(d => d.image === image);
    if (!rec) return <p className="text-xs text-gray-300">not in this run</p>;
    return (
      <table className="w-full text-[11px]">
        <tbody>
          {rec.fields.map(f => (
            <tr key={f.name} className="border-t border-gray-50">
              <td className="py-0.5 pr-2 text-gray-500">{f.name}</td>
              <td className="px-1 text-gray-700">{f.gt == null ? "—" : String(f.gt)}</td>
              <td className={`px-1 font-medium ${VERDICT[f.verdict] || ""}`}>{f.pred == null ? "—" : String(f.pred)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    );
  }
  return null;
}

export function ImgThumb({ image }) {
  return <img src={imgUrl(image)} alt={image} onError={e => { e.currentTarget.style.display = "none"; }}
    className="rounded border border-gray-100 bg-gray-50 w-full object-contain max-h-56" />;
}
