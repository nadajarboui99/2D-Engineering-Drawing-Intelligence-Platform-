import { useState } from "react";

export function Card({ children, className = "" }) {
  return <div className={`bg-white border border-gray-100 rounded-xl p-5 ${className}`}>{children}</div>;
}

export function MetricCard({ label, value, sub, good }) {
  return (
    <Card>
      <p className="text-xs text-gray-500 mb-1">{label}</p>
      <p className={`text-2xl font-medium ${good === true ? "text-emerald-600" : good === false ? "text-amber-500" : "text-gray-800"}`}>{value ?? "—"}</p>
      {sub && <p className="text-xs text-gray-400 mt-1">{sub}</p>}
    </Card>
  );
}

export function Panel({ title, badge, action, children }) {
  return (
    <div className="border border-gray-100 rounded-xl mb-4 overflow-hidden">
      <div className="flex items-center justify-between px-5 py-3 border-b border-gray-100 bg-white">
        <h3 className="text-sm font-medium text-gray-800">{title}</h3>
        <div className="flex items-center gap-2">
          {badge && <span className="text-xs px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-700 font-medium">{badge}</span>}
          {action}
        </div>
      </div>
      <div className="bg-white px-5 py-4">{children}</div>
    </div>
  );
}

export function FormRow({ label, children, hint }) {
  return (
    <div className="flex items-start gap-3 mb-3">
      <div className="w-32 shrink-0 pt-1.5">
        <label className="text-sm text-gray-600">{label}</label>
        {hint && <p className="text-xs text-gray-400 mt-0.5">{hint}</p>}
      </div>
      <div className="flex-1">{children}</div>
    </div>
  );
}

export function Select({ value, onChange, options, disabled }) {
  return (
    <select value={value} onChange={e => onChange(e.target.value)} disabled={disabled}
      className="w-full text-sm border border-gray-200 rounded-lg px-3 py-1.5 bg-white text-gray-800 focus:outline-none focus:ring-1 focus:ring-gray-300 disabled:opacity-40">
      {options.map(o => <option key={o.value ?? o} value={o.value ?? o}>{o.label ?? o}</option>)}
    </select>
  );
}

export function Input({ value, onChange, type = "text", placeholder, disabled }) {
  return (
    <input type={type} value={value} onChange={e => onChange(type === "number" ? Number(e.target.value) : e.target.value)}
      placeholder={placeholder} disabled={disabled}
      className="w-full text-sm border border-gray-200 rounded-lg px-3 py-1.5 bg-white text-gray-800 focus:outline-none focus:ring-1 focus:ring-gray-300 disabled:opacity-40" />
  );
}

export function Btn({ children, onClick, primary, small, disabled, loading, danger }) {
  return (
    <button onClick={onClick} disabled={disabled || loading}
      className={`inline-flex items-center gap-1.5 rounded-lg border font-medium transition
        ${small ? "text-xs px-3 py-1" : "text-sm px-4 py-1.5"}
        ${danger ? "bg-red-50 text-red-600 border-red-200 hover:bg-red-100" :
          primary ? "bg-gray-900 text-white border-transparent hover:bg-gray-700" :
                    "bg-white text-gray-700 border-gray-200 hover:bg-gray-50"}
        ${disabled || loading ? "opacity-40 cursor-not-allowed" : "cursor-pointer"}`}>
      {loading ? <span className="animate-spin inline-block">⟳</span> : children}
    </button>
  );
}

export function Tabs({ tabs, active, onChange }) {
  return (
    <div className="flex border-b border-gray-100 mb-5">
      {tabs.map(t => (
        <button key={t.id} onClick={() => onChange(t.id)}
          className={`px-4 py-2 text-sm border-b-2 -mb-px transition
            ${active === t.id ? "border-gray-900 text-gray-900 font-medium" : "border-transparent text-gray-500 hover:text-gray-700"}`}>
          {t.label}
        </button>
      ))}
    </div>
  );
}

export function RadioGroup({ options, value, onChange, name = "rg" }) {
  return (
    <div className="flex flex-col gap-2">
      {options.map(o => (
        <label key={o.value} className="flex items-start gap-2.5 cursor-pointer">
          <input type="radio" name={name} value={o.value} checked={value === o.value} onChange={() => onChange(o.value)} className="mt-0.5" />
          <div>
            <span className="text-sm text-gray-700">{o.label}</span>
            {o.hint && <p className="text-xs text-gray-400">{o.hint}</p>}
          </div>
        </label>
      ))}
    </div>
  );
}

export function Toggle({ label, checked, onChange, hint }) {
  return (
    <label className="flex items-center gap-3 cursor-pointer">
      <div onClick={() => onChange(!checked)} className={`w-9 h-5 rounded-full transition relative cursor-pointer ${checked ? "bg-gray-900" : "bg-gray-200"}`}>
        <div className={`absolute top-0.5 w-4 h-4 rounded-full bg-white shadow transition-all ${checked ? "left-4" : "left-0.5"}`} />
      </div>
      <div>
        <span className="text-sm text-gray-700">{label}</span>
        {hint && <p className="text-xs text-gray-400">{hint}</p>}
      </div>
    </label>
  );
}

export function JobLog({ logs = [] }) {
  return (
    <div className="bg-gray-950 rounded-lg p-3 font-mono text-xs text-green-400 max-h-48 overflow-y-auto">
      {logs.length === 0 ? <span className="text-gray-500">Waiting for logs...</span>
        : logs.map((l, i) => <div key={i}>[{l.ts?.slice(11,19)}] {l.msg}</div>)}
    </div>
  );
}

export function Badge({ children, variant = "gray" }) {
  const s = { green:"bg-emerald-50 text-emerald-700", amber:"bg-amber-50 text-amber-700", blue:"bg-blue-50 text-blue-700", red:"bg-red-50 text-red-600", gray:"bg-gray-100 text-gray-500" };
  return <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${s[variant]}`}>{children}</span>;
}

export function ResultBar({ label, value, max = 1 }) {
  const pct = Math.round((value / max) * 100);
  const color = pct > 80 ? "bg-emerald-500" : pct > 60 ? "bg-amber-400" : "bg-red-400";
  return (
    <div className="flex items-center gap-3 py-2 border-b border-gray-50 last:border-0 text-sm">
      <span className="text-gray-600 w-40 shrink-0 truncate">{label}</span>
      <div className="flex-1 h-1.5 bg-gray-100 rounded-full overflow-hidden">
        <div className={`h-full rounded-full ${color}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-gray-700 w-12 text-right font-mono text-xs">{value > 0 ? value.toFixed(3) : "—"}</span>
    </div>
  );
}

export function InfoBox({ children }) {
  return <div className="bg-blue-50 border border-blue-100 rounded-lg px-4 py-3 text-xs text-blue-700 mb-4">{children}</div>;
}

export function Collapsible({ title, children, defaultOpen = false }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="border border-gray-100 rounded-xl mb-4 overflow-hidden">
      <button onClick={() => setOpen(o => !o)} className="w-full flex items-center justify-between px-5 py-3 bg-white text-sm font-medium text-gray-800 hover:bg-gray-50 transition">
        {title}<span className="text-gray-400">{open ? "▴" : "▾"}</span>
      </button>
      {open && <div className="bg-white px-5 py-4 border-t border-gray-100">{children}</div>}
    </div>
  );
}