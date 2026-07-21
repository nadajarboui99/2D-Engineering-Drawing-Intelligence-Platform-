import { useState } from "react";
import DetectionPage from "./pages/Detection";
import OCRPage from "./pages/OCR";
import VLMPage from "./pages/VLM";
import PipelinePage from "./pages/Pipeline";
import ResultsPage from "./pages/Results";
import SettingsPage from "./pages/Settings";
import AnnotatePage from "./pages/Annotate";
import { FeaturesPage, ModelsPage } from "./pages/Other";

const NAV = [
  { section: "Evaluate", items: [
    { id:"detection", label:"Detection",     icon:"⬡" },
    { id:"ocr",       label:"OCR",           icon:"Aa" },
    { id:"vlm",       label:"VLM",           icon:"◈" },
    { id:"pipeline",  label:"Full pipeline", icon:"⟶" },
  ]},
  { section: "Compare", items: [
    { id:"results",   label:"Results",       icon:"▦" },
  ]},
  { section: "Configure", items: [
    { id:"annotate",  label:"Annotate",      icon:"✎" },
    { id:"features",  label:"Features",      icon:"≡" },
    { id:"models",    label:"Add model",     icon:"＋" },
    { id:"settings",  label:"API keys",      icon:"🔑" },
  ]},
];

const PAGES = {
  detection: { title:"Detection",       component:<DetectionPage /> },
  ocr:       { title:"OCR",             component:<OCRPage /> },
  vlm:       { title:"VLM",             component:<VLMPage /> },
  pipeline:  { title:"Full pipeline",   component:<PipelinePage /> },
  results:   { title:"Results",         component:<ResultsPage /> },
  annotate:  { title:"Annotate",        component:<AnnotatePage /> },
  features:  { title:"Feature schema",  component:<FeaturesPage /> },
  models:    { title:"Add model",       component:<ModelsPage /> },
  settings:  { title:"API keys",        component:<SettingsPage /> },
};

export default function App() {
  const [active, setActive] = useState("detection");
  const page = PAGES[active];

  return (
    <div className="flex h-screen bg-gray-50 font-sans text-gray-800 overflow-hidden">
      <aside className="w-52 shrink-0 bg-white border-r border-gray-100 flex flex-col">
        <div className="px-5 py-5 border-b border-gray-100">
          <h1 className="text-sm font-semibold tracking-tight text-gray-900">Eval Platform</h1>
          <p className="text-xs text-gray-400 mt-0.5">Engineering drawings</p>
        </div>
        <nav className="flex-1 overflow-y-auto py-3">
          {NAV.map(group => (
            <div key={group.section} className="mb-4">
              <p className="text-[10px] text-gray-400 uppercase tracking-widest px-4 mb-1">{group.section}</p>
              {group.items.map(item => (
                <button key={item.id} onClick={() => setActive(item.id)}
                  className={`w-full flex items-center gap-2.5 px-4 py-2 text-sm text-left transition
                    ${active === item.id
                      ? "bg-gray-50 text-gray-900 font-medium border-r-2 border-gray-900"
                      : "text-gray-600 hover:text-gray-900 hover:bg-gray-50"}`}>
                  <span className="text-base leading-none">{item.icon}</span>
                  {item.label}
                </button>
              ))}
            </div>
          ))}
        </nav>
        <div className="px-4 py-4 border-t border-gray-100">
          <p className="text-xs text-gray-400">v0.2 · local</p>
        </div>
      </aside>
      <div className="flex-1 flex flex-col min-w-0">
        <header className="bg-white border-b border-gray-100 px-7 py-4 shrink-0">
          <h2 className="text-sm font-medium text-gray-900">{page.title}</h2>
        </header>
        <main className="flex-1 overflow-y-auto px-7 py-6">
          {page.component}
        </main>
      </div>
    </div>
  );
}