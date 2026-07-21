import { useState, useEffect } from "react";
import { api } from "../api/client";
import { Panel, Btn, InfoBox, Badge } from "../components/ui";

const PROVIDER_DOCS = {
  anthropic: "https://console.anthropic.com/settings/keys",
  openai:    "https://platform.openai.com/api-keys",
  google:    "https://aistudio.google.com/app/apikey",
  mistral:   "https://console.mistral.ai/api-keys/",
};

export default function SettingsPage() {
  const [providers, setProviders] = useState([]);
  const [keys, setKeys]           = useState({});
  const [saved, setSaved]         = useState({});

  useEffect(() => {
    api.getProviders().then(setProviders).catch(() => {});
  }, []);

  async function saveKey(provider) {
    if (!keys[provider]) return;
    await api.setKey({ provider, key: keys[provider] });
    setSaved(s => ({...s, [provider]: true}));
    setTimeout(() => setSaved(s => ({...s, [provider]: false})), 2000);
    const fresh = await api.getProviders();
    setProviders(fresh);
  }

  return (
    <div>
      <InfoBox>
        API keys are stored locally on your machine only — never sent anywhere except the respective provider's API.
        Once saved, they persist across restarts.
      </InfoBox>

      <Panel title="VLM API keys">
        {providers.map(p => (
          <div key={p.id} className="py-4 border-b border-gray-50 last:border-0">
            <div className="flex items-center justify-between mb-2">
              <div>
                <span className="text-sm font-medium text-gray-800">{p.label}</span>
                {p.configured
                  ? <Badge variant="green" className="ml-2">configured</Badge>
                  : <Badge variant="gray"  className="ml-2">not set</Badge>
                }
              </div>
              {PROVIDER_DOCS[p.id] && (
                <a href={PROVIDER_DOCS[p.id]} target="_blank" rel="noreferrer"
                  className="text-xs text-blue-500 hover:underline">Get API key ↗</a>
              )}
            </div>
            <div className="flex gap-2">
              <input
                type="password"
                placeholder={p.configured ? "••••••••••••••••" : "Paste key here"}
                value={keys[p.id] || ""}
                onChange={e => setKeys(k => ({...k, [p.id]: e.target.value}))}
                className="flex-1 text-sm border border-gray-200 rounded-lg px-3 py-1.5 font-mono focus:outline-none focus:ring-1 focus:ring-gray-300"
              />
              <Btn primary onClick={() => saveKey(p.id)} disabled={!keys[p.id]}>
                {saved[p.id] ? "✓ Saved" : "Save"}
              </Btn>
            </div>
            <div className="flex gap-2 mt-2 flex-wrap">
              {p.models?.map(m => (
                <span key={m} className="text-xs px-2 py-0.5 bg-gray-100 text-gray-500 rounded-full font-mono">{m}</span>
              ))}
            </div>
          </div>
        ))}
      </Panel>

      <Panel title="How VLM models work">
        <div className="text-xs text-gray-500 space-y-2">
          <p>This platform uses <strong className="text-gray-700">LiteLLM</strong> — a unified interface that lets you call any VLM with the same code.</p>
          <p>To use a model: save its provider's API key above, then type its model string in the VLM page selector.</p>
          <p className="font-medium text-gray-700 mt-3">Model string examples:</p>
          <div className="bg-gray-50 rounded-lg p-3 font-mono space-y-1">
            <p>claude-sonnet-4-6</p>
            <p>gpt-4o</p>
            <p>gpt-4o-mini</p>
            <p>gemini/gemini-1.5-pro</p>
            <p>mistral/mistral-large-latest</p>
            <p>ollama/llava  <span className="text-gray-400 font-sans">(local, no key needed)</span></p>
          </div>
        </div>
      </Panel>
    </div>
  );
}