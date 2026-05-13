import { useState, useEffect } from 'react';
import { getModels } from '../services/api';
import type { ModelInfo } from '../types';

export default function Dashboard() {
  const [models, setModels] = useState<ModelInfo[]>([]);

  useEffect(() => {
    getModels().then(setModels).catch(() => {});
  }, []);

  return (
    <div className="min-h-screen bg-dark-900 pt-24 pb-12 px-6">
      <div className="max-w-6xl mx-auto">
        <h1 className="text-3xl font-bold text-white mb-8">Dashboard</h1>

        {/* Stats grid */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-12">
          {[
            { label: 'Total Analyses', value: '—', icon: '📊' },
            { label: 'Avg Confidence', value: '—', icon: '🎯' },
            { label: 'Models Active', value: String(models.length), icon: '🧠' },
            { label: 'Avg Latency', value: '<120ms', icon: '⚡' },
          ].map((stat) => (
            <div key={stat.label} className="bg-dark-700/50 border border-dark-400/20 rounded-2xl p-6">
              <div className="flex items-center gap-3 mb-3">
                <span className="text-2xl">{stat.icon}</span>
                <span className="text-sm text-dark-100">{stat.label}</span>
              </div>
              <p className="text-2xl font-bold text-white">{stat.value}</p>
            </div>
          ))}
        </div>

        {/* Models table */}
        <div className="bg-dark-700/50 border border-dark-400/20 rounded-2xl overflow-hidden">
          <div className="px-6 py-4 border-b border-dark-400/20">
            <h2 className="text-lg font-semibold text-white">Active Models</h2>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-dark-400/10">
                  <th className="text-left px-6 py-3 text-xs font-medium text-dark-200 uppercase tracking-wider">Model</th>
                  <th className="text-left px-6 py-3 text-xs font-medium text-dark-200 uppercase tracking-wider">Version</th>
                  <th className="text-left px-6 py-3 text-xs font-medium text-dark-200 uppercase tracking-wider">Accuracy</th>
                  <th className="text-left px-6 py-3 text-xs font-medium text-dark-200 uppercase tracking-wider">Latency</th>
                  <th className="text-left px-6 py-3 text-xs font-medium text-dark-200 uppercase tracking-wider">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-dark-400/10">
                {models.map((model) => (
                  <tr key={model.name} className="hover:bg-dark-600/30 transition-colors">
                    <td className="px-6 py-4">
                      <span className="text-white font-medium">{model.name}</span>
                    </td>
                    <td className="px-6 py-4">
                      <span className="text-dark-100 font-mono text-sm">{model.version}</span>
                    </td>
                    <td className="px-6 py-4">
                      <span className="text-white">{model.accuracy ? `${(model.accuracy * 100).toFixed(0)}%` : '—'}</span>
                    </td>
                    <td className="px-6 py-4">
                      <span className="text-dark-100">{model.latency_ms ? `${model.latency_ms}ms` : '—'}</span>
                    </td>
                    <td className="px-6 py-4">
                      <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium ${
                        model.is_active
                          ? 'bg-green-500/15 text-green-400'
                          : 'bg-dark-400/20 text-dark-200'
                      }`}>
                        <span className={`w-1.5 h-1.5 rounded-full ${model.is_active ? 'bg-green-400' : 'bg-dark-300'}`} />
                        {model.is_active ? 'Active' : 'Inactive'}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}
