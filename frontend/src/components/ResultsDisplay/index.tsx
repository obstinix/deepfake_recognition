import type { TaskResponse } from '../../types';
import ConfidenceGauge from '../ConfidenceGauge';
import HeatmapViewer from '../HeatmapViewer';

interface ResultsDisplayProps {
  data: TaskResponse;
}

export default function ResultsDisplay({ data }: ResultsDisplayProps) {
  if (data.status === 'processing') {
    return (
      <div className="flex flex-col items-center justify-center py-20 gap-6">
        <div className="relative">
          <div className="w-20 h-20 border-4 border-primary-500/20 border-t-primary-500 rounded-full animate-spin" />
          <div className="absolute inset-0 w-20 h-20 border-4 border-transparent border-b-primary-300/40 rounded-full animate-spin" style={{ animationDirection: 'reverse', animationDuration: '1.5s' }} />
        </div>
        <div className="text-center">
          <p className="text-xl font-semibold text-white">Analyzing...</p>
          <p className="text-sm text-dark-100 mt-2">Our ensemble of models is processing your file</p>
        </div>
      </div>
    );
  }

  if (data.status === 'failed') {
    return (
      <div className="bg-red-500/10 border border-red-500/30 rounded-2xl p-8 text-center">
        <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-red-500/20 flex items-center justify-center">
          <svg className="w-8 h-8 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </div>
        <p className="text-red-400 text-lg font-semibold">Analysis Failed</p>
        <p className="text-dark-100 text-sm mt-2">{data.error || 'An unexpected error occurred'}</p>
      </div>
    );
  }

  const result = data.result;
  
  if (!result || result.confidence === undefined) return null;

  const { verdict, confidence, confidence_real, confidence_fake, heatmap_data, models_used, processing_time_ms } = result;

  return (
    <div className="space-y-8">
      {/* Main Result */}
      <div className="bg-dark-700/50 rounded-2xl border border-dark-400/20 p-8">
        <div className="flex flex-col md:flex-row items-center gap-8">
          <ConfidenceGauge confidence={confidence} verdict={verdict} size={220} />

          <div className="flex-1 space-y-6">
            <div>
              <h3 className="text-2xl font-bold text-white">Analysis Complete</h3>
              <p className="text-dark-100 mt-1">
                {data.filename} • {processing_time_ms}ms
              </p>
            </div>

            {/* Confidence bars */}
            <div className="space-y-3">
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-green-400 font-medium">Real</span>
                  <span className="text-dark-100">{((confidence_real ?? 0) * 100).toFixed(1)}%</span>
                </div>
                <div className="h-2 bg-dark-500 rounded-full overflow-hidden">
                  <div className="h-full bg-gradient-to-r from-green-600 to-green-400 rounded-full transition-all duration-1000" style={{ width: `${(confidence_real ?? 0) * 100}%` }} />
                </div>
              </div>
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-red-400 font-medium">Fake</span>
                  <span className="text-dark-100">{((confidence_fake ?? 0) * 100).toFixed(1)}%</span>
                </div>
                <div className="h-2 bg-dark-500 rounded-full overflow-hidden">
                  <div className="h-full bg-gradient-to-r from-red-600 to-red-400 rounded-full transition-all duration-1000" style={{ width: `${(confidence_fake ?? 0) * 100}%` }} />
                </div>
              </div>
            </div>

            {/* Models used */}
            {models_used && (
              <div className="flex flex-wrap gap-2">
                {models_used.map((model) => (
                  <span key={model} className="px-3 py-1 rounded-full bg-primary-500/10 text-primary-300 text-xs font-medium border border-primary-500/20">
                    {model}
                  </span>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Heatmap */}
      {heatmap_data && (
        <div>
          <h3 className="text-lg font-semibold text-white mb-4">Attention Heatmap</h3>
          <HeatmapViewer heatmapData={heatmap_data} />
        </div>
      )}
    </div>
  );
}
