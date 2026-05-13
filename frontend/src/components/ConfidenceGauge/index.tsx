interface ConfidenceGaugeProps {
  confidence: number;
  verdict: 'real' | 'fake';
  size?: number;
}

export default function ConfidenceGauge({ confidence, verdict, size = 200 }: ConfidenceGaugeProps) {
  const radius = (size - 20) / 2;
  const circumference = 2 * Math.PI * radius;
  const progress = confidence * circumference;
  const center = size / 2;

  const color = verdict === 'real'
    ? { stroke: '#22c55e', glow: 'rgba(34, 197, 94, 0.3)', text: '#4ade80' }
    : { stroke: '#ef4444', glow: 'rgba(239, 68, 68, 0.3)', text: '#f87171' };

  return (
    <div className="flex flex-col items-center gap-4">
      <svg width={size} height={size} className="transform -rotate-90 drop-shadow-2xl">
        {/* Background circle */}
        <circle
          cx={center}
          cy={center}
          r={radius}
          fill="none"
          stroke="rgba(255,255,255,0.05)"
          strokeWidth="10"
        />
        {/* Progress arc */}
        <circle
          cx={center}
          cy={center}
          r={radius}
          fill="none"
          stroke={color.stroke}
          strokeWidth="10"
          strokeDasharray={circumference}
          strokeDashoffset={circumference - progress}
          strokeLinecap="round"
          style={{
            filter: `drop-shadow(0 0 8px ${color.glow})`,
            transition: 'stroke-dashoffset 1.5s ease-out',
          }}
        />
        {/* Center text */}
        <text
          x={center}
          y={center}
          textAnchor="middle"
          dominantBaseline="central"
          className="transform rotate-90"
          style={{ transformOrigin: 'center', fill: color.text, fontSize: size * 0.18, fontWeight: 700 }}
        >
          {Math.round(confidence * 100)}%
        </text>
      </svg>

      <div className="text-center">
        <span
          className={`inline-block px-4 py-1.5 rounded-full text-sm font-bold uppercase tracking-wider ${
            verdict === 'real'
              ? 'bg-green-500/15 text-green-400 border border-green-500/30'
              : 'bg-red-500/15 text-red-400 border border-red-500/30'
          }`}
        >
          {verdict}
        </span>
      </div>
    </div>
  );
}
