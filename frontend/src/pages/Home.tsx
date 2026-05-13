import { Link } from 'react-router-dom';

export default function Home() {
  return (
    <div className="min-h-screen bg-dark-900 relative overflow-hidden">
      {/* Animated background */}
      <div className="absolute inset-0">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-primary-600/10 rounded-full blur-3xl animate-float" />
        <div className="absolute bottom-1/4 right-1/4 w-80 h-80 bg-purple-600/10 rounded-full blur-3xl animate-float" style={{ animationDelay: '2s' }} />
        <div className="absolute top-1/2 left-1/2 w-64 h-64 bg-cyan-600/5 rounded-full blur-3xl animate-pulse-slow" />
        {/* Grid overlay */}
        <div
          className="absolute inset-0 opacity-[0.03]"
          style={{
            backgroundImage: `linear-gradient(rgba(255,255,255,0.1) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.1) 1px, transparent 1px)`,
            backgroundSize: '60px 60px',
          }}
        />
      </div>

      {/* Hero content */}
      <div className="relative z-10 flex flex-col items-center justify-center min-h-screen px-6 pt-20">
        <div className="text-center max-w-4xl mx-auto">
          {/* Badge */}
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary-500/10 border border-primary-500/20 mb-8">
            <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
            <span className="text-sm text-primary-300 font-medium">AI-Powered Detection Engine</span>
          </div>

          {/* Main headline */}
          <h1 className="text-6xl md:text-8xl font-bold text-white leading-tight tracking-tight">
            Is This{' '}
            <span className="relative inline-block">
              <span className="relative z-10 bg-gradient-to-r from-primary-400 via-purple-400 to-cyan-400 bg-clip-text text-transparent">
                Real?
              </span>
              <span className="absolute -inset-1 bg-gradient-to-r from-primary-500/20 to-purple-500/20 blur-2xl" />
            </span>
          </h1>

          <p className="mt-8 text-xl text-dark-100 max-w-2xl mx-auto leading-relaxed">
            Detect deepfakes with our ensemble of ResNet-18, EfficientNet-B3, and Vision Transformers.
            Upload any image or video — get instant AI-powered analysis with explainable heatmaps.
          </p>

          {/* CTA buttons */}
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4 mt-12">
            <Link
              to="/detect"
              id="cta-analyze-now"
              className="group relative px-8 py-4 rounded-xl bg-gradient-to-r from-primary-600 to-primary-500 text-white font-semibold text-lg shadow-xl shadow-primary-600/25 hover:shadow-primary-500/40 transition-all duration-300 hover:-translate-y-0.5"
            >
              <span className="relative z-10">Analyze Now</span>
              <div className="absolute inset-0 rounded-xl bg-gradient-to-r from-primary-500 to-primary-400 opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
            </Link>
            <a
              href="https://github.com/obstinix/deepfake_recognition"
              target="_blank"
              rel="noopener noreferrer"
              className="px-8 py-4 rounded-xl border border-dark-300 text-dark-50 font-semibold text-lg hover:bg-dark-700/50 hover:border-dark-200 transition-all duration-300"
            >
              View on GitHub
            </a>
          </div>

          {/* Stats */}
          <div className="flex items-center justify-center gap-12 mt-20">
            {[
              { value: '95%+', label: 'Ensemble Accuracy' },
              { value: '3', label: 'Model Ensemble' },
              { value: '<120ms', label: 'Avg Latency' },
            ].map((stat) => (
              <div key={stat.label} className="text-center">
                <p className="text-3xl font-bold text-white">{stat.value}</p>
                <p className="text-sm text-dark-200 mt-1">{stat.label}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
