import { useNavigate } from 'react-router-dom';
import FileUpload from '../components/FileUpload';
import { useFileUpload } from '../hooks/useFileUpload';

export default function Detect() {
  const navigate = useNavigate();
  const { status, error, upload } = useFileUpload();

  const handleFileSelect = async (file: File) => {
    const taskId = await upload(file);
    if (taskId) {
      navigate(`/results/${taskId}`);
    }
  };

  return (
    <div className="min-h-screen bg-dark-900 pt-24 pb-12 px-6">
      <div className="max-w-3xl mx-auto">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold text-white">Detect Deepfakes</h1>
          <p className="text-dark-100 mt-3 text-lg">
            Upload an image or video to analyze it with our AI ensemble
          </p>
        </div>

        {/* Upload area */}
        <FileUpload
          onFileSelect={handleFileSelect}
          isUploading={status === 'uploading'}
        />

        {/* Error state */}
        {error && (
          <div className="mt-6 p-4 rounded-xl bg-red-500/10 border border-red-500/30 text-red-400 text-sm">
            {error}
          </div>
        )}

        {/* Info cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-12">
          {[
            {
              icon: (
                <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                </svg>
              ),
              title: 'Multi-Model Ensemble',
              description: 'ResNet-18, EfficientNet-B3, and ViT analyze your media simultaneously',
            },
            {
              icon: (
                <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                </svg>
              ),
              title: 'Explainable AI',
              description: 'Grad-CAM heatmaps show exactly where the model detects manipulation',
            },
            {
              icon: (
                <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
              ),
              title: 'Real-Time Results',
              description: 'Get predictions in under 120ms with confidence scores and analysis',
            },
          ].map((card) => (
            <div
              key={card.title}
              className="p-6 rounded-2xl bg-dark-700/30 border border-dark-400/15 hover:border-primary-500/30 transition-all duration-300"
            >
              <div className="w-12 h-12 rounded-xl bg-primary-500/10 flex items-center justify-center text-primary-400 mb-4">
                {card.icon}
              </div>
              <h3 className="text-white font-semibold">{card.title}</h3>
              <p className="text-dark-100 text-sm mt-2">{card.description}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
