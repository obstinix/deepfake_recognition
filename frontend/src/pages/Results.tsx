import { useParams, Link } from 'react-router-dom';
import { usePolling } from '../hooks/usePolling';
import ResultsDisplay from '../components/ResultsDisplay';

export default function Results() {
  const { taskId } = useParams<{ taskId: string }>();
  const { data } = usePolling(taskId || null);

  return (
    <div className="min-h-screen bg-dark-900 pt-24 pb-12 px-6">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-3xl font-bold text-white">Analysis Results</h1>
            <p className="text-dark-200 text-sm mt-1 font-mono">Task: {taskId}</p>
          </div>
          <Link
            to="/detect"
            className="px-5 py-2.5 rounded-xl bg-dark-600 text-white text-sm font-medium hover:bg-dark-500 transition-colors"
          >
            New Analysis
          </Link>
        </div>

        {/* Results */}
        {data ? (
          <ResultsDisplay data={data} />
        ) : (
          <div className="flex flex-col items-center justify-center py-20 gap-6">
            <div className="w-16 h-16 border-4 border-primary-500/20 border-t-primary-500 rounded-full animate-spin" />
            <p className="text-dark-100">Loading task data...</p>
          </div>
        )}
      </div>
    </div>
  );
}
