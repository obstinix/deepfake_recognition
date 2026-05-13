interface HeatmapViewerProps {
  heatmapData: string;
}

export default function HeatmapViewer({ heatmapData }: HeatmapViewerProps) {
  return (
    <div className="rounded-2xl overflow-hidden bg-dark-700/50 border border-dark-400/20 p-1">
      <div className="relative group">
        <img
          src={`data:image/png;base64,${heatmapData}`}
          alt="Grad-CAM Heatmap"
          className="w-full rounded-xl"
        />
        <div className="absolute inset-0 bg-gradient-to-t from-dark-900/60 to-transparent rounded-xl opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex items-end">
          <div className="p-4">
            <p className="text-white text-sm font-medium">Grad-CAM Attention Map</p>
            <p className="text-dark-100 text-xs mt-1">Highlights regions the model focused on for its prediction</p>
          </div>
        </div>
      </div>

      {/* Legend */}
      <div className="flex items-center justify-between px-4 py-3">
        <span className="text-xs text-dark-100">Low attention</span>
        <div className="flex-1 mx-3 h-2 rounded-full bg-gradient-to-r from-blue-500 via-green-400 via-yellow-400 to-red-500" />
        <span className="text-xs text-dark-100">High attention</span>
      </div>
    </div>
  );
}
