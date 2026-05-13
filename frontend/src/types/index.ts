export interface AnalysisResult {
  verdict: 'real' | 'fake';
  confidence: number;
  confidence_real: number;
  confidence_fake: number;
  heatmap_data?: string;
  frame_analysis?: FrameAnalysis[];
  processing_time_ms: number;
  models_used: string[];
}

export interface FrameAnalysis {
  frame_num: number;
  confidence: number;
  verdict: string;
}

export interface TaskResponse {
  task_id: string;
  filename: string;
  status: 'processing' | 'completed' | 'failed';
  progress?: number;
  result?: AnalysisResult;
  error?: string;
  created_at?: string;
}

export interface SubmitResponse {
  task_id: string;
  status: string;
  filename: string;
  message: string;
}

export interface ModelInfo {
  name: string;
  version: string;
  accuracy?: number;
  latency_ms?: number;
  is_active: boolean;
}

export type UploadStatus = 'idle' | 'uploading' | 'processing' | 'completed' | 'error';
