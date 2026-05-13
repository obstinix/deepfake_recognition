import axios from 'axios';
import type { SubmitResponse, TaskResponse, ModelInfo } from '../types';

const api = axios.create({
  baseURL: '/api/v1',
  timeout: 30000,
});

export async function submitFile(file: File): Promise<SubmitResponse> {
  const formData = new FormData();
  formData.append('file', file);
  const { data } = await api.post<SubmitResponse>('/analyze', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return data;
}

export async function getTaskResult(taskId: string): Promise<TaskResponse> {
  const { data } = await api.get<TaskResponse>(`/analyze/${taskId}`);
  return data;
}

export async function getModels(): Promise<ModelInfo[]> {
  const { data } = await api.get<{ models: ModelInfo[] }>('/models');
  return data.models;
}

export async function checkHealth(): Promise<{ status: string; version: string }> {
  const { data } = await api.get('/health');
  return data;
}

export default api;
