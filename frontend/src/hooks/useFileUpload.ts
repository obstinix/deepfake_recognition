import { useState, useCallback } from 'react';
import { submitFile } from '../services/api';
import type { UploadStatus, SubmitResponse } from '../types';

export function useFileUpload() {
  const [status, setStatus] = useState<UploadStatus>('idle');
  const [error, setError] = useState<string | null>(null);
  const [taskId, setTaskId] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);

  const upload = useCallback(async (selectedFile: File) => {
    setFile(selectedFile);
    setError(null);
    setStatus('uploading');

    try {
      const response: SubmitResponse = await submitFile(selectedFile);
      setTaskId(response.task_id);
      setStatus('processing');
      return response.task_id;
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Upload failed';
      setError(message);
      setStatus('error');
      return null;
    }
  }, []);

  const reset = useCallback(() => {
    setStatus('idle');
    setError(null);
    setTaskId(null);
    setFile(null);
  }, []);

  return { status, error, taskId, file, upload, reset };
}
