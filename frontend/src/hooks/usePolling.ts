import { useState, useEffect, useCallback, useRef } from 'react';
import { getTaskResult } from '../services/api';
import type { TaskResponse } from '../types';

export function usePolling(taskId: string | null, interval: number = 2000) {
  const [data, setData] = useState<TaskResponse | null>(null);
  const [isPolling, setIsPolling] = useState(false);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const stopPolling = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    setIsPolling(false);
  }, []);

  useEffect(() => {
    if (!taskId) return;

    setIsPolling(true);

    const poll = async () => {
      try {
        const result = await getTaskResult(taskId);
        setData(result);

        if (result.status === 'completed' || result.status === 'failed') {
          stopPolling();
        }
      } catch {
        stopPolling();
      }
    };

    poll(); // immediate first poll
    timerRef.current = setInterval(poll, interval);

    return () => stopPolling();
  }, [taskId, interval, stopPolling]);

  return { data, isPolling, stopPolling };
}
