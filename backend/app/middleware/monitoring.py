import time
from fastapi import Request
from prometheus_client import Counter, Histogram, make_asgi_app

REQUEST_COUNT = Counter("http_requests_total", "Total HTTP requests", ["method", "path", "status"])
REQUEST_LATENCY = Histogram("http_request_duration_seconds", "HTTP request latency", ["method", "path"])


async def add_monitoring(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    import re
    path = request.url.path
    path = re.sub(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", ":id", path)
    
    REQUEST_COUNT.labels(request.method, path, response.status_code).inc()
    REQUEST_LATENCY.labels(request.method, path).observe(duration)
    return response


metrics_app = make_asgi_app()
