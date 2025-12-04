# Observability Stack

This directory contains the Docker Compose setup for the observability stack: OpenTelemetry Collector, Prometheus, Tempo, and Grafana.

## Services

- **OpenTelemetry Collector**: Receives OTLP traces and metrics, forwards to Prometheus and Tempo
- **Prometheus**: Time-series database for metrics
- **Tempo**: Distributed tracing backend
- **Grafana**: Visualization platform for metrics and traces

## Ports

- **Prometheus**: `http://localhost:9090`
- **Grafana**: `http://localhost:3000` (admin/admin)
- **Tempo**: `http://localhost:3200`
- **OTel Collector OTLP gRPC**: `localhost:4317`
- **OTel Collector OTLP HTTP**: `localhost:4318`

## How to use 

1. Start all services:
   ```bash
   cd observability
   docker-compose up -d
   ```
2. Stop all services:
   ```bash
   docker-compose down
   ```