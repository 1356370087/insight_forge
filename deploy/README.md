# InsightForge monitoring examples

These files are deployment examples, not a bundled managed monitoring stack.

## Enable metrics

Set the following API environment variables:

```dotenv
OBSERVABILITY_ENABLED=true
PROMETHEUS_ENABLED=true
PROMETHEUS_METRICS_PATH=/metrics
RUNS_DIR_MAX_BYTES=107374182400
```

`RUNS_DIR_MAX_BYTES=0` means unlimited and suppresses the storage quota alert.
The supplied rules and dashboard target the default
`PROMETHEUS_NAMESPACE=open_deep_research`; update their metric prefixes if the
namespace is customized.

## Prometheus scrape job

Add an API scrape job and load the provided rule file:

```yaml
global:
  scrape_interval: 30s

rule_files:
  - /etc/prometheus/rules/insightforge-alerts.yml

scrape_configs:
  - job_name: insightforge
    metrics_path: /metrics
    honor_labels: true
    static_configs:
      - targets: ["api:2024"]
```

Mount `deploy/prometheus/alerts.yml` at the `rule_files` path. Keeping
`honor_labels: true` preserves the service's bounded provider, model, role, and
operation labels. The application deliberately does not expose run IDs, user
IDs, paths, prompts, or URLs as metric labels.

Validate rule changes before deployment:

```bash
promtool check rules deploy/prometheus/alerts.yml
```

## Alert delivery

Prometheus sends firing alerts to Alertmanager. Configure the receiver in
Alertmanager rather than embedding credentials in this repository. A minimal
Prometheus connection looks like this:

```yaml
alerting:
  alertmanagers:
    - static_configs:
        - targets: ["alertmanager:9093"]
```

Route `severity=warning` to the operational on-call channel. Tune `for` windows
and thresholds only after collecting a representative workload baseline.

## Grafana

Import `deploy/grafana/dashboard.json`, then select the Prometheus datasource.
The dashboard covers run throughput and duration, token usage and estimated
cost, prompt-cache hits, research queue depth and wait, rate limits and retries,
durable storage capacity, and observability export failures.

## Runtime topology and maintenance

Run one API worker per deployment instance. API admission counters, SSE
connection limits, and the in-memory run registry are process-local. SQLite uses
WAL with a five-second busy timeout, but trace writes remain best-effort with
multiple workers.

For advanced memory, use an external daily timer or enable the commented
`memory-maintenance` Compose service. Its loop holds one file lock and exits
immediately if another maintenance process already owns it.

Sandbox V7 never injects provider credentials into Worker containers. Build
`sandbox-worker-image`, run
`python -m open_deep_research.sandbox.pin_image`, then set
`SANDBOX_ENABLED=true`, `ENABLE_ASYNC_RESEARCH=true`, an immutable policy image
digest, and a shared `SANDBOX_ROOT_SIGNING_KEY`, then start the Compose
`sandbox` profile. Run `python -m open_deep_research.sandbox.doctor` before
admitting traffic.
