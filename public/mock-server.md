# Mock Backend

The frontend ships with an Optuna-ready mock backend so the UI can run without any services.

## Available Endpoints

The `ApiClient` exposes the same contract expected from the production trainer service:

| Method | Endpoint                     | Description                                      |
| ------ | ---------------------------- | ------------------------------------------------ |
| POST   | `/api/trainer/init`          | Initializes a trainer session.                   |
| POST   | `/api/trainer/build_backbone`| Validates uploaded backbone code.                |
| POST   | `/api/train/start`           | Starts a mock training loop.                     |
| POST   | `/api/train/stop`            | Stops the mock training loop.                    |
| GET    | `/api/status`                | Returns `{is_running, current_epoch, ...}`.      |
| GET    | `/api/logs`                  | Returns aggregated logs.                         |
| POST   | `/api/logs/clear`            | Clears the log buffer.                           |

## How It Works

When the “Use Mock Backend” toggle is enabled, the `ApiClient` bypasses `fetch` calls and
routes requests to an in-memory simulator:

- Generates synthetic loss curves and epoch counters.
- Emits timestamped log lines.
- Mimics trainer initialization and backbone validation responses.
- Enforces the same payload shape used by the real trainer service.

Disable the toggle to forward requests to your own backend implementation.
