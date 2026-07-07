# GPU / CUDA запуск в Docker

Проект уже настроен так, чтобы использовать NVIDIA GPU для локальных моделей эмбеддингов и reranker, если CUDA доступна внутри контейнера.

## Требования

1. Windows 11 + Docker Desktop с WSL2 backend.
2. Установленный NVIDIA драйвер.
3. NVIDIA Container Toolkit / GPU support for Docker.

Проверка на хосте:

```powershell
nvidia-smi
```

Проверка GPU внутри Docker:

```powershell
docker run --rm --gpus all nvidia/cuda:12.9.0-base-ubuntu24.04 nvidia-smi
```

Если эта команда показывает вашу видеокарту, Docker умеет передавать GPU в контейнер.

## Запуск

```powershell
docker compose up --build
```

В логах должно быть:

```text
Device: cuda (NVIDIA GeForce RTX 4070 Ti)
```

Если в логах `Device: cpu`, контейнер не видит CUDA. Проверьте `docker run --rm --gpus all ... nvidia-smi`.

## Принудительный выбор устройства

В `.env` можно указать:

```env
DEVICE=auto
```

Возможные значения:

- `auto` — использовать CUDA, если доступна, иначе CPU;
- `cuda` — требовать GPU, при отсутствии CUDA будет предупреждение и fallback на CPU;
- `cpu` — принудительно использовать процессор.

## Проверка через API

Откройте:

```text
http://localhost:8000/config
```

В ответе должно быть:

```json
"device": "cuda"
```
