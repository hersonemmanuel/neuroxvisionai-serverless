# NeuroXVisionAI RunPod Serverless Worker

This worker performs the GPU-intensive 3D MRI segmentation stage.

## Shared storage
Attach the same RunPod Network Volume used by the always-on CPU web/API Pod.
RunPod mounts it at `/runpod-volume` inside Serverless workers.

The CPU host writes jobs under:

`/workspace/neuroxvisionai_data/jobs/<job_id>/...`

The Serverless worker sees the same files at:

`/runpod-volume/neuroxvisionai_data/jobs/<job_id>/...`

## Endpoint configuration
- Type: Queue
- Active/min workers: 0
- Max workers: 1
- GPU: start with a 24 GB class
- Attach the shared Network Volume
- Worker timeout: 900 seconds is a safe initial value

## Request

```json
{
  "input": {
    "mode": "full_study",
    "job_id": "UUID"
  }
}
```

Health test:

```json
{"input":{"mode":"health"}}
```
