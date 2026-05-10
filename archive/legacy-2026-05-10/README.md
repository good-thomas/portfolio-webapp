# Legacy archive

These files were moved out of the active project root on 2026-05-10.

Active production entrypoint:

- `equity-engine-v3.py`

Render start command:

```sh
gunicorn --timeout 120 --workers 2 --threads 4 --worker-class gthread equity-engine-v3:app
```

The archived files are older application, frontend, data, and engine versions kept for reference.
