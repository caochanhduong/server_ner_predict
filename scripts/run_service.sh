env PYTHONUNBUFFERED=true gunicorn \
    --workers 2 \
    --max-requests 10000 \
    --timeout 300 \
    --access-logfile - \
    app.run_app:ner_app -b 0.0.0.0:5000
