from celery import Celery

celery_app = Celery(
    "omnidoc",
    broker = "redis://localhost:6379/0",
    backend = "redis://localhost:6379/0",
    include=["app.tasks.document_task"]
)
