# routing.py
from django.urls import path
from . import consumers

websocket_urlpatterns = [
    path('ws/data/<str:user_id>/', consumers.EpochConsumer.as_asgi()),
]