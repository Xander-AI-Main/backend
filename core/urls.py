from django.urls import path,include
from rest_framework.routers import DefaultRouter
from .views import DatasetUploadView, TrainModelView, LoginView
from . import views

router=DefaultRouter()

router.register('signup/',views.signupViewset, basename='signup'),

urlpatterns=[
    path('signup/', include(router.urls)),
    path('login/', LoginView.as_view(), name='login'),
    path('upload/', DatasetUploadView.as_view(), name='dataset-upload'),
    path('train/', TrainModelView.as_view(), name='train_model'),
]
