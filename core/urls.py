from django.urls import path,include
from rest_framework.routers import DefaultRouter
from .views import DatasetUploadView
from . import views

router=DefaultRouter()

router.register('studentfees/',views.signupViewset, basename='signup'),

urlpatterns=[
    path('signup/',include(router.urls)),
    path('upload/', DatasetUploadView.as_view(), name='dataset-upload'),
    # path('zipped/', ZippedFileUploadView.as_view(), name='zippedfile-upload')
]