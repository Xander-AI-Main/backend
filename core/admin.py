from django.contrib import admin
from .models import userSignup,S3StorageUsage,Dataset

admin.site.register(userSignup)
admin.site.register(S3StorageUsage)
# admin.site.register(ZippedFile)
admin.site.register(Dataset)