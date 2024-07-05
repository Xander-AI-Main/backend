from .models import userSignup,Dataset
from rest_framework import serializers

class signupSerializer(serializers.ModelSerializer):
    country = serializers.SerializerMethodField()
    class Meta:
        model = userSignup
        fields='__all__'
        
        
    def get_country(self, obj):
        return str(obj.country) if obj.country else None

class FileURLSerializer(serializers.Serializer):
    url = serializers.URLField()

class DatasetUploadSerializer(serializers.Serializer):
    file = serializers.FileField()

class DatasetSerializer(serializers.ModelSerializer):
    class Meta:
        model = Dataset
        fields = '__all__'
        


# class S3StorageSerializer(serializers.ModelSerializer):
#      class Meta:
#          model = S3Storage
#          fields = ['id', 'total_storage_gb']
        

# class ZippedFileSerializer(serializers.ModelSerializer):
#     class Meta:
#         model = ZippedFile
#         fields = ['id', 'file', 'uploaded_at']