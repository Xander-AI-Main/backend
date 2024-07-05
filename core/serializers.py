from .models import userSignup,Dataset
from rest_framework import serializers

class signupSerializer(serializers.ModelSerializer):
    country = serializers.SerializerMethodField()
    class Meta:
        model = userSignup
        fields='__all__'
        
        
    def get_country(self, obj):
        return str(obj.country) if obj.country else None


class DatasetUploadSerializer(serializers.Serializer):
    file = serializers.FileField()

class DatasetSerializer(serializers.ModelSerializer):
    class Meta:
        model = Dataset
        fields = '__all__'
        

