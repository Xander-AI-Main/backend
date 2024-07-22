from .models import userSignup,Dataset
from rest_framework import serializers
from django_countries.serializers import CountryFieldMixin
import json
from django.contrib.auth import authenticate, get_user_model
from django.contrib.auth.hashers import make_password
from bson import ObjectId

User = get_user_model()

class LoginSerializer(serializers.Serializer):
    username_or_email = serializers.CharField()
    password = serializers.CharField()

    def validate(self, data):
        username_or_email = data.get('username_or_email')
        password = data.get('password')

        if '@' in username_or_email:
            try:
                user = User.objects.get(email=username_or_email)
            except User.DoesNotExist:
                raise serializers.ValidationError("No user found with this email address.")
            
            if user.check_password(password):
                return user
        else:
            user = authenticate(username=username_or_email, password=password)
            if user:
                return user

        raise serializers.ValidationError("Incorrect credentials")
    
class ObjectIdField(serializers.Field):
    def to_representation(self, value):
        if isinstance(value, ObjectId):
            return str(value)
        return value

class signupSerializer(CountryFieldMixin, serializers.ModelSerializer):
    _id = ObjectIdField(read_only=True)
    dataset_url = serializers.ListField(child=serializers.URLField(), default=list)
    trained_model_url = serializers.ListField(child=serializers.URLField(), default=list)
    team = serializers.ListField(child=serializers.JSONField(), default=list)
    password = serializers.CharField(write_only=True)

    class Meta:
        model = userSignup
        fields = ('_id', 'username', 'email', 'password', 'phone_number', 'country', 
                  'currency', 'workin_in_team', 's3_storage_used', 'cpu_hours_used', 
                  'gpu_hours_used', 'dataset_url', 'trained_model_url', 'plan', 
                  'max_storage_allowed', 'max_cpu_hours_allowed', 'max_gpu_hours_allowed', 
                  'team')
        extra_kwargs = {
            'password': {'write_only': True},
        }

    def create(self, validated_data):
        validated_data['password'] = make_password(validated_data.get('password'))
        return super(signupSerializer, self).create(validated_data)

    def to_representation(self, instance):
        ret = super().to_representation(instance)
        for field in ['dataset_url', 'trained_model_url', 'team']:
            ret[field] = self.parse_json_field(getattr(instance, field))
        return ret

    @staticmethod
    def parse_json_field(value):
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return value if isinstance(value, list) else []
    

class DatasetUploadSerializer(serializers.Serializer):
    file = serializers.FileField()

class DatasetSerializer(serializers.ModelSerializer):
    class Meta:
        model = Dataset
        fields = '__all__'
        
class TaskSerializer(serializers.Serializer):
    dataset_url = serializers.URLField()
    hasChanged = serializers.BooleanField()
    task = serializers.ChoiceField(choices=['regression'])
    mainType = serializers.ChoiceField(choices=['DL', 'ML'])
    archType = serializers.CharField()
    arch_data = serializers.JSONField(required=False)

class ResultSerializer(serializers.Serializer):
    model_obj = serializers.JSONField()
        

