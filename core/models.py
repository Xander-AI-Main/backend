from django.db import models
from django.contrib.auth.models import AbstractUser
from django_countries.fields import CountryField
import pycountry
from decimal import Decimal
import json
from datetime import datetime, timedelta
from django.utils import timezone

current_date = datetime.now()
future_date = current_date + timedelta(days=30)

class userSignup (AbstractUser):
    phone_number = models.CharField(max_length=15, blank=True, null=False)
    country = CountryField(blank_label='(select country)',blank=True,null=False)
    currency = models.CharField(max_length=3, blank=True, null=True)
    workin_in_team=models.BooleanField(default=False)
    s3_storage_used=models.FloatField(default=0.0)
    cpu_hours_used=models.FloatField(default=0.0)
    gpu_hours_used=models.FloatField(default=0.0)
    dataset_url=models.JSONField(default=list)
    trained_model_url=models.JSONField(default=list)
    plan=models.TextField(default='', blank=True, null=False)
    photo=models.TextField(default='', blank=True, null=False)
    max_storage_allowed=models.BigIntegerField(default=0,blank=False, null=False)
    max_cpu_hours_allowed=models.IntegerField(default=0,blank=False, null=False) 
    max_gpu_hours_allowed=models.IntegerField(default=0,blank=False, null=False) 
    team=models.JSONField(default=list)
    dummy=models.BooleanField(default=False)
    purchase_date = models.DateTimeField(default=timezone.now)
    expired_date = models.DateTimeField(default=lambda: timezone.now() + timezone.timedelta(days=30))
    has_expired = models.BooleanField(default=False)

    def save(self, *args, **kwargs):
        if self.country:
            country = pycountry.countries.get(alpha_2=self.country.code)
            if country:
                currency = pycountry.currencies.get(numeric=country.numeric)
                self.currency = currency.alpha_3 if currency else None

        if isinstance(self.dataset_url, str):
            self.dataset_url = json.loads(self.dataset_url)
        if isinstance(self.trained_model_url, str):
            self.trained_model_url = json.loads(self.trained_model_url)
        if isinstance(self.team, str):
            self.team = json.loads(self.team)

        super().save(*args, **kwargs)
        
class Dataset(models.Model):
    name = models.CharField(max_length=255)
    size_gb = models.FloatField(default=0.0)
    task_type = models.CharField(max_length=255)
    architecture_details = models.TextField()
    uploaded_at = models.DateTimeField(auto_now_add=True)
    hyperparameter=models.JSONField(default=list)

# class S3StorageUsage(models.Model):
#     used_gb = models.DecimalField(default=0)
