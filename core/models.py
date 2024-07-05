from django.db import models
from django.contrib.auth.models import AbstractUser
from django_countries.fields import CountryField
import pycountry


class userSignup (AbstractUser):
    phone_number = models.CharField(max_length=15, blank=False, null=False)
    country = CountryField(blank_label='(select country)',blank=False,null=False)
    currency = models.CharField(max_length=3, blank=True, null=True)
    workin_in_team=models.BooleanField(default=False)
    # s3_storage_used=models.models.FloatField(default=0.0)
    cpu_hours_used=models.IntegerField(default=0,blank=False, null=False)
    gpu_hours_used=models.IntegerField(default=0,blank=False, null=False)
    dataset_url=models.JSONField(default=list)
    trained_model_url=models.JSONField(default=list)
    plan=models.TextField()
    max_storage_allowed=models.BigIntegerField(default=0,blank=False, null=False)
    max_cpu_hours_allowed=models.IntegerField(default=0,blank=False, null=False) 
    max_gpu_hours_allowed=models.IntegerField(default=0,blank=False, null=False) 
    team=models.JSONField(default=list)
    
    
    def save(self, *args, **kwargs):
        if self.country:
            country = pycountry.countries.get(alpha_2=self.country.code)
            if country:
                currency = pycountry.currencies.get(numeric=country.numeric)
                self.currency = currency.alpha_3 if currency else None
        super().save(*args, **kwargs)
        


class Dataset(models.Model):
    name = models.CharField(max_length=255)
    size_gb = models.FloatField()
    task_type = models.CharField(max_length=255)
    architecture_details = models.TextField()
    uploaded_at = models.DateTimeField(auto_now_add=True)

class S3StorageUsage(models.Model):
    used_gb = models.FloatField(default=0)
