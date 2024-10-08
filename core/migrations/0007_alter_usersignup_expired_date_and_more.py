# Generated by Django 4.2 on 2024-09-18 18:51

import core.models
from django.db import migrations, models
import django.utils.timezone


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0006_alter_usersignup_expired_date_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='usersignup',
            name='expired_date',
            field=models.DateTimeField(default=core.models.default_expired_date),
        ),
        migrations.AlterField(
            model_name='usersignup',
            name='purchase_date',
            field=models.DateTimeField(default=django.utils.timezone.now),
        ),
    ]
