# Generated by Django 4.2 on 2024-08-09 13:14

import datetime
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0002_alter_usersignup_expired_date'),
    ]

    operations = [
        migrations.AddField(
            model_name='usersignup',
            name='dummy',
            field=models.BooleanField(default=False),
        ),
        migrations.AlterField(
            model_name='usersignup',
            name='expired_date',
            field=models.DateTimeField(default=datetime.datetime(2024, 9, 8, 18, 44, 31, 566451)),
        ),
    ]