# Generated by Django 5.0.6 on 2024-07-04 17:03

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0003_zippedfile'),
    ]

    operations = [
        migrations.AlterField(
            model_name='dataset',
            name='file',
            field=models.URLField(max_length=250),
        ),
    ]
