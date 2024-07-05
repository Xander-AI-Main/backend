# Generated by Django 5.0.6 on 2024-07-05 08:46

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0005_delete_dataset'),
    ]

    operations = [
        migrations.CreateModel(
            name='Dataset',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=255)),
                ('size_gb', models.FloatField()),
                ('task_type', models.CharField(max_length=255)),
                ('architecture_details', models.TextField()),
                ('uploaded_at', models.DateTimeField(auto_now_add=True)),
            ],
        ),
        migrations.RenameModel(
            old_name='S3Storage',
            new_name='S3StorageUsage',
        ),
        migrations.DeleteModel(
            name='ZippedFile',
        ),
        migrations.RenameField(
            model_name='s3storageusage',
            old_name='total_storage_gb',
            new_name='used_gb',
        ),
    ]
