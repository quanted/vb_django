# Generated by Django 3.2.8 on 2021-12-08 19:25

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('vb_django', '0002_auto_20210301_1044'),
    ]

    operations = [
        migrations.CreateModel(
            name='PipelineLog',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('logtype', models.CharField(max_length=24)),
                ('log', models.CharField(max_length=512)),
                ('timestamp', models.CharField(max_length=56)),
                ('parent', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='vb_django.pipeline')),
            ],
        ),
    ]