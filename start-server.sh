#!/bin/bash

if [ ! -d "/opt/app/vb_django/static" ]
then
  mkdir /opt/app/vb_django/static
fi
python /opt/app/vb_django/manage.py django-admin collectstatic --noinput
python /opt/app/vb_django/manage.py migrate --noinput /opt/app/vb_django
python /opt/app/vb_django/manage.py migrate auth --noinput /opt/app/vb_django
python /opt/app/vb_django/manage.py migrate sessions --noinput /opt/app/vb_django
python /opt/app/vb_django/manage.py runserver 0.0.0.0:8080 /opt/app/vb_django
# exec uwsgi /etc/uwsgi/uwsgi.ini --show-config