#!/bin/bash

if [ ! -d "/opt/app/vb_django/vb_django/static" ]
then
  mkdir /opt/app/vb_django/static
fi
python /opt/app/vb_django/manage.py collectstatic --noinput
python /opt/app/vb_django/manage.py migrate --noinput
python /opt/app/vb_django/manage.py migrate auth --noinput
python /opt/app/vb_django/manage.py migrate sessions --noinput
python /opt/app/vb_django/manage.py runserver 0.0.0.0:8080
# exec uwsgi /etc/uwsgi/uwsgi.ini --show-config