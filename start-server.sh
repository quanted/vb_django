#!/bin/bash
python /opt/app/manage.py collectstatic --noinput
python /opt/app/manage.py migrate
django-admin migrate auth --noinput          # used for login
django-admin migrate sessions --noinput      # used for login
exec uwsgi /etc/uwsgi/uwsgi.ini
