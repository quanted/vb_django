#!/bin/bash

python /opt/app/vb_django/manage.py collectstatic --noinput
python /opt/app/vb_django/manage.py migrate
django-admin migrate auth --noinput          # used for login
django-admin migrate sessions --noinput      # used for login
#echo "ls /opt/app"
#ls /opt/app
#echo "ls /opt/app/vb_django"
#ls /opt/app/vb_django
exec uwsgi /etc/uwsgi/uwsgi.ini
