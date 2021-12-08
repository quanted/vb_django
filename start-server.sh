#!/bin/bash

python /opt/app/vb_django/manage.py collectstatic --noinput
python /opt/app/vb_django/manage.py migrate --noinput
python /opt/app/vb_django/manage.py auth --noinput          # used for login
python /opt/app/vb_django/manage.py --noinput      # used for login
exec uwsgi /etc/uwsgi/uwsgi.ini
