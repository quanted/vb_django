FROM python:3.9.7-slim

ARG APP_USER=appuser
RUN groupadd -r ${APP_USER} && useradd --no-log-init -r -g ${APP_USER} ${APP_USER}

RUN apt-get update
RUN apt-get install -y python-pip libpq-dev python-dev
RUN python -m pip install --upgrade pip setuptools wheel

RUN mkdir -p /opt/app

COPY requirements.txt /opt/app/requirements.txt
RUN pip install -r /opt/app/requirements.txt
RUN pip install uwsgi==2.0.19.1

COPY vb_django/uwsgi.ini /etc/uwsgi/
COPY start-server.sh /tmp/start-django-server.sh
RUN chown -R www-data:www-data /opt/app
RUN chmod 777 /tmp/start-django-server.sh && \
    chown www-data:www-data /tmp/start-django-server.sh

WORKDIR /opt/app
ENV PYTHONPATH="/opt/app:/opt/app/vb_django:${PYTHONPATH}"
USER ${APP_USER}:${APP_USER}
#CMD ["sh", "/tmp/start-django-server.sh"]