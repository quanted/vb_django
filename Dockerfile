FROM python:3.9.7-slim

ARG APP_USER=appuser
RUN groupadd -r ${APP_USER} && useradd --no-log-init -r -g ${APP_USER} ${APP_USER}

RUN apt-get update
RUN apt-get install -y python3-pip libpq-dev python-dev
RUN python -m pip install --upgrade pip setuptools wheel

COPY . /opt/app/

RUN pip install -r /opt/app/requirements.txt
RUN pip install uwsgi

COPY vb_django/uwsgi.ini /etc/uwsgi/
RUN chown -R www-data:www-data /opt/app

WORKDIR /opt/app
ENV PYTHONPATH="/opt/app:/opt/app/vb_django:${PYTHONPATH}"
ENV NPATH="/opt/app:/opt/app/vb_django:${PATH}"
USER ${APP_USER}:${APP_USER}
#CMD ["sh", "/tmp/start-django-server.sh"]