from django.utils import timezone
import datetime
from django.utils.timezone import utc
from rest_framework.authtoken.models import Token
from rest_framework.authentication import TokenAuthentication
from rest_framework import exceptions

EXPIRATION = 24


class ExpiringTokenAuthentication(TokenAuthentication):
    def authenticate_credentials(self, key):

        try:
            token = Token.objects.get(key=key)
        except Token.DoesNotExist:
            raise exceptions.AuthenticationFailed('Invalid token')

        if not token.user.is_active:
            raise exceptions.AuthenticationFailed('User inactive or deleted')

        current_datetime = timezone.now()

        if token.created < current_datetime - datetime.timedelta(hours=EXPIRATION):
            raise exceptions.AuthenticationFailed('Token has expired')

        return (token.user, token)
