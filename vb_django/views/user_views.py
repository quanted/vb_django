from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.models import User
from rest_framework.response import Response
from rest_framework.parsers import JSONParser, FormParser, MultiPartParser
from rest_framework import views, status
from rest_framework.authtoken.models import Token
from rest_framework.authtoken.views import ObtainAuthToken
from vb_django.serializers import UserSerializer


class UserLoginView(ObtainAuthToken):
    """
    Custom login authentication view
    """
    def post(self, request, *args, **kwargs):
        """
        Expand default login authentication request to return user details, in addition to the token.
        :param request:
        :param args:
        :param kwargs:
        :return:
        """
        # response = super(UserLoginView, self).post(request, *args, **kwargs)
        serializer = self.serializer_class(data=request.data, context={'request': request})
        if serializer.is_valid():
            username = serializer.validated_data['user']
            user = User.objects.get(username=username)
            old_token = Token.objects.get(user=user)
            old_token.delete()
            token = Token.objects.create(user=user)
            return Response({'id': user.id, 'email': user.email, 'username': user.username, 'token': token.key})
        return Response(None, status=status.HTTP_401_UNAUTHORIZED)


class UserView(views.APIView):
    """
    User registration view for creating a new user in the VB Web database.
    """
    parser_classes = [JSONParser, FormParser, MultiPartParser]

    @csrf_exempt
    def post(self, request):
        """
        Register a new user.
        :param request: Request body should contain 'username', 'email', and 'password'
        :return: The newly created user and a authentication token, or the appropriate error message.
        """
        serializer = UserSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.save()
            if user:
                data = serializer.data
                token = Token.objects.create(user=user)
                data["token"] = token.key
                return Response(data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class UserResetView(views.APIView):
    """
    User reset password view.
    """
    parser_classes = [JSONParser, FormParser, MultiPartParser]

    @csrf_exempt
    def post(self, request):
        """
        Register a new user.
        :param request: Request body should contain 'username', 'email'
        :return: TODO
        """
        return Response(None, status=status.HTTP_501_NOT_IMPLEMENTED)
