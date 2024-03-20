"""vb_django URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include, re_path
from rest_framework import permissions, routers
from drf_yasg.views import get_schema_view
from drf_yasg import openapi
from .landing import landing, asset_redirect
from vb_django.views.user_views import UserView, UserLoginView, UserResetView
from vb_django.views.locations_views import LocationView
from vb_django.views.project_views import ProjectView
from vb_django.views.pipeline_views import PipelineView
from vb_django.views.dataset_views import DatasetView
from vb_django.views.utilities_views import pipeline_details
from vb_django.db_setup import load_pipelines


router = routers.SimpleRouter()
# ---------- Location API endpoints ---------- #
router.register('vb/api/location', LocationView, basename='location')
# --------- Project API endpoints ---------- #
router.register('vb/api/project', ProjectView, basename='project')
# ------ Pipeline API endpoints ------ #
router.register('vb/api/pipeline', PipelineView, basename='pipeline')
# --------- Dataset API endpoints ---------- #
router.register('vb/api/dataset', DatasetView, basename='dataset')

# api_patterns = path(include()


schema_view = get_schema_view(
    openapi.Info(
        title="Virtual Beach Web API",
        description="Open API documentation for the Virtual Beach REST Web API",
        default_version="0.0.1",
    ),
    patterns=router.urls,
    public=True,
    permission_classes=(permissions.AllowAny,)
)

urlpatterns = [
    path('', landing),
    re_path(r'^assets/*', asset_redirect),      # additional statics files
    path('admin/', admin.site.urls),

    # ----------- Swagger Docs/UI ------------- #
    re_path(r'^openapi(?P<format>\.json|\.yaml)$', schema_view.without_ui(cache_timeout=0), name='schema-json'),
    re_path(r'^openapi/$', schema_view.with_ui('swagger', cache_timeout=0), name='schema-swagger-ui'),

    # ---------- User API endpoints ----------- #
    path('api/user/login/', UserLoginView.as_view()),                           # POST - Login
    path('api/user/register/', UserView.as_view()),                             # POST - Register
    path('api/user/reset/', UserResetView.as_view()),                           # POST - Password reset

    # ------ ADD the DRF urls registered to the router ------ #
    # path('api/', include(router.urls)),

    path('info/pipelines/', pipeline_details),
]

urlpatterns = [path('vb/', include(urlpatterns))] + router.urls

load_pipelines(purge=True)

