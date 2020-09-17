from django.template.loader import render_to_string
from django.http import HttpResponse
from django.shortcuts import redirect


def landing(request, page=""):
    html = render_to_string('index.html')
    response = HttpResponse()
    response.write(html)
    return response


def asset_redirect(request):
    return redirect("/static" + request.path, permanent=True)
