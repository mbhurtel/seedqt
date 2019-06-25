from django.urls import path
from . import views

app_name = 'seed_app'

urlpatterns = [
    path('', views.index, name='index'),
    path('seedinfo/', views.seedinfo, name='seedinfo'),
    path('gallery/', views.gallery, name='gallery'),
    path('aboutus/', views.aboutus, name='aboutus'),
]
