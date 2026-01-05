from django.urls import path
from . import views


app_name = 'products'

urlpatterns = [
    path('', views.product_list, name='product_list'),
    path('products/create', views.product_create, name='product_create'),
    path('products/<int:product_id>/edit', views.product_edit, name='product_edit'),
    path('products/<int:product_id>/delete', views.product_delete, name='product_delete'),

]