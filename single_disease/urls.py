from django.urls import include, path
from rest_framework.routers import DefaultRouter

from .views import SingleDiseaseReportViewSet, SingleDiseaseValueDomainViewSet

router = DefaultRouter()
router.register(r"single-disease-report", SingleDiseaseReportViewSet, basename="single-disease-report")
router.register(r"single-disease-value-domain", SingleDiseaseValueDomainViewSet, basename="single-disease-value-domain")

urlpatterns = [
    path("", include(router.urls)),
]