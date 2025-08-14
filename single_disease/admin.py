from django.contrib import admin

from .models import Single_disease_report, SingleDiseaseValueDomain


@admin.register(Single_disease_report)
class SingleDiseaseReportAdmin(admin.ModelAdmin):
    list_display = (
        "month",
        "organization_code",
        "organization_name",
        "disease_code",
        "disease_name",
        "case_count",
        "average_length_of_stay_days",
        "average_cost",
        "mortality_rate",
    )
    list_filter = ("month", "organization_code", "disease_code")
    search_fields = ("organization_name", "disease_name")


@admin.register(SingleDiseaseValueDomain)
class SingleDiseaseValueDomainAdmin(admin.ModelAdmin):
    list_display = (
        "disease_code",
        "disease_name",
        "metric",
        "percentile_25",
        "median_50",
        "percentile_75",
        "unit",
        "is_active",
        "created_at",
    )
    list_filter = ("disease_code", "metric", "is_active")
    search_fields = ("disease_name",)