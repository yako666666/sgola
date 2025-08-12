from datetime import date
from calendar import monthrange

import django_filters
from django.db.models import QuerySet

from .models import Single_disease_report, SingleDiseaseValueDomain


def first_day_of_month_from_str(month_str: str) -> date:
    """将 YYYY-MM 转换为该月第一天的 date。"""
    year, month = [int(x) for x in month_str.split("-")]
    return date(year, month, 1)


class SingleDiseaseReportFilter(django_filters.FilterSet):
    """过滤器：单病种报表"""

    # 前端传 YYYY-MM
    month = django_filters.CharFilter(method="filter_month")
    disease_code = django_filters.CharFilter(field_name="disease_code", lookup_expr="exact")
    organization_code = django_filters.CharFilter(field_name="organization_code", lookup_expr="exact")

    class Meta:
        model = Single_disease_report
        fields = ["month", "disease_code", "organization_code"]

    def filter_month(self, queryset: QuerySet, name: str, value: str) -> QuerySet:
        month_first_day = first_day_of_month_from_str(value)
        days_in_month = monthrange(month_first_day.year, month_first_day.month)[1]
        month_last_day = month_first_day.replace(day=days_in_month)
        return queryset.filter(month__range=(month_first_day, month_last_day))


class SingleDiseaseValueDomainFilter(django_filters.FilterSet):
    disease_code = django_filters.CharFilter(lookup_expr="exact")
    metric = django_filters.CharFilter(lookup_expr="exact")
    is_active = django_filters.BooleanFilter()

    class Meta:
        model = SingleDiseaseValueDomain
        fields = ["disease_code", "metric", "is_active"]