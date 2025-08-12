from __future__ import annotations

from datetime import date, timedelta
from typing import Dict, List

from django.db.models import F, QuerySet
from django.utils.timezone import now
from rest_framework import viewsets, mixins
from rest_framework.request import Request
from rest_framework.response import Response
from django_filters.rest_framework import DjangoFilterBackend

from .filters import (
    SingleDiseaseReportFilter,
    SingleDiseaseValueDomainFilter,
    first_day_of_month_from_str,
)
from .models import Single_disease_report, SingleDiseaseValueDomain
from .serializers import (
    ReportResponseSerializer,
    SingleDiseaseReportSerializer,
    SingleDiseaseValueDomainSerializer,
)


def get_default_month_str() -> str:
    """默认月份：上一个自然月，格式 YYYY-MM"""
    today = now().date()
    first_this_month = today.replace(day=1)
    last_month_last_day = first_this_month - timedelta(days=1)
    return last_month_last_day.strftime("%Y-%m")


class SingleDiseaseValueDomainViewSet(viewsets.ModelViewSet):
    """视图：单病种值域表 CRUD"""

    queryset = SingleDiseaseValueDomain.objects.all().order_by("disease_code", "metric")
    serializer_class = SingleDiseaseValueDomainSerializer
    filterset_class = SingleDiseaseValueDomainFilter
    filter_backends = [DjangoFilterBackend]


class SingleDiseaseReportViewSet(mixins.ListModelMixin, viewsets.GenericViewSet):
    """
    视图：单病种报表（只读列表）

    规则：
    - 未筛选机构 organization_code：按机构分组，展示某单病种在所有机构的情况（附顶部 25/50/75 分位值）。
    - 已筛选机构：按单病种分组，展示该机构的所有单病种（同样附顶部分位值，针对各指标）。
    - month 必填，默认值=上个月；single disease 可不传，默认选择该月内第一条的 disease_code。
    """

    queryset = Single_disease_report.objects.all().order_by("organization_name", "disease_name")
    serializer_class = SingleDiseaseReportSerializer
    filterset_class = SingleDiseaseReportFilter
    filter_backends = [DjangoFilterBackend]

    def list(self, request: Request, *args, **kwargs) -> Response:
        # 解析 query params（带默认值）
        month_str = request.query_params.get("month") or get_default_month_str()
        organization_code = request.query_params.get("organization_code")
        disease_code = request.query_params.get("disease_code")

        # 构造基础查询：限定月份
        month_first_day = first_day_of_month_from_str(month_str)
        queryset = (
            self.queryset if hasattr(self, "queryset") else Single_disease_report.objects
        )
        qs_for_month: QuerySet = queryset.filter(month__year=month_first_day.year, month__month=month_first_day.month)

        # 自动补默认 disease_code
        if not organization_code and not disease_code:
            first_row = qs_for_month.order_by("disease_code").first()
            disease_code = first_row.disease_code if first_row else None

        # 根据是否筛选机构进行分组展示
        if organization_code:
            # 展示该机构的所有单病种
            result_qs = qs_for_month.filter(organization_code=organization_code)
            mode = "by_organization"
        else:
            # 展示某单病种的所有机构
            result_qs = qs_for_month
            if disease_code:
                result_qs = result_qs.filter(disease_code=disease_code)
            mode = "by_disease"

        # 查值域的分位值（四项指标，各三条），按 disease 或 organization 取一个参照 disease_code（若没传机构则用 disease_code；传了机构则不限定病种，返回该机构不区分病种的通用值域也可以，这里仍按 disease_code 取，若空则 None）
        quartiles = self._build_quartile_rows(disease_code)

        result = {
            "month": month_first_day,
            "mode": mode,
            "disease_code": disease_code,
            "organization_code": organization_code,
            "quartiles": quartiles,
            "results": SingleDiseaseReportSerializer(result_qs, many=True).data,
        }
        return Response(result)

    def _build_quartile_rows(self, disease_code: str | None) -> List[Dict]:
        # 如果没有 disease_code，返回空数组。前端可选择隐藏顶部行。
        if not disease_code:
            return []

        vd_qs = SingleDiseaseValueDomain.objects.filter(disease_code=disease_code, is_active=True)
        # 将四项指标按 metric 聚合成一个三行结构
        # 默认各指标缺失时置为 0
        metric_to_values: Dict[str, Dict[str, float]] = {
            m: {"p25": 0, "p50": 0, "p75": 0} for m, _ in SingleDiseaseValueDomain.Metric.choices
        }
        for row in vd_qs:
            metric_to_values[row.metric] = {
                "p25": float(row.percentile_25),
                "p50": float(row.median_50),
                "p75": float(row.percentile_75),
            }

        def pick(metric_key: str, tier: str) -> float:
            return metric_to_values.get(metric_key, {}).get(tier, 0.0)

        return [
            {
                "label": "25分位值",
                "case_count": pick(SingleDiseaseValueDomain.Metric.CASE_COUNT, "p25"),
                "average_length_of_stay_days": pick(SingleDiseaseValueDomain.Metric.AVG_LOS, "p25"),
                "average_cost": pick(SingleDiseaseValueDomain.Metric.AVG_COST, "p25"),
                "mortality_rate": pick(SingleDiseaseValueDomain.Metric.MORTALITY, "p25"),
            },
            {
                "label": "中位值",
                "case_count": pick(SingleDiseaseValueDomain.Metric.CASE_COUNT, "p50"),
                "average_length_of_stay_days": pick(SingleDiseaseValueDomain.Metric.AVG_LOS, "p50"),
                "average_cost": pick(SingleDiseaseValueDomain.Metric.AVG_COST, "p50"),
                "mortality_rate": pick(SingleDiseaseValueDomain.Metric.MORTALITY, "p50"),
            },
            {
                "label": "75分位值",
                "case_count": pick(SingleDiseaseValueDomain.Metric.CASE_COUNT, "p75"),
                "average_length_of_stay_days": pick(SingleDiseaseValueDomain.Metric.AVG_LOS, "p75"),
                "average_cost": pick(SingleDiseaseValueDomain.Metric.AVG_COST, "p75"),
                "mortality_rate": pick(SingleDiseaseValueDomain.Metric.MORTALITY, "p75"),
            },
        ]