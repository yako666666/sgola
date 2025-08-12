from typing import Any, Dict, List

from rest_framework import serializers

from .models import Single_disease_report, SingleDiseaseValueDomain


class SingleDiseaseReportSerializer(serializers.ModelSerializer):
    """序列化器：单病种报表行"""

    class Meta:
        model = Single_disease_report
        fields = [
            "id",
            "month",
            "organization_code",
            "organization_name",
            "disease_code",
            "disease_name",
            "case_count",
            "average_length_of_stay_days",
            "average_cost",
            "mortality_rate",
            "created_at",
        ]


class SingleDiseaseValueDomainSerializer(serializers.ModelSerializer):
    """序列化器：单病种值域表"""

    metric_display = serializers.CharField(source="get_metric_display", read_only=True)

    class Meta:
        model = SingleDiseaseValueDomain
        fields = [
            "id",
            "disease_code",
            "disease_name",
            "metric",
            "metric_display",
            "percentile_25",
            "median_50",
            "percentile_75",
            "unit",
            "is_active",
            "created_at",
            "remark",
        ]


class QuartileRowSerializer(serializers.Serializer):
    """用于报表顶部三行（25分位/中位/75分位）的返回结构。"""

    label = serializers.CharField()
    case_count = serializers.DecimalField(max_digits=12, decimal_places=1)
    average_length_of_stay_days = serializers.DecimalField(max_digits=12, decimal_places=1)
    average_cost = serializers.DecimalField(max_digits=12, decimal_places=1)
    mortality_rate = serializers.DecimalField(max_digits=12, decimal_places=1)


class ReportResponseSerializer(serializers.Serializer):
    """统一报表响应：含表头三行分位值与数据行"""

    month = serializers.DateField()
    mode = serializers.ChoiceField(choices=["by_disease", "by_organization"])  # by_disease: 机构分组；by_organization: 单病种分组
    disease_code = serializers.CharField(allow_null=True)
    organization_code = serializers.CharField(allow_null=True)

    quartiles = QuartileRowSerializer(many=True)
    results = SingleDiseaseReportSerializer(many=True)