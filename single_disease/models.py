from django.db import models


class Single_disease_report(models.Model):
    """
    单病种报表模型 (Single_disease_report)

    每条记录表示某个月、某机构、某单病种的汇总指标。
    过滤维度：month(YYYY-MM)、disease_code、organization_code。
    """

    id = models.BigAutoField(primary_key=True)

    # 月份：存储为该月第一天，便于索引与区间查询
    month = models.DateField(help_text="统计月份，使用该月第一天表示，例如 2025-07-01")

    # 机构信息（没有现成机构表时以代码/名称直存；若有机构模型，可替换为外键）
    organization_code = models.CharField(max_length=64, db_index=True)
    organization_name = models.CharField(max_length=255)

    # 单病种信息（若有字典表，可替换为外键）
    disease_code = models.CharField(max_length=64, db_index=True)
    disease_name = models.CharField(max_length=255)

    # 指标
    case_count = models.IntegerField(help_text="病种例数")
    average_length_of_stay_days = models.DecimalField(max_digits=10, decimal_places=1, help_text="平均住院日")
    average_cost = models.DecimalField(max_digits=12, decimal_places=0, help_text="次均费用")
    mortality_rate = models.DecimalField(max_digits=5, decimal_places=1, help_text="病死率，单位%")

    # 元数据
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "single_disease_report"
        verbose_name = "单病种报表"
        verbose_name_plural = "单病种报表"
        indexes = [
            models.Index(fields=["month", "organization_code", "disease_code"], name="idx_sdr_month_org_dis"),
        ]
        unique_together = ("month", "organization_code", "disease_code")

    def __str__(self) -> str:
        return f"{self.month:%Y-%m} {self.organization_name} {self.disease_name}"


class SingleDiseaseValueDomain(models.Model):
    """
    单病种值域表

    存储：每个单病种在不同指标下的 25/50/75 分位值及单位。
    这里的 metric 采用枚举型，覆盖四项指标：病种例数、平均住院日、次均费用、病死率。
    """

    class Metric(models.TextChoices):
        CASE_COUNT = "case_count", "病种例数"
        AVG_LOS = "average_length_of_stay_days", "平均住院日"
        AVG_COST = "average_cost", "次均费用"
        MORTALITY = "mortality_rate", "病死率"

    id = models.BigAutoField(primary_key=True)
    disease_code = models.CharField(max_length=64, db_index=True)
    disease_name = models.CharField(max_length=255)

    metric = models.CharField(max_length=64, choices=Metric.choices, db_index=True)

    percentile_25 = models.DecimalField(max_digits=12, decimal_places=1)
    median_50 = models.DecimalField(max_digits=12, decimal_places=1)
    percentile_75 = models.DecimalField(max_digits=12, decimal_places=1)

    unit = models.CharField(max_length=32, help_text="单位，如 天/元/%/例")

    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    remark = models.TextField(blank=True, null=True)

    class Meta:
        db_table = "single_disease_value_domain"
        verbose_name = "单病种值域"
        verbose_name_plural = "单病种值域"
        unique_together = ("disease_code", "metric")
        indexes = [
            models.Index(fields=["disease_code", "metric"], name="idx_sdvd_dis_metric"),
        ]

    def __str__(self) -> str:
        return f"{self.disease_name} - {self.get_metric_display()}"