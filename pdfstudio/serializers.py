# pdfstudio/serializers.py
from rest_framework import serializers

class ReorderSerializer(serializers.Serializer):
    order = serializers.ListField(
        child=serializers.IntegerField(min_value=1),
        allow_empty=False
    )

class PagesSerializer(serializers.Serializer):
    page_no = serializers.IntegerField()
    width = serializers.FloatField()
    height = serializers.FloatField()
    preview_url = serializers.CharField()
    deleted = serializers.BooleanField()

class PageListResponse(serializers.Serializer):
    pages = PagesSerializer(many=True)

class DeleteRestoreSerializer(serializers.Serializer):
    pages = serializers.ListField(
        child=serializers.IntegerField(min_value=1),
        allow_empty=False
    )
