from rest_framework import serializers
from .models import Book


class FileUploadSerializer(serializers.Serializer):
    file = serializers.FileField()
    target_language = serializers.CharField(max_length=10)
    title = serializers.CharField(max_length=255, required=False, allow_blank=True)
    genre = serializers.CharField(max_length=100, required=False, allow_blank=True)


class RetranslateRequestSerializer(serializers.Serializer):
    feedback = serializers.CharField(max_length=1000, help_text="User feedback for re-translation", required=False, allow_blank=True)
    model_type = serializers.CharField(max_length=16, required=False, default="standard")
    thinking_level = serializers.CharField(max_length=16, required=False, default="medium")


class BookSettingsSerializer(serializers.ModelSerializer):
    class Meta:
        model = Book
        fields = ("title", "genre", "glossary")


class StartTranslationSerializer(serializers.Serializer):
    title = serializers.CharField(max_length=255, required=False, allow_blank=True)
    genre = serializers.CharField(max_length=100, required=False, allow_blank=True)
    glossary = serializers.CharField(required=False, allow_blank=True)
    target_language = serializers.CharField(max_length=10)
    model_type = serializers.CharField(max_length=16, required=False, default="standard")
    thinking_level = serializers.CharField(max_length=16, required=False, default="medium")

class RegisterSerializer(serializers.Serializer):
    username = serializers.CharField(max_length=150)
    email = serializers.EmailField()
    password = serializers.CharField(write_only=True)


class LoginSerializer(serializers.Serializer):
    username = serializers.CharField(max_length=150)
    password = serializers.CharField(write_only=True)


class ContactSerializer(serializers.Serializer):
    name = serializers.CharField(max_length=80)
    email = serializers.EmailField()
    subject = serializers.CharField(max_length=120)
    message = serializers.CharField()
    # 봇 트랩(honeypot): 사람은 비워둠. 채워져 있으면 스팸으로 간주
    website = serializers.CharField(required=False, allow_blank=True, write_only=True)

    def validate(self, attrs):
        # 여기서는 유효성만 확인하고, 실제로는 view에서 is_bot 플래그로 처리
        return attrs
    
class BulkDeleteSerializer(serializers.Serializer):
    ids = serializers.ListField(
        child=serializers.IntegerField(min_value=1),
        allow_empty=False
    )

class PublishRequestSerializer(serializers.Serializer):
    # 기본은 faithful 모드 PDF
    lang = serializers.CharField(max_length=10, required=False, allow_blank=True)
    mode = serializers.ChoiceField(
        choices=["faithful", "readable"],  # TranslatedPage.TRANSLATION_MODES
        required=False, default="faithful"
    )


class PageEditChangeSerializer(serializers.Serializer):
    element_id = serializers.CharField(max_length=100)
    style = serializers.CharField(allow_blank=True, required=False)
    text = serializers.CharField(allow_blank=True, required=False)


class PageEditSerializer(serializers.Serializer):
    mode = serializers.ChoiceField(choices=["faithful", "readable"])
    lang = serializers.CharField(max_length=10)
    changes = serializers.ListField(
        child=PageEditChangeSerializer()
    )