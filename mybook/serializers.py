from rest_framework import serializers


class FileUploadSerializer(serializers.Serializer):
    file = serializers.FileField()
    target_language = serializers.CharField(max_length=10)
    title = serializers.CharField(max_length=255, required=False, allow_blank=True)
    genre = serializers.CharField(max_length=100, required=False, allow_blank=True)


class RetranslateRequestSerializer(serializers.Serializer):
    feedback = serializers.CharField(max_length=1000, help_text="User feedback for re-translation", required=False, allow_blank=True)


class BookSettingsSerializer(serializers.Serializer):
    title = serializers.CharField(max_length=255, required=False, allow_blank=True)
    genre = serializers.CharField(max_length=100, required=False, allow_blank=True)
    glossary = serializers.CharField(required=False, allow_blank=True)


class StartTranslationSerializer(serializers.Serializer):
    title = serializers.CharField(max_length=255, required=False, allow_blank=True)
    genre = serializers.CharField(max_length=100, required=False, allow_blank=True)
    glossary = serializers.CharField(required=False, allow_blank=True)
    target_language = serializers.CharField(max_length=10)

class RegisterSerializer(serializers.Serializer):
    username = serializers.CharField(max_length=150)
    email = serializers.EmailField()
    password = serializers.CharField(write_only=True)


class LoginSerializer(serializers.Serializer):
    username = serializers.CharField(max_length=150)
    password = serializers.CharField(write_only=True)
