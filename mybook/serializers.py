from rest_framework import serializers

class FileUploadSerializer(serializers.Serializer):
    file = serializers.FileField()
    target_language = serializers.CharField(max_length=10)

class RegisterSerializer(serializers.Serializer):
    username = serializers.CharField(max_length=150)
    password = serializers.CharField(write_only=True)
    email = serializers.EmailField()

class LoginSerializer(serializers.Serializer):
    username = serializers.CharField(max_length=150)
    password = serializers.CharField(write_only=True)
