from datetime import datetime, date, time as dtime

from django.utils import timezone
from rest_framework import serializers

from .models import Book, UserProfile


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

class SubscriptionSerializer(serializers.Serializer):
    plan_type = serializers.ChoiceField(choices=['Starter', 'Growth', 'Pro', 'Enterprise'])


class PaymentWebhookUpdateSerializer(serializers.ModelSerializer):
    # username은 식별/검증용으로만 받고, 모델 값은 바꾸지 않음
    username = serializers.CharField(write_only=True)

    class Meta:
        model = UserProfile
        # 웹훅으로 바꿔도 되는 필드만 노출 (화이트리스트)
        fields = ("username", "plan_type", "is_paid_member", "end_date")
        extra_kwargs = {
            "plan_type": {"required": False},
            "is_paid_member": {"required": False},
            "end_date": {"required": False, "allow_null": True},
        }

    def _normalize_end_date(self, value):
        """YYYY-MM-DD(날짜만)도 허용 → 당일 23:59:59 로 승격"""
        if isinstance(value, date) and not isinstance(value, datetime):
            tz = timezone.get_current_timezone()
            return timezone.make_aware(datetime.combine(value, dtime.max), tz)
        return value

    def validate(self, attrs):
        request = self.context.get("request")
        secret_key = request.headers.get("X-Secret-Key") if request else None
        if not secret_key:
            raise serializers.ValidationError({"detail": "X-Secret-Key header is missing."})

        # 인스턴스는 뷰에서 username으로 찾아서 주입함
        if not self.instance:
            raise serializers.ValidationError({"detail": "Target user not resolved."})

        # 헤더의 시크릿키와 DB의 시크릿키 일치 검증
        if secret_key != self.instance.secret_key:
            raise serializers.ValidationError({"detail": "Invalid secret key."})

        # payload로 들어온 username과 instance.username도 일치 검증(스푸핑 방지)
        payload_username = attrs.get("username")
        if payload_username and payload_username != self.instance.username:
            raise serializers.ValidationError({"username": ["Username mismatch."]})

        # end_date 포맷 유연 처리(날짜/문자열/ISO8601 모두 허용)
        if "end_date" in attrs and attrs["end_date"] is not None:
            attrs["end_date"] = self._normalize_end_date(attrs["end_date"])

        # (선택) 비즈니스 규칙을 여기에 추가할 수 있음.
        #if attrs.get("end_date") and attrs.get("is_paid_member") is False:
        #    raise serializers.ValidationError({"detail": "Non-paid member cannot have end_date."})

        return attrs

    def update(self, instance: UserProfile, validated_data):
        """
        username은 모델 갱신 금지. 나머지만 부분 업데이트하고
        실제로 바뀐 필드만 save(update_fields=...)로 커밋.
        """
        validated_data.pop("username", None)

        updated = []
        for field in ("plan_type", "is_paid_member", "end_date"):
            if field in validated_data:
                new_val = validated_data[field]
                if getattr(instance, field) != new_val:
                    setattr(instance, field, new_val)
                    updated.append(field)

        # 플랜이 'Free'로 변경되면, 구독 해지 요청 상태를 초기화합니다.
        if validated_data.get("plan_type") == 'Free':
            if instance.cancel_requested is not False:
                instance.cancel_requested = False
                updated.append("cancel_requested")

        if updated:
            updated.append("updated_at")
            instance.save(update_fields=updated)

        # save()의 반환값은 통상 instance지만, 뷰에서 업데이트 목록이 필요하면 보조값 전달
        # 여기서는 관용적으로 instance를 반환하고, updated 목록은 컨텍스트로 빼도 되지만,
        # 편의상 save() 리턴값으로 updated 리스트를 돌려주려면 아래처럼 커스텀 save()를 사용:
        self._updated_fields = updated
        return instance

    def save(self, **kwargs):
        instance = super().save(**kwargs)
        # 뷰에서 업데이트 목록이 필요하면 꺼내쓰도록 반환
        return getattr(self, "_updated_fields", [])