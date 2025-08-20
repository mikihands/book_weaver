from django.conf import settings

def global_settings(request):
    """
    settings.DEBUG 값을 템플릿에 'DEBUG' 변수로 전달합니다.
    """
    return {
        'DEBUG_MODE': settings.DEBUG, # 더 명확한 변수 이름을 사용합니다.
    }