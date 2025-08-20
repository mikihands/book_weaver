import logging, os
import requests
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings
from django.db import transaction
from django.contrib.auth.models import User
from django.shortcuts import get_object_or_404, render
from django.http import HttpResponse

from .serializers import FileUploadSerializer, RegisterSerializer
from .models import UploadedFile, TranslatedPage, UserProfile
from .utils.gemini_helper import GeminiHelper
from common.mixins.hmac_sign_mixin import HmacSignMixin

logger = logging.getLogger(__name__)
AUTH_SERVER_URL=settings.AUTH_SERVER_URL

class BookUploadView(APIView):
    def post(self, request, *args, **kwargs):
        """
        사용자로부터 PDF/이미지 파일을 받아 Gemini API를 통해 번역하고 HTML로 변환합니다.
        """
        serializer = FileUploadSerializer(data=request.data)
        if serializer.is_valid():
            uploaded_file_obj = serializer.validated_data['file'] #type:ignore
            target_language = serializer.validated_data['target_language'] #type:ignore
            
            # TODO: 실제 사용자 인증 로직으로 변경 (현재는 임시로 첫 번째 유저 사용)
            user, created = User.objects.get_or_create(username='jesse')
            
            try:
                # 1. 파일 임시 저장
                # Django의 MEDIA_ROOT 설정을 사용
                file_name = uploaded_file_obj.name
                # 업로드 파일을 저장할 디렉터리 경로 설정 (예: media/uploads)
                upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')

                # 디렉터리가 없으면 생성
                os.makedirs(upload_dir, exist_ok=True)

                temp_file_path = os.path.join(upload_dir, file_name)

                with open(temp_file_path, 'wb+') as destination:
                    for chunk in uploaded_file_obj.chunks():
                        destination.write(chunk)

                # 2. 업로드 파일 정보 DB에 저장
                with transaction.atomic():
                    uploaded_file_db = UploadedFile.objects.create(
                        user=user,
                        original_file=uploaded_file_obj
                    )

                    # 3. GeminiHelper를 사용하여 API 호출
                    gemini_helper = GeminiHelper()
                    translated_html = gemini_helper.process_and_translate_document(temp_file_path, target_language)

                    if translated_html:
                        # 4. 번역된 HTML을 DB에 저장
                        # 현재는 모든 HTML을 한 페이지로 가정
                        # TODO: 향후 페이지별로 분할하여 저장하는 로직 추가
                        translated_page = TranslatedPage.objects.create(
                            uploaded_file=uploaded_file_db,
                            page_number=1, # 임시 페이지 번호
                            translated_html=translated_html,
                            html_url=f"/books/{uploaded_file_db.id}/pages/1/" # type: ignore
                        )

                        return Response({
                            "message": "File processed and translated.",
                            "translated_page_url": translated_page.html_url
                        }, status=status.HTTP_201_CREATED)
                    else:
                        return Response({
                            "error": "Failed to process the document."
                        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            except Exception as e:
                # 오류 발생 시 임시 파일 삭제
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                return Response({
                    "error": f"An unexpected error occurred: {str(e)}"
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
class BookPageView(APIView):
    def get(self, request, book_id, page_number, *args, **kwargs):
        """
        특정 책의 번역된 페이지 HTML을 반환합니다.
        """
        translated_page = get_object_or_404(
            TranslatedPage, 
            uploaded_file__id=book_id, 
            page_number=page_number
        )
        
        # render() 함수를 사용하여 템플릿을 렌더링
        context = {
            'translated_html': translated_page.translated_html,
            'page_number': translated_page.page_number
        }
        return render(request, 'mybook/book_page.html', context)
    
class UploadPage(APIView):
    def get(self, request, *args, **kwargs):
        return render(request, 'mybook/upload_page.html')


class RegisterView(HmacSignMixin, APIView):
    def post(self, request, *args, **kwargs):
        serializer = RegisterSerializer(data=request.data)
        AUTH_REGISTER_ENDPOINT = "/api/accounts/register/"

        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        try:
            resp = self.hmac_post(AUTH_REGISTER_ENDPOINT, serializer.validated_data) # type: ignore
        except Exception as e:
            return Response(
                {"error": f"Failed to connect to authentication server: {e}"},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        if resp.status_code == status.HTTP_201_CREATED:
            auth_data = resp.json()
            # 멱등 필요 시 get_or_create 권장
            UserProfile.objects.create(
                username=auth_data["user"]["username"],
                email=auth_data["user"]["email"],
                secret_key=auth_data["user"]["secret_key"],
            )
            return Response({"message": "User registered successfully."}, status=status.HTTP_201_CREATED)

        # 실패를 그대로 전달
        try:
            print(f"Server response status code: {resp.status_code}")
            print(f"Server response text: {resp.text}")
            return Response(resp.json(), status=resp.status_code)
        except ValueError:
            return Response({"error": "Auth server returned non-JSON error."}, status=resp.status_code)

    
class RegisterSuccessView(APIView):
    def get(self, request, *args, **kwargs):
        return render(request, 'mybook/register_success.html')
    
class RegisterPage(APIView):
    def get(self, request, *args, **kwargs):
        return render(request, 'mybook/register_page.html')