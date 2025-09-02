import os
import subprocess
import polib
import logging
from django.core.management.base import BaseCommand, CommandError
from django.conf import settings
from google import genai
from google.genai import types

# 로깅 설정
log_file_path = os.path.join(settings.BASE_DIR, 'logs/fuzzy_updates.log')
logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class Command(BaseCommand):
    help = '자동으로 makemessages 및 GPT를 이용한 po파일 번역 후 compilemessages를 실행합니다.'

    def add_arguments(self, parser):
        parser.add_argument('language', type=str, help='번역할 언어 코드 (예: ko, ja, en, es, de, fr)')

    def handle(self, *args, **options):
        target_lang = options['language']
        base_dir = settings.BASE_DIR  # Django 환경으로부터 BASE_DIR 자동 인식
        locale_dir = os.path.join(base_dir, 'locale')

        # GPT 번역 함수
        def translate_with_gemini(text, target_language):
            api_key = os.getenv('OPENAI_API_KEY')
            client = genai.Client(api_key=settings.GEMINI_API_KEY)

            sys_msg = f"""
                You are translating a Django `.po` file.
                Translate the following msgid to {target_language}.
                Keep the HTML tags and placeholders unchanged if present.
                Do not modify the HTML tags, whitespaces, or placeholders.
                Ensure the result is in plain text format suitable for Django `.po` files without additional formatting or markdown characters.
                결과물에는 'msgstr:'라고 명시적으로 쓰지 말고 msgstr에 해당하는 내용만 작성해줘.
                `\n`(줄 바꿈) 부분은 그대로 줄바꿈을 유지해야 하므로 그대로 `\n` 이라고 반환해줘.
                이 Django 앱은 PDF형태의 도서를 번역하여 번역본 도서를 생성하는 기능을 제공해. 나는 이 앱의 UI의 문구를 너에게 제공할 것이고, 너는 이러한 앱의 취지에 맞게 UI의 문구를 자연스럽고 정확하며, 신뢰감 있는 표현으로 번역해줘.
                번역 결과물은 타겟언어의 네이티브가 이해하기 쉬운 자연스러운 표현으로 작성해줘. 너무 직역하지 말고, 상황에 맞게 자연스럽게 의역해야해.
                """

            user_msg = (
                f"msgid: {text}"
            )

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                config=types.GenerateContentConfig(
                    system_instruction=sys_msg,
                ),
                contents=user_msg
            )
            return response.text

        # .po 파일 자동 번역 함수
        def auto_translate_po_file(po_file_path, target_language):
            po = polib.pofile(po_file_path)

            for entry in po:
                if not entry.msgstr or 'fuzzy' in entry.flags:
                    if 'fuzzy' in entry.flags:
                        logging.info(f"Fuzzy 처리 항목 업데이트 시작 - msgid: '{entry.msgid}', 기존 번역: '{entry.msgstr}'")
                    else:
                        logging.info(f"번역 시작 - msgid: '{entry.msgid}'")

                    translated_text = translate_with_gemini(entry.msgid, target_language) or ""
                    entry.msgstr = translated_text

                    if 'fuzzy' in entry.flags:
                        entry.flags.remove('fuzzy')
                        logging.info(f"Fuzzy 플래그 제거 및 번역 완료 - msgid: '{entry.msgid}', 새 번역: '{translated_text}'")
                    else:
                        logging.info(f"번역 완료 - msgid: '{entry.msgid}', 새 번역: '{translated_text}'")

            po.save(po_file_path)
            logging.info(f"PO 파일 저장 완료: {po_file_path}")

        # 명령어 실행 (makemessages)
        try:
            subprocess.run([
                "python","manage.py",
                "makemessages",
                "-l", target_lang,
                "--ignore=venv",
                "--ignore=migrations",
                "--ignore=static",
                "--ignore=media",
                "--ignore=node_modules",
                "--ignore=__pycache__",
                "--ignore=tests"
            ], check=True, cwd=base_dir)
            self.stdout.write(self.style.SUCCESS(f"'{target_lang}' 언어에 대한 .po 파일 생성 완료."))
        except subprocess.CalledProcessError as e:
            raise CommandError(f"makemessages 실행 중 오류 발생: {e}")

        # .po 파일 자동 번역 수행
        po_file_path = os.path.join(locale_dir, target_lang, 'LC_MESSAGES', 'django.po')
        auto_translate_po_file(po_file_path, target_lang)

        # 명령어 실행 (compilemessages)
        try:
            subprocess.run(["python", "manage.py", "compilemessages"], check=True, cwd=base_dir)
            self.stdout.write(self.style.SUCCESS(f"'{target_lang}' 번역 메시지 컴파일 완료."))
        except subprocess.CalledProcessError as e:
            raise CommandError(f"compilemessages 실행 중 오류 발생: {e}")

