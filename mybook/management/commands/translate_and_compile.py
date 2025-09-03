import os
import subprocess
import polib
import logging
import time
from django.core.management.base import BaseCommand, CommandError
from django.conf import settings
from google import genai
from google.genai import types, errors

# 로깅 설정
log_file_path = os.path.join(settings.BASE_DIR, 'logs/fuzzy_updates.log')
logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class Command(BaseCommand):
    help = '자동으로 makemessages 및 Gemini를 이용한 po파일 번역 후 compilemessages를 실행합니다.'

    def add_arguments(self, parser):
        parser.add_argument('language', type=str, help='번역할 언어 코드 (예: ko, ja, en, es, de, fr)')

    def handle(self, *args, **options):
        target_lang = options['language']
        base_dir = settings.BASE_DIR
        locale_dir = os.path.join(base_dir, 'locale')

        client = genai.Client(api_key=settings.PAID_GEMINI_KEY)

        def translate_with_gemini(text, target_language, max_retries=5):

            # sys_msg = f"""
            #     You are translating a Django `.po` file.
            #     Translate the following msgid to {target_language}.
            #     Keep the HTML tags and placeholders unchanged if present.
            #     Do not modify the HTML tags, whitespaces, or placeholders.
            #     Ensure the result is in plain text format suitable for Django `.po` files without additional formatting or markdown characters.
            #     결과물에는 'msgstr:'라고 명시적으로 쓰지 말고 msgstr에 해당하는 내용만 작성해줘.
            #     `\n`(줄 바꿈) 부분은 그대로 줄바꿈을 유지해야 하므로 그대로 `\n` 이라고 반환해줘.
            #     이 Django 앱은 PDF형태의 도서를 번역하여 번역본 도서를 생성하는 기능을 제공해. 나는 이 앱의 UI의 문구를 너에게 제공할 것이고, 너는 이러한 앱의 취지에 맞게 UI의 문구를 자연스럽고 정확하며, 신뢰감 있는 표현으로 번역해줘.
            #     번역 결과물은 타겟언어의 네이티브가 이해하기 쉬운 자연스러운 표현으로 작성해줘. 너무 직역하지 말고, 상황에 맞게 자연스럽게 의역해야해.
            #     브랜드명 'BookWeaver'는 번역하지 말고 그대로 'BookWeaver'라고 써줘.
            #     """
            sys_msg = f"""
                You are a professional translator for a Django web application.
                Your task is to translate UI strings from a `.po` file into {target_language}.
                Follow these strict rules:

                1.  **Translate the provided `msgid` text.** Do not include `msgstr:` or any other `.po` file syntax in your output. Provide only the translated content.
                2.  **Preserve all original formatting and syntax.** This includes HTML tags (e.g., `<b>`, `</b>`), Django placeholders (e.g., `%(name)s`), and all whitespace characters, including newlines (`\n`). Do not modify or remove them.
                3.  **Translate in a natural, native tone.** The app, named **BookWeaver**, translates books from PDF files while preserving their original layout. Your translations should reflect this brand identity: sound natural, accurate, and trustworthy. Avoid literal, word-for-word translations. Instead, use idiomatic expressions that a native speaker of {target_language} would find clear and relatable.
                4.  **Do not translate the brand name 'BookWeaver'.** Keep it as 'BookWeaver' in the final output.
                """

            user_msg = f"msgid: {text}"

            for attempt in range(max_retries):
                try:
                    response = client.models.generate_content(
                        model="gemini-2.5-flash",
                        config=types.GenerateContentConfig(
                            system_instruction=sys_msg,
                        ),
                        contents=user_msg
                    )
                    return response.text
                except errors.ServerError as e:
                    if e.code == 503 and attempt < max_retries - 1:
                        sleep_time = 2 ** attempt  # 지수 백오프
                        logging.warning(f"503 UNAVAILABLE 에러 발생. {sleep_time}초 후 재시도... (시도 {attempt + 1}/{max_retries})")
                        time.sleep(sleep_time)
                    else:
                        raise CommandError(f"API 요청 실패 후 재시도 횟수 초과: {e}")
                except Exception as e:
                    raise CommandError(f"알 수 없는 오류 발생: {e}")
            return ""

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
        if not os.path.exists(po_file_path):
             raise CommandError(f"'{po_file_path}' 파일이 존재하지 않습니다. makemessages가 제대로 실행되었는지 확인하세요.")
        auto_translate_po_file(po_file_path, target_lang)

        # 명령어 실행 (compilemessages)
        try:
            subprocess.run(["python", "manage.py", "compilemessages"], check=True, cwd=base_dir)
            self.stdout.write(self.style.SUCCESS(f"'{target_lang}' 번역 메시지 컴파일 완료."))
        except subprocess.CalledProcessError as e:
            raise CommandError(f"compilemessages 실행 중 오류 발생: {e}")
