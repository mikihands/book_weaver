# core/limits.py
PLAN_LIMITS = {
    'Free': {'max_pages': 150, 'Precision':150, 'max_books': 5, 'pdf_download': False, 'max_pages_per_book': 30},
    'Starter': {'max_pages': 1000, 'Precision':100, 'max_books': 'unlimited', 'pdf_download': True},
    'Growth': {'max_pages': 2000, 'Precision':200, 'max_books': 'unlimited', 'pdf_download': True},
    'Pro': {'max_pages': 5000, 'Precision':500, 'max_books': 'unlimited', 'pdf_download': True},
    'Enterprise': {'max_pages': 10000, 'Precision':1000, 'max_books': 'unlimited', 'pdf_download': True},
}

def get_limits(user):
    from .plans import PLAN_LEVELS
    return PLAN_LIMITS.get(getattr(user, "plan_type", "Free"), PLAN_LIMITS['Free'])

#뷰에서 사용예시
"""
limits = get_limits(request.user)
if user_count >= limits['max_pages']:
    return Response({"detail":"limit exceeded"}, status=403)
"""