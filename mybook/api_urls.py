from django.urls import path
from .views import (
    CreateSubscriptionView,
    PaymentWebhookView,
    PayPalSuccessView,
    PayPalCancelView,
    UpgradeSubscriptionView,
    TossPaymentSuccessView,
    TossPaymentFailView,
    CancelSubscriptionView,
)

app_name = 'mybook_api'

# 이 URL들은 i18n_patterns 외부에 위치하여 언어 코드 접두사가 붙지 않습니다.
urlpatterns = [
    path('api/payments/subscribe/', CreateSubscriptionView.as_view(), name='create_subscription'),
    path('api/webhook/user-update/', PaymentWebhookView.as_view(), name='payment_webhook'),
    path('payment/toss/success/', TossPaymentSuccessView.as_view(), name='toss_payment_success'), # API 서버가 리디렉션할 최종 성공 URL
    path('payment/toss/fail/', TossPaymentFailView.as_view(), name='toss_payment_fail'), # API 서버가 리디렉션할 최종 실패 URL
    path('payment/paypal/success/', PayPalSuccessView.as_view(), name='paypal_success'),
    path('payment/paypal/cancel/', PayPalCancelView.as_view(), name='paypal_cancel'),
    path('api/payments/upgrade-subscription/', UpgradeSubscriptionView.as_view(), name='upgrade_subscription'),
    path('api/payments/cancel-subscription/', CancelSubscriptionView.as_view(), name='cancel_subscription'),
]
