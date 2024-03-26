import requests
import datetime
def check_credit(apikey):
    subscription_url = "https://api.openai.com/v1/dashboard/billing/subscription"
    headers = {
        "Authorization": "Bearer " + apikey,
        "Content-Type": "application/json"
    }
    subscription_response = requests.get(subscription_url, headers=headers)
    if subscription_response.status_code == 200:
        data = subscription_response.json()
        total = data.get("hard_limit_usd")
        expire_date = datetime.datetime.fromtimestamp(data.get("access_until"))
    else:
        return subscription_response.text

    if expire_date < datetime.datetime.now():
        total = 0

    start_date = (datetime.datetime.now() - datetime.timedelta(days=99)).strftime("%Y-%m-%d")
    end_date = (datetime.datetime.now() + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    billing_url = f"https://api.openai.com/v1/dashboard/billing/usage?start_date={start_date}&end_date={end_date}"
    billing_response = requests.get(billing_url, headers=headers)
    if billing_response.status_code == 200:
        data = billing_response.json()
        total_usage = data.get("total_usage") / 100
    else:
        return billing_response.text

    result = {
        "total": total,
        "used": total_usage,
        "rest": total-total_usage,
        "text": f"{total-total_usage:.2f}/{total:.2f}",
        "date": expire_date
    }
    return result

api_keys = [
    "OPENAI_API_KEYS"
]
for api_key in api_keys:
    result = check_credit(api_key)
    print(api_key, result["text"], result["date"])