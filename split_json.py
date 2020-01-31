import requests
import pandas as pd


def get_data():
    response = requests.get('https://pure-wildwood-95580.herokuapp.com/api/v1/data')

    if response.status_code == 200:
        data = response.json()
        print("Got the data")
    else:
        print("We got nothing")

    return data


if __name__ == '__main__':
    data = get_data()
    for k, v in data.items():
        if k == 'results':
            results = v

    for item in results:
        df = pd.DataFrame.from_dict(results)

    cookies = df[['is_cookie_notice', 'inner_text']]
    print(cookies)
    cookies[140:].to_json(r'./cookies_train.json', orient='index')
    cookies[:140].to_json(r'./cookies_test.json', orient='index')
