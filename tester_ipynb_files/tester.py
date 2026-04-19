import requests

# Test with no auth at all — if we get 401 it's auth, if 404 it's the endpoint itself
r = requests.get(
    'https://www.fotmob.com/api/leagues',
    params={'id': 47, 'ccode3': 'GBR'},
    headers={'User-Agent': 'Mozilla/5.0'}
)
print(r.status_code)
print(r.text[:500])