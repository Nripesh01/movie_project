import requests

TMDB_API_KEY = '656e5806caeedbf67149f2ce2aed064b'  

def get_movie_details(movie_id):
    url = f'https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=en-US'
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return None
