from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib import messages
from django.contrib.auth import login, logout
from django.contrib.auth.decorators import login_required
from django.db.models import Avg
import pandas as pd
from movies.models import Rating, Movie
from sklearn.metrics.pairwise import cosine_similarity
from django.contrib.auth.decorators import login_required
from movies.tmdb_api import search_movies, get_movie_details
from movies.utils import get_movie_details
from django.core.paginator import Paginator


def recommendations_view(request):

    user = request.user

    # Load ratings from the database
    ratings_qs = Rating.objects.all().values('user_id', 'movie_id', 'rating')
    df = pd.DataFrame(ratings_qs)

    # Handle case when no ratings exist or if the user hasn't rated any movie
    if df.empty or user.id not in df['user_id'].unique():
        return render(request, 'movies/recommendations.html', {
            'recommendations': [],
            'message': 'Not enough data to generate recommendations.'
        })

    # Create user-movie matrix
    rating_matrix = df.pivot_table(index='user_id', columns='movie_id', values='rating').fillna(0)

    # Compute cosine similarity between users
    user_sim = cosine_similarity(rating_matrix)
    user_sim_df = pd.DataFrame(user_sim, index=rating_matrix.index, columns=rating_matrix.index)

    # Get similarities for the current user (exclude themselves)
    if user.id not in user_sim_df.index:
        return render(request, 'movies/recommendations.html', {
            'recommendations': [],
            'message': 'You have not rated any movies yet.'
        })

    sim_scores = user_sim_df.loc[user.id].drop(user.id)
    similar_users = sim_scores[sim_scores > 0]

    # If no similar users, return early
    if similar_users.empty:
        return render(request, 'movies/recommendations.html', {
            'recommendations': [],
            'message': 'No similar users found for recommendations.'
        })

    # Get ratings from similar users
    aligned_ratings = rating_matrix.loc[similar_users.index]

    # Multiply each user's ratings by their similarity score
    ratings_matrix_np = aligned_ratings.to_numpy().T  # shape: (movies, users)
    sim_weights_np = similar_users.to_numpy()         # shape: (users,)

    weighted_scores = ratings_matrix_np.dot(sim_weights_np)
    sim_sum = sim_weights_np.sum() or 1e-9  # avoid division by zero
    final_scores = weighted_scores / sim_sum

    # Convert results to Series with movie IDs as index
    movie_ids = aligned_ratings.columns
    scores_series = pd.Series(final_scores, index=movie_ids).sort_values(ascending=False)

    # Filter out movies already rated by the user
    already_rated = set(df[df['user_id'] == user.id]['movie_id'])
    recommended_movie_ids = [mid for mid in scores_series.index if mid not in already_rated][:10]

    recommended_movies = Movie.objects.filter(id__in=recommended_movie_ids)

    return render(request, 'movies/recommendations.html', {
        'recommendations': recommended_movies,
        'message': None if recommended_movies else 'No new recommendations found.'
    })

def login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            messages.success(request, "You have successfully logged in.")
            return redirect('home')  
        else:
            messages.error(request, "Invalid username or password.")
    else:
        form = AuthenticationForm()

    return render(request, 'movies/login.html', {'form': form})



def register_view(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, 'Registration successfull You can now log in.')
            return redirect('login')  
    else:
        form = UserCreationForm()
    return render(request, 'movies/register.html', {'form': form})



@login_required
def rate_movies_view(request):
    movies = Movie.objects.all()

    if request.method == 'POST':
        for movie in movies:
            rating_val = request.POST.get(f'rating_{movie.id}')
            if rating_val:
                rating_obj, created = Rating.objects.update_or_create(
                    user=request.user,
                    movie=movie,
                    defaults={'rating': rating_val}
                )
        messages.success(request, "Ratings submitted successfully!")
        return redirect('home')

    user_rated = Rating.objects.filter(user=request.user)
    user_ratings = {r.movie.id: r.rating for r in user_rated}

    return render(request, 'movies/rate_movies.html', {
        'movies': movies,
        'user_ratings': user_ratings,
    })


def home_view(request):
    return render(request, 'movies/home.html')


def get_item(dictionary, key):
    return dictionary.get(key)


def logout_view(request):
    logout(request)
    return redirect('login')

def search_results_view(request):
    query = request.GET.get('q', '')
    local_results = []
    tmdb_results = []
    if query:
        local_results = Movie.objects.filter(title__icontains=query)
        tmdb_results = search_movies(query)  # returns list of dicts from TMDb API
        
        local_titles = set(m.title.lower() for m in local_results)
        tmdb_results = [movie for movie in tmdb_results if movie['title'].lower() not in local_titles]
    return render(request, 'movies/search_results.html', {'local_results': local_results, 'tmdb_results': tmdb_results, 'query': query})



def movie_detail_view(request, movie_id):
    # First try to fetch the movie from the local database
    movie_instance = Movie.objects.filter(id=movie_id).first()

    # If not found locally, fetch from TMDB API
    if not movie_instance:
        movie_data = get_movie_details(movie_id)
        if not movie_data:
            return render(request, 'movies/movie_not_found.html')
     
        genres = movie_data.get('genres', [])
        genre_names = ', '.join([genre['name'] for genre in genres]) if genres else ''

        # Save to DB so it can be used as ForeignKey
        movie_instance = Movie.objects.create(
            id=movie_id,
            title=movie_data['title'],
            genre=genre_names,
            year=movie_data.get('release_date', '')[:4] if movie_data.get('release_date') else None,
            description=movie_data.get('overview', ''),
            poster_url=f"https://image.tmdb.org/t/p/w500{movie_data.get('poster_path')}" if movie_data.get('poster_path') else '',
            release_year=int(movie_data.get('release_date')[:4]) if movie_data.get('release_date') else None
        )

    # Handle rating retrieval and submission
    user_rating = None
    if request.user.is_authenticated:
        user_rating = Rating.objects.filter(user=request.user, movie=movie_instance).first()

        if request.method == "POST":
            rating_val = request.POST.get("rating")
            if rating_val:
                Rating.objects.update_or_create(
                    user=request.user,
                    movie=movie_instance,
                    defaults={'rating': rating_val}
                )
                return redirect('movie_detail', movie_id=movie_id)

    context = {
        'movie': movie_instance,
        'user_rating': user_rating,
    }
    return render(request, 'movies/movie_detail.html', context)


def filter_movies_view(request):
    genre = request.GET.get('genre')
    year = request.GET.get('year')
    min_rating = request.GET.get('min_rating')
    query = request.GET.get('q')

    movies = Movie.objects.all()

    if genre and genre != 'All':
        movies = movies.filter(genre__icontains=genre)
    
    if year:
        movies = movies.filter(release_year=year)

    if query:
        movies = movies.filter(title__icontains=query)

    # ✅ Only annotate once — and do it before pagination
    movies = movies.annotate(avg_rating=Avg('rating__rating'))

    if min_rating:
        try:
            min_rating = float(min_rating)
            movies = movies.filter(avg_rating__gte=min_rating)
        except ValueError:
            pass  # Ignore invalid rating input


    # ✅ Pagination
    paginator = Paginator(movies, 5)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    print("Total movies:", movies.count())
    print("Page:", page_obj.number)
    print("Number of movies on this page:", len(page_obj))
    
    context = {
        'page_obj': page_obj,
        'q': query or '',
        'genre': genre or 'All',
        'year': year or '',
        'min_rating': min_rating or '',
    }

    return render(request, 'filter_movies.html', context)
    
