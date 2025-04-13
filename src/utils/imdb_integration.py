import imdb
from typing import Dict, List, Optional
import json
import os
from datetime import datetime, timedelta

class IMDbIntegration:
    def __init__(self):
        self.ia = imdb.IMDb()
        self.cache_dir = 'src/data/cache'
        self._memory_cache = {}
        self._cache_duration = timedelta(days=7)  # Cache for 7 days
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        
        # Load popular movies cache on initialization
        self._popular_movies_cache = self._load_cache('popular_movies.json')

    def _load_cache(self, filename: str) -> Dict:
        """Load cache from file"""
        cache_file = os.path.join(self.cache_dir, filename)
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                cache = json.load(f)
                if cache.get('timestamp'):
                    cache_time = datetime.fromisoformat(cache['timestamp'])
                    if datetime.now() - cache_time < self._cache_duration:
                        return cache.get('data', {})
        return {}

    def _save_cache(self, filename: str, data: Dict):
        """Save cache to file"""
        cache_file = os.path.join(self.cache_dir, filename)
        cache = {
            'timestamp': datetime.now().isoformat(),
            'data': data
        }
        with open(cache_file, 'w') as f:
            json.dump(cache, f)

    def search_movie(self, title: str, max_results: int = 5) -> List[Dict]:
        """Search for movies by title"""
        cache_key = f"search_{title}_{max_results}"
        
        # Check memory cache
        if cache_key in self._memory_cache:
            return self._memory_cache[cache_key]
        
        try:
            movies = self.ia.search_movie(title)[:max_results]
            results = []
            
            for movie in movies:
                movie_data = {
                    'imdb_id': movie.movieID,
                    'title': movie.get('title', ''),
                    'year': movie.get('year', ''),
                    'kind': movie.get('kind', 'movie')
                }
                results.append(movie_data)
            
            # Cache results in memory
            self._memory_cache[cache_key] = results
            return results
        except Exception as e:
            print(f"Error searching for movie: {e}")
            return []

    def get_movie_details(self, imdb_id: str) -> Optional[Dict]:
        """Get detailed information about a movie by IMDb ID"""
        # Check memory cache
        if imdb_id in self._memory_cache:
            return self._memory_cache[imdb_id]
            
        # Check file cache
        cache = self._load_cache(f"movie_{imdb_id}.json")
        if cache:
            self._memory_cache[imdb_id] = cache
            return cache

        try:
            movie = self.ia.get_movie(imdb_id)
            
            details = {
                'imdb_id': imdb_id,
                'title': movie.get('title', ''),
                'year': movie.get('year', ''),
                'rating': movie.get('rating', 0.0),
                'votes': movie.get('votes', 0),
                'genres': movie.get('genres', []),
                'plot': movie.get('plot outline', ''),
                'director': [d.get('name', '') for d in movie.get('directors', [])],
                'cast': [a.get('name', '') for a in movie.get('cast', [])[:5]],
                'runtime': movie.get('runtimes', [''])[0] if movie.get('runtimes') else '',
                'poster_url': movie.get('full-size cover url', ''),
                'languages': movie.get('languages', []),
                'countries': movie.get('countries', [])
            }
            
            # Cache results
            self._memory_cache[imdb_id] = details
            self._save_cache(f"movie_{imdb_id}.json", details)
            return details
            
        except Exception as e:
            print(f"Error getting movie details: {e}")
            return None

    def get_popular_movies(self, max_results: int = 10) -> List[Dict]:
        """Get popular movies from IMDb"""
        # Check cache first
        if self._popular_movies_cache:
            return self._popular_movies_cache[:max_results]

        try:
            top_movies = self.ia.get_top250_movies()[:max_results]
            results = []
            
            for movie in top_movies:
                movie_data = {
                    'imdb_id': movie.movieID,
                    'title': movie.get('title', ''),
                    'year': movie.get('year', ''),
                    'rating': movie.get('rating', 0.0)
                }
                results.append(movie_data)
            
            # Cache results
            self._popular_movies_cache = results
            self._save_cache('popular_movies.json', results)
            return results
        except Exception as e:
            print(f"Error getting popular movies: {e}")
            return []

    def get_movie_recommendations(self, movie_id: str, max_results: int = 5) -> List[Dict]:
        """Get movie recommendations based on a movie ID"""
        cache_key = f"recommendations_{movie_id}_{max_results}"
        
        # Check memory cache
        if cache_key in self._memory_cache:
            return self._memory_cache[cache_key]
            
        # Check file cache
        cache = self._load_cache(f"recommendations_{movie_id}.json")
        if cache:
            self._memory_cache[cache_key] = cache
            return cache[:max_results]

        try:
            movie = self.ia.get_movie(movie_id)
            genres = movie.get('genres', [])
            
            similar_movies = []
            for genre in genres:
                movies = self.ia.get_popular_movies_by_genre(genre)
                similar_movies.extend(movies)
            
            similar_movies = list({m.movieID: m for m in similar_movies 
                                if m.movieID != movie_id}.values())
            
            similar_movies.sort(key=lambda x: x.get('rating', 0), reverse=True)
            results = []
            
            for movie in similar_movies[:max_results]:
                movie_data = {
                    'imdb_id': movie.movieID,
                    'title': movie.get('title', ''),
                    'year': movie.get('year', ''),
                    'rating': movie.get('rating', 0.0)
                }
                results.append(movie_data)
            
            # Cache results
            self._memory_cache[cache_key] = results
            self._save_cache(f"recommendations_{movie_id}.json", results)
            return results
        except Exception as e:
            print(f"Error getting movie recommendations: {e}")
            return [] 