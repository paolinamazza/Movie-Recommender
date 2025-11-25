"""
Data preprocessing module for movie/series recommendation system.

This module handles:
- Loading raw TMDB data
- Extracting and normalizing fields
- Mood mapping based on genres and keywords
- Context suitability scoring
- Dataset enrichment and validation
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MoodMapper:
    """Maps genres, keywords, and themes to emotional moods."""

    MOOD_TAXONOMY = {
        "happy": {
            "genres": ["Comedy", "Family", "Animation", "Musical", "Romance"],
            "keywords": ["friendship", "love", "celebration", "wedding", "success", "reunion"],
            "emotional_profile": "Light-hearted, uplifting, feel-good content",
            "energy_level": "high"
        },
        "sad": {
            "genres": ["Drama", "Romance", "War"],
            "keywords": ["death", "loss", "tragedy", "separation", "terminal illness", "funeral"],
            "emotional_profile": "Heavy, emotional, tear-jerking content",
            "energy_level": "low"
        },
        "excited": {
            "genres": ["Action", "Adventure", "Thriller", "Science Fiction"],
            "keywords": ["chase", "explosion", "superhero", "spy", "heist", "race"],
            "emotional_profile": "High-energy, adrenaline-pumping content",
            "energy_level": "very_high"
        },
        "relaxed": {
            "genres": ["Documentary", "Family", "Animation"],
            "keywords": ["nature", "cooking", "travel", "meditation", "slice of life"],
            "emotional_profile": "Calm, soothing, non-stressful content",
            "energy_level": "low"
        },
        "scared": {
            "genres": ["Horror", "Thriller", "Mystery"],
            "keywords": ["monster", "ghost", "haunted", "killer", "supernatural", "zombie"],
            "emotional_profile": "Suspenseful, frightening, anxiety-inducing content",
            "energy_level": "high"
        },
        "curious": {
            "genres": ["Mystery", "Documentary", "Science Fiction", "Thriller"],
            "keywords": ["investigation", "detective", "mystery", "conspiracy", "puzzle"],
            "emotional_profile": "Intellectually engaging, thought-provoking content",
            "energy_level": "medium"
        },
        "inspired": {
            "genres": ["Biography", "Documentary", "Drama", "History"],
            "keywords": ["based on true story", "biography", "achievement", "overcoming", "hero"],
            "emotional_profile": "Motivational, aspirational, uplifting content",
            "energy_level": "medium"
        },
        "stressed": {
            "genres": ["Thriller", "Horror", "Crime", "War"],
            "keywords": ["tension", "conflict", "chase", "survival", "danger"],
            "emotional_profile": "Intense, nerve-wracking, high-stakes content",
            "energy_level": "very_high"
        },
        "nostalgic": {
            "genres": ["Drama", "Romance", "Family", "History"],
            "keywords": ["childhood", "memory", "past", "vintage", "retro", "coming of age"],
            "emotional_profile": "Reminiscent, sentimental, reflective content",
            "energy_level": "low"
        },
        "bored": {
            "genres": ["Action", "Comedy", "Adventure", "Fantasy"],
            "keywords": ["entertaining", "fun", "adventure", "escape", "fantasy world"],
            "emotional_profile": "Engaging, entertaining, escapist content",
            "energy_level": "high"
        }
    }

    @classmethod
    def get_moods(cls, genres: List[str], keywords: List[str]) -> List[str]:
        """
        Determine applicable moods based on genres and keywords.

        Args:
            genres: List of genre names
            keywords: List of keyword strings

        Returns:
            List of applicable mood tags with confidence scores
        """
        mood_scores = {}

        for mood, taxonomy in cls.MOOD_TAXONOMY.items():
            score = 0

            # Genre matching (higher weight)
            for genre in genres:
                if genre in taxonomy["genres"]:
                    score += 2

            # Keyword matching (lower weight)
            for keyword in keywords:
                keyword_lower = keyword.lower()
                if any(tax_kw in keyword_lower for tax_kw in taxonomy["keywords"]):
                    score += 1

            if score > 0:
                mood_scores[mood] = score

        # Return moods sorted by score
        sorted_moods = sorted(mood_scores.items(), key=lambda x: x[1], reverse=True)
        return [mood for mood, score in sorted_moods]

    @classmethod
    def get_taxonomy(cls) -> Dict[str, Any]:
        """Return the complete mood taxonomy."""
        return cls.MOOD_TAXONOMY


class ContextAnalyzer:
    """Analyzes content suitability for different viewing contexts."""

    CONTEXT_PROFILES = {
        "solo_watch": {
            "description": "Best for individual viewing",
            "allowed_ratings": ["G", "PG", "PG-13", "R", "NC-17", "TV-Y", "TV-Y7", "TV-G", "TV-PG", "TV-14", "TV-MA"],
            "preferred_genres": ["Drama", "Thriller", "Documentary", "Mystery", "Horror"],
            "min_rating": 6.0
        },
        "family_watch": {
            "description": "Safe for all ages",
            "allowed_ratings": ["G", "PG", "PG-13", "TV-Y", "TV-Y7", "TV-G", "TV-PG"],
            "preferred_genres": ["Family", "Animation", "Adventure", "Comedy", "Fantasy"],
            "min_rating": 6.5
        },
        "date_night": {
            "description": "Romantic or engaging content for couples",
            "allowed_ratings": ["PG-13", "R", "TV-14", "TV-MA"],
            "preferred_genres": ["Romance", "Comedy", "Drama", "Thriller"],
            "min_rating": 7.0
        },
        "background_watch": {
            "description": "Low attention required, suitable for multitasking",
            "allowed_ratings": ["G", "PG", "PG-13", "TV-G", "TV-PG", "TV-14"],
            "preferred_genres": ["Comedy", "Documentary", "Reality"],
            "min_rating": 5.5
        },
        "binge_worthy": {
            "description": "Series with high episode count and engagement",
            "allowed_ratings": ["PG-13", "R", "TV-14", "TV-MA"],
            "preferred_genres": ["Drama", "Thriller", "Science Fiction", "Fantasy", "Crime"],
            "min_rating": 7.5,
            "min_seasons": 2
        }
    }

    @classmethod
    def analyze_suitability(cls,
                           content_rating: Optional[str],
                           genres: List[str],
                           rating: float,
                           content_type: str,
                           seasons: Optional[int] = None) -> Dict[str, float]:
        """
        Calculate suitability scores for each viewing context.

        Args:
            content_rating: MPAA/TV rating
            genres: List of genres
            rating: TMDB/IMDB rating
            content_type: "movie" or "series"
            seasons: Number of seasons (for series)

        Returns:
            Dictionary mapping context to suitability score (0-1)
        """
        suitability = {}

        for context, profile in cls.CONTEXT_PROFILES.items():
            score = 0.5  # Base score

            # Skip binge_worthy for movies
            if context == "binge_worthy" and content_type == "movie":
                suitability[context] = 0.0
                continue

            # Rating compatibility
            if content_rating and content_rating in profile["allowed_ratings"]:
                score += 0.3

            # Genre matching
            genre_matches = sum(1 for g in genres if g in profile["preferred_genres"])
            if genre_matches > 0:
                score += min(0.3, genre_matches * 0.15)

            # Quality threshold
            if rating >= profile["min_rating"]:
                score += 0.2

            # Special handling for binge_worthy
            if context == "binge_worthy" and seasons:
                if seasons >= profile.get("min_seasons", 2):
                    score += 0.2
                else:
                    score -= 0.3

            suitability[context] = max(0.0, min(1.0, score))

        return suitability

    @classmethod
    def get_profiles(cls) -> Dict[str, Any]:
        """Return the complete context profiles."""
        return cls.CONTEXT_PROFILES


class DataProcessor:
    """Main data processing pipeline."""

    def __init__(self, raw_data_dir: Path, processed_data_dir: Path):
        """
        Initialize the data processor.

        Args:
            raw_data_dir: Path to raw data directory
            processed_data_dir: Path to processed data output directory
        """
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)

    def load_raw_data(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Load raw TMDB data from JSON files.

        Returns:
            Tuple of (movies_list, series_list)
        """
        logger.info("Loading raw data...")

        movies_path = self.raw_data_dir / "tmdb_movies_enriched.json"
        series_path = self.raw_data_dir / "tmdb_series_enriched.json"

        with open(movies_path, 'r', encoding='utf-8') as f:
            movies = json.load(f)

        with open(series_path, 'r', encoding='utf-8') as f:
            series = json.load(f)

        logger.info(f"Loaded {len(movies)} movies and {len(series)} series")
        return movies, series

    def extract_keywords(self, item: Dict) -> List[str]:
        """Extract keywords from TMDB data structure."""
        keywords = []
        if "keywords" in item:
            if isinstance(item["keywords"], dict) and "keywords" in item["keywords"]:
                keywords = [kw.get("name", "") for kw in item["keywords"]["keywords"]]
            elif isinstance(item["keywords"], list):
                keywords = [kw.get("name", "") for kw in item["keywords"]]
        return keywords

    def get_content_rating(self, item: Dict, content_type: str) -> Optional[str]:
        """Extract content rating (MPAA for movies, TV rating for series)."""
        if content_type == "movie":
            if "releases" in item and "countries" in item["releases"]:
                for country in item["releases"]["countries"]:
                    if country.get("iso_3166_1") == "US":
                        return country.get("certification", "")
        else:  # series
            if "content_ratings" in item and "results" in item["content_ratings"]:
                for rating in item["content_ratings"]["results"]:
                    if rating.get("iso_3166_1") == "US":
                        return rating.get("rating", "")
        return None

    def calculate_total_hours(self, runtime: Optional[int],
                            seasons: Optional[int],
                            episodes: Optional[int]) -> float:
        """Calculate total viewing hours."""
        if runtime:
            if seasons and episodes:
                # Series with known episodes
                return (runtime * episodes) / 60.0
            else:
                # Movie or unknown episode count
                return runtime / 60.0
        return 0.0

    def get_cast_crew(self, item: Dict, content_type: str) -> Tuple[List[str], List[str]]:
        """Extract top 5 cast and directors/creators."""
        cast = []
        crew = []

        # Cast
        if "credits" in item and "cast" in item["credits"]:
            cast = [
                person.get("name", "")
                for person in item["credits"]["cast"][:5]
            ]

        # Crew (directors for movies, creators for series)
        if content_type == "movie":
            if "credits" in item and "crew" in item["credits"]:
                directors = [
                    person.get("name", "")
                    for person in item["credits"]["crew"]
                    if person.get("job") == "Director"
                ]
                crew = directors[:3]
        else:  # series
            if "created_by" in item:
                crew = [creator.get("name", "") for creator in item["created_by"]]

        return cast, crew

    def get_similar_content(self, item: Dict) -> List[int]:
        """Extract similar content IDs."""
        similar_ids = []
        if "similar" in item and "results" in item["similar"]:
            similar_ids = [s.get("id") for s in item["similar"]["results"][:10]]
        return similar_ids

    def process_movie(self, movie: Dict) -> Dict[str, Any]:
        """
        Process a single movie into standardized format.

        Args:
            movie: Raw TMDB movie data

        Returns:
            Processed movie dictionary
        """
        genres = [g.get("name", "") for g in movie.get("genres", [])]
        keywords = self.extract_keywords(movie)
        moods = MoodMapper.get_moods(genres, keywords)
        content_rating = self.get_content_rating(movie, "movie")
        rating = movie.get("vote_average", 0.0)
        runtime = movie.get("runtime", 0)
        cast, crew = self.get_cast_crew(movie, "movie")

        context_suitability = ContextAnalyzer.analyze_suitability(
            content_rating=content_rating,
            genres=genres,
            rating=rating,
            content_type="movie"
        )

        return {
            "id": movie.get("id"),
            "type": "movie",
            "title": movie.get("title", ""),
            "original_title": movie.get("original_title", ""),
            "overview": movie.get("overview", ""),
            "year": movie.get("release_date", "")[:4] if movie.get("release_date") else None,
            "language": movie.get("original_language", ""),
            "tmdb_rating": rating,
            "imdb_rating": None,  # Could be enriched from OMDb
            "vote_count": movie.get("vote_count", 0),
            "genres": genres,
            "keywords": keywords,
            "moods": moods,
            "content_rating": content_rating,
            "runtime_minutes": runtime,
            "total_hours": self.calculate_total_hours(runtime, None, None),
            "context_suitability": context_suitability,
            "cast": cast,
            "crew": crew,
            "similar_ids": self.get_similar_content(movie),
            "poster_url": f"https://image.tmdb.org/t/p/w500{movie.get('poster_path')}" if movie.get('poster_path') else None,
            "backdrop_url": f"https://image.tmdb.org/t/p/original{movie.get('backdrop_path')}" if movie.get('backdrop_path') else None,
            "trailer_url": None,  # Could be extracted from videos
            "popularity": movie.get("popularity", 0.0),
            "status": movie.get("status", ""),
            "tagline": movie.get("tagline", "")
        }

    def process_series(self, series: Dict) -> Dict[str, Any]:
        """
        Process a single series into standardized format.

        Args:
            series: Raw TMDB series data

        Returns:
            Processed series dictionary
        """
        genres = [g.get("name", "") for g in series.get("genres", [])]
        keywords = self.extract_keywords(series)
        moods = MoodMapper.get_moods(genres, keywords)
        content_rating = self.get_content_rating(series, "series")
        rating = series.get("vote_average", 0.0)

        # Episode runtime (average if multiple)
        episode_runtimes = series.get("episode_run_time", [])
        avg_runtime = sum(episode_runtimes) / len(episode_runtimes) if episode_runtimes else 45

        seasons = series.get("number_of_seasons", 0)
        episodes = series.get("number_of_episodes", 0)
        cast, crew = self.get_cast_crew(series, "series")

        context_suitability = ContextAnalyzer.analyze_suitability(
            content_rating=content_rating,
            genres=genres,
            rating=rating,
            content_type="series",
            seasons=seasons
        )

        return {
            "id": series.get("id"),
            "type": "series",
            "title": series.get("name", ""),
            "original_title": series.get("original_name", ""),
            "overview": series.get("overview", ""),
            "year": series.get("first_air_date", "")[:4] if series.get("first_air_date") else None,
            "language": series.get("original_language", ""),
            "tmdb_rating": rating,
            "imdb_rating": None,
            "vote_count": series.get("vote_count", 0),
            "genres": genres,
            "keywords": keywords,
            "moods": moods,
            "content_rating": content_rating,
            "runtime_minutes": int(avg_runtime),
            "seasons": seasons,
            "episodes": episodes,
            "total_hours": self.calculate_total_hours(int(avg_runtime), seasons, episodes),
            "context_suitability": context_suitability,
            "cast": cast,
            "crew": crew,
            "similar_ids": self.get_similar_content(series),
            "poster_url": f"https://image.tmdb.org/t/p/w500{series.get('poster_path')}" if series.get('poster_path') else None,
            "backdrop_url": f"https://image.tmdb.org/t/p/original{series.get('backdrop_path')}" if series.get('backdrop_path') else None,
            "trailer_url": None,
            "popularity": series.get("popularity", 0.0),
            "status": series.get("status", ""),
            "in_production": series.get("in_production", False)
        }

    def process_all(self) -> List[Dict[str, Any]]:
        """
        Process all movies and series.

        Returns:
            Combined list of processed content
        """
        movies_raw, series_raw = self.load_raw_data()

        logger.info("Processing movies...")
        processed_movies = []
        for movie in movies_raw:
            try:
                processed = self.process_movie(movie)
                processed_movies.append(processed)
            except Exception as e:
                logger.warning(f"Error processing movie {movie.get('id')}: {e}")

        logger.info("Processing series...")
        processed_series = []
        for series in series_raw:
            try:
                processed = self.process_series(series)
                processed_series.append(processed)
            except Exception as e:
                logger.warning(f"Error processing series {series.get('id')}: {e}")

        all_content = processed_movies + processed_series
        logger.info(f"Successfully processed {len(processed_movies)} movies and {len(processed_series)} series")

        return all_content

    def save_processed_data(self, content: List[Dict[str, Any]]):
        """Save processed data to JSON file."""
        output_path = self.processed_data_dir / "content_dataset.json"

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(content, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved processed data to {output_path}")

    def save_auxiliary_data(self):
        """Save mood taxonomy and context profiles."""
        # Mood taxonomy
        mood_path = self.processed_data_dir / "mood_taxonomy.json"
        with open(mood_path, 'w', encoding='utf-8') as f:
            json.dump(MoodMapper.get_taxonomy(), f, indent=2)
        logger.info(f"Saved mood taxonomy to {mood_path}")

        # Context profiles
        context_path = self.processed_data_dir / "context_profiles.json"
        with open(context_path, 'w', encoding='utf-8') as f:
            json.dump(ContextAnalyzer.get_profiles(), f, indent=2)
        logger.info(f"Saved context profiles to {context_path}")

    def generate_statistics(self, content: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate dataset statistics.

        Args:
            content: List of processed content items

        Returns:
            Statistics dictionary
        """
        movies = [c for c in content if c["type"] == "movie"]
        series = [c for c in content if c["type"] == "series"]

        # Genre distribution
        all_genres = []
        for item in content:
            all_genres.extend(item["genres"])
        genre_counts = {}
        for genre in all_genres:
            genre_counts[genre] = genre_counts.get(genre, 0) + 1

        # Mood distribution
        all_moods = []
        for item in content:
            all_moods.extend(item["moods"])
        mood_counts = {}
        for mood in all_moods:
            mood_counts[mood] = mood_counts.get(mood, 0) + 1

        # Rating statistics
        ratings = [c["tmdb_rating"] for c in content if c["tmdb_rating"] > 0]

        stats = {
            "total_items": len(content),
            "total_movies": len(movies),
            "total_series": len(series),
            "genre_distribution": dict(sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)),
            "mood_distribution": dict(sorted(mood_counts.items(), key=lambda x: x[1], reverse=True)),
            "rating_stats": {
                "min": min(ratings) if ratings else 0,
                "max": max(ratings) if ratings else 0,
                "mean": sum(ratings) / len(ratings) if ratings else 0,
                "count": len(ratings)
            },
            "language_distribution": self._count_field(content, "language"),
            "year_range": {
                "earliest": min([c["year"] for c in content if c["year"]], default=None),
                "latest": max([c["year"] for c in content if c["year"]], default=None)
            },
            "content_with_moods": sum(1 for c in content if c["moods"]),
            "content_with_ratings": sum(1 for c in content if c["content_rating"])
        }

        return stats

    def _count_field(self, content: List[Dict], field: str) -> Dict[str, int]:
        """Helper to count occurrences of a field value."""
        counts = {}
        for item in content:
            value = item.get(field)
            if value:
                counts[value] = counts.get(value, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10])


def main():
    """Main execution function."""
    base_dir = Path(__file__).parent.parent.parent
    raw_data_dir = base_dir / "data" / "raw"
    processed_data_dir = base_dir / "data" / "processed"

    processor = DataProcessor(raw_data_dir, processed_data_dir)

    # Process all content
    content = processor.process_all()

    # Save processed data
    processor.save_processed_data(content)
    processor.save_auxiliary_data()

    # Generate and print statistics
    stats = processor.generate_statistics(content)
    print("\n" + "="*80)
    print("DATASET STATISTICS")
    print("="*80)
    print(f"\nTotal Items: {stats['total_items']}")
    print(f"Movies: {stats['total_movies']}")
    print(f"Series: {stats['total_series']}")
    print(f"\nYear Range: {stats['year_range']['earliest']} - {stats['year_range']['latest']}")
    print(f"\nRating Statistics:")
    print(f"  Min: {stats['rating_stats']['min']:.2f}")
    print(f"  Max: {stats['rating_stats']['max']:.2f}")
    print(f"  Mean: {stats['rating_stats']['mean']:.2f}")
    print(f"\nTop 10 Genres:")
    for genre, count in list(stats['genre_distribution'].items())[:10]:
        print(f"  {genre}: {count}")
    print(f"\nMood Distribution:")
    for mood, count in stats['mood_distribution'].items():
        print(f"  {mood}: {count}")
    print(f"\nTop 10 Languages:")
    for lang, count in list(stats['language_distribution'].items())[:10]:
        print(f"  {lang}: {count}")
    print(f"\nData Quality:")
    print(f"  Items with moods: {stats['content_with_moods']} ({stats['content_with_moods']/stats['total_items']*100:.1f}%)")
    print(f"  Items with content ratings: {stats['content_with_ratings']} ({stats['content_with_ratings']/stats['total_items']*100:.1f}%)")
    print("\n" + "="*80)

    # Save statistics
    stats_path = processed_data_dir / "dataset_statistics.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    print(f"\nStatistics saved to {stats_path}")

    # Print sample items
    print("\n" + "="*80)
    print("SAMPLE PROCESSED ITEMS")
    print("="*80)

    sample_movie = next((c for c in content if c["type"] == "movie"), None)
    if sample_movie:
        print("\nSample Movie:")
        print(json.dumps(sample_movie, indent=2))

    sample_series = next((c for c in content if c["type"] == "series"), None)
    if sample_series:
        print("\nSample Series:")
        print(json.dumps(sample_series, indent=2))


if __name__ == "__main__":
    main()
