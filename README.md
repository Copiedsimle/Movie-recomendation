# Movie Recommendation System

A web-based movie recommendation system built with Flask and collaborative filtering. The system suggests movies based on user ratings and preferences.

## Features

- Browse popular movies
- Search for movies by title
- Rate movies on a scale of 1-5
- Get personalized movie recommendations
- Responsive web interface

## Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/movie-recommendation.git
cd movie-recommendation
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
.\venv\Scripts\activate  # On Windows
source venv/bin/activate  # On Unix or MacOS
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Flask application:
```bash
cd src
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

## Technologies Used

- Python
- Flask
- NumPy
- Pandas
- scikit-learn
- MovieLens dataset

## Project Structure

```
movie-recommendation/
├── src/
│   ├── app.py              # Flask application
│   ├── main.py            # Main script for training
│   ├── data/              # Data loading and processing
│   ├── models/            # Recommendation models
│   ├── templates/         # HTML templates
│   └── utils/             # Utility functions
├── requirements.txt       # Python dependencies
└── README.md             # Project documentation
```

## Contributing

Feel free to submit issues and enhancement requests! 