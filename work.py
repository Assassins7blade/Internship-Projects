from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI()

def is_title_case(movie_name: str) -> bool:
    return movie_name == movie_name.title()

class Movie(BaseModel):
    id: int
    title: str
    director: str
    year: int
    genre: Optional[str] = None

movies_db: List[Movie] = []

# Sample data for testing
@app.get("/")
def read_root():
    return {
        "message": "Welcome to my movies!",
        "movies": movies_db
    }

# Get all the movies
@app.get("/movies")
def read_all_movies():
    return movies_db

# Get a movie by the ID
# @app.get("/movies/{movie_id}")
# def read_movie_by_id(movie_id: int):
#     for movie in movies_db:
#         if movie.id == movie_id:
#             return movie
#     raise HTTPException(status_code=404, detail="Movie not found")

# Create a new movie
@app.post("/movies")
def create_movie(movie: Movie):
    # Title case validation
    if not is_title_case(movie.title):
        raise HTTPException(
            status_code=400,
            detail="Invalid format! Please use Title Case (e.g., 'The Dark Knight')."
        )
    
    # Duplicate ID check
    for m in movies_db:
        if m.id == movie.id:
            raise HTTPException(status_code=400, detail="Movie with this ID already exists.")
    
    # Duplicate Title check
    for m in movies_db:
        if m.title == movie.title:
            raise HTTPException(status_code=400, detail="Movie with this Title already exists.")
    
    movies_db.append(movie)
    return {"message": "Movie added successfully", "movie": movie}

# Update a movie by ID
@app.put("/movies/{movie_id}")
def update_movie_by_id(movie_id: int, updated_movie: Movie):
    # Title case validation
    if not is_title_case(updated_movie.title):
        raise HTTPException(
            status_code=400,
            detail="Invalid format! Please use Title Case (e.g., 'The Dark Knight')."
        )
    
    # Duplicate Title check (exclude current movie)
    for m in movies_db:
        if m.title == updated_movie.title and m.id != movie_id:
            raise HTTPException(status_code=400, detail="Movie with this Title already exists.")
    
    for index, m in enumerate(movies_db):
        if m.id == movie_id:
            movies_db[index] = updated_movie
            return {"message": "Movie updated", "movie": updated_movie}
    
    raise HTTPException(status_code=404, detail="Movie not found")

# Delete a movie by ID
@app.delete("/movies/{movie_id}")
def delete_movie_by_id(movie_id: int):
    for m in movies_db:
        if m.id == movie_id:
            movies_db.remove(m)
            return {"message": "Movie deleted"}
    raise HTTPException(status_code=404, detail="Movie not found")
