import { useState, useEffect } from 'react'

const API_BASE = 'http://localhost:8000/api'

function App() {
  const [userId, setUserId] = useState('')
  const [isLoggedIn, setIsLoggedIn] = useState(false)
  const [userProfile, setUserProfile] = useState(null)
  
  const [recommendations, setRecommendations] = useState([])
  const [loading, setLoading] = useState(false)
  const [stats, setStats] = useState(null)

  useEffect(() => {
    fetch(`${API_BASE}/stats`)
      .then(res => res.json())
      .then(data => setStats(data))
      .catch(err => console.error(err))
  }, [])

  const handleLogin = async (e) => {
    e.preventDefault()
    if (!userId) return
    
    setLoading(true)
    try {
      const res = await fetch(`${API_BASE}/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: parseInt(userId) })
      })
      const data = await res.json()
      setUserProfile(data)
      setIsLoggedIn(true)
      fetchRecommendations(userId)
    } catch (err) {
      alert('Failed to login')
    }
  }

  const fetchRecommendations = async (uid) => {
    setLoading(true)
    try {
      const res = await fetch(`${API_BASE}/recommendations/${uid}?k=10`)
      const data = await res.json()
      setRecommendations(data.recommendations || [])
    } catch (err) {
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  const handleLogout = () => {
    setIsLoggedIn(false)
    setUserId('')
    setUserProfile(null)
    setRecommendations([])
  }

  const handleRate = async (movieId, rating) => {
    try {
      await fetch(`${API_BASE}/rate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: parseInt(userId),
          movie_id: movieId,
          rating: rating
        })
      })
      alert(`Rated ${rating} stars! Recommendations will update.`)
      // Refresh recommendations to use Item-CF
      fetchRecommendations(userId)
    } catch (err) {
      console.error(err)
    }
  }

  if (!isLoggedIn) {
    return (
      <div className="container">
        <div className="header">
          <h2>🍿 Netflix Recommender</h2>
          {stats && <small>Model RMSE: {stats.model_comparison[0].RMSE}</small>}
        </div>
        <form className="login-form" onSubmit={handleLogin}>
          <h3>Sign In</h3>
          <p>Enter any User ID to see personalized recommendations.</p>
          <input 
            type="number" 
            placeholder="User ID (e.g. 123)" 
            value={userId}
            onChange={(e) => setUserId(e.target.value)}
          />
          <button className="btn" type="submit" disabled={loading}>
            {loading ? 'Loading...' : 'Sign In'}
          </button>
        </form>
      </div>
    )
  }

  return (
    <div className="container">
      <div className="header">
        <h2>🍿 Netflix Recommender</h2>
        <div>
          <span style={{ marginRight: '1rem' }}>User: #{userId}</span>
          <button className="btn btn-small" onClick={handleLogout}>Log Out</button>
        </div>
      </div>

      <div style={{ backgroundColor: '#e5091420', padding: '1rem', borderRadius: '4px' }}>
        <strong>{userProfile?.message}</strong>
        <p style={{ margin: '0.5rem 0 0 0', fontSize: '0.9rem' }}>
          {userProfile?.is_existing_svd_user 
            ? "You are an existing user. Using advanced SVD Math." 
            : (userProfile?.has_recent_ratings 
                ? "You are a new user with recent ratings. Using Item-Based CF." 
                : "You are a new user. Showing globally popular movies.")}
        </p>
      </div>

      <div>
        <h3 className="section-title">Top 10 Recommendations For You</h3>
        {loading ? (
          <p>Loading your perfect movies...</p>
        ) : (
          <div className="movie-grid">
            {recommendations.map(movie => (
              <div key={movie.movie_id} className="movie-card">
                <div>
                  <div className="movie-title">{movie.title}</div>
                  <div className="movie-year">{movie.year}</div>
                  <div className="movie-score">Predicted Match: {movie.predicted_rating} ⭐</div>
                </div>
                
                <div className="movie-explanation">
                  💡 {movie.explanation}
                </div>

                <div style={{ marginTop: '1rem' }}>
                  <small>Rate this movie:</small>
                  <div className="rating-container">
                    {[1, 2, 3, 4, 5].map(star => (
                      <button 
                        key={star} 
                        className="star-btn"
                        onClick={() => handleRate(movie.movie_id, star)}
                      >
                        ★
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

export default App
