import React, { useState, useEffect } from 'react';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [stats, setStats] = useState(null);
  const [movies, setMovies] = useState([]);
  const [loadingStats, setLoadingStats] = useState(true);

  const [userIdInput, setUserIdInput] = useState('305344');
  const [queriedUser, setQueriedUser] = useState(null);
  const [userHistory, setUserHistory] = useState([]);
  const [userRecs, setUserRecs] = useState([]);
  const [loadingUser, setLoadingUser] = useState(false);
  const [userError, setUserError] = useState(null);

  const [selectedMovieId, setSelectedMovieId] = useState('');
  const [similarData, setSimilarData] = useState(null);
  const [loadingSimilar, setLoadingSimilar] = useState(false);
  const [similarError, setSimilarError] = useState(null);

  const sampleUsers = ['305344', '387418', '2439493', '2118461', '1488844'];

  useEffect(() => {
    fetch(`${API_URL}/api/stats`)
      .then((res) => {
        if (!res.ok) throw new Error('API server not ready or stats unavailable');
        return res.json();
      })
      .then((data) => { setStats(data); setLoadingStats(false); })
      .catch(() => setLoadingStats(false));

    fetch(`${API_URL}/api/movies`)
      .then((res) => res.json())
      .then((data) => {
        setMovies(data);
        if (data.length > 0) setSelectedMovieId(data[0].movie_id.toString());
      })
      .catch((err) => console.error('Error fetching movies:', err));
  }, []);

  const fetchUserRecommendations = (uid) => {
    if (!uid) return;
    setLoadingUser(true);
    setUserError(null);
    setQueriedUser(uid);

    fetch(`${API_URL}/api/user/${uid}/history`)
      .then((res) => {
        if (!res.ok) throw new Error('User not found or has no history');
        return res.json();
      })
      .then((history) => {
        setUserHistory(history);
        return fetch(`${API_URL}/api/user/${uid}/recommend?n=10`);
      })
      .then((res) => res.json())
      .then((recs) => { setUserRecs(recs); setLoadingUser(false); })
      .catch((err) => {
        setUserError(err.message);
        setUserHistory([]);
        setUserRecs([]);
        setLoadingUser(false);
      });
  };

  useEffect(() => {
    if (!selectedMovieId) return;
    setLoadingSimilar(true);
    setSimilarError(null);

    fetch(`${API_URL}/api/movie/${selectedMovieId}/similar?n=10`)
      .then((res) => {
        if (!res.ok) throw new Error('Failed to query movie similarity');
        return res.json();
      })
      .then((data) => { setSimilarData(data); setLoadingSimilar(false); })
      .catch((err) => { setSimilarError(err.message); setSimilarData(null); setLoadingSimilar(false); });
  }, [selectedMovieId]);

  useEffect(() => { fetchUserRecommendations(userIdInput); }, []);

  const renderRatingChart = () => {
    if (!stats || !stats.rating_distribution) return null;
    const dist = stats.rating_distribution;
    const values = Object.values(dist);
    const maxVal = Math.max(...values) || 1;
    const labels = Object.keys(dist);
    const chartHeight = 160;
    const chartWidth = 400;
    const barWidth = 45;
    const gap = 20;

    return (
      <svg viewBox={`0 0 ${chartWidth} ${chartHeight + 40}`} width="100%" height="100%">
        {values.map((val, idx) => {
          const barHeight = (val / maxVal) * chartHeight;
          const x = idx * (barWidth + gap) + 40;
          const y = chartHeight - barHeight + 10;
          return (
            <g key={idx}>
              <rect x={x} y={10} width={barWidth} height={chartHeight} fill="#1f1f1f" rx={2} />
              <rect x={x} y={y} width={barWidth} height={barHeight} fill="#e50914" rx={2}
                style={{ transition: 'height 0.4s ease, y 0.4s ease' }} />
              <text x={x + barWidth / 2} y={y - 6} textAnchor="middle" fill="#e5e5e5" fontSize="10" fontWeight="600">
                {val > 1000 ? `${(val / 1000).toFixed(0)}k` : val}
              </text>
              <text x={x + barWidth / 2} y={chartHeight + 25} textAnchor="middle" fill="#555" fontSize="11">
                {labels[idx]}★
              </text>
            </g>
          );
        })}
      </svg>
    );
  };

  const renderPopularityCurve = () => {
    const chartHeight = 160;
    const chartWidth = 400;
    const points = [];
    const numPoints = 50;
    for (let i = 0; i <= numPoints; i++) {
      const x = (i / numPoints) * chartWidth;
      const y = chartHeight - (Math.exp(-i / 10) * (chartHeight - 20) + 10);
      points.push(`${x},${y}`);
    }
    const pathData = `M 0,${chartHeight} L ${points.join(' L ')} L ${chartWidth},${chartHeight} Z`;
    const lineData = `M ${points.join(' L ')}`;

    return (
      <svg viewBox={`0 0 ${chartWidth} ${chartHeight + 30}`} width="100%" height="100%">
        <line x1="0" y1={chartHeight / 2} x2={chartWidth} y2={chartHeight / 2} stroke="#2a2a2a" strokeWidth="1" />
        <path d={pathData} fill="rgba(229,9,20,0.1)" />
        <path d={lineData} fill="none" stroke="#e50914" strokeWidth="2" />
        <text x="10" y="20" fill="#e5e5e5" fontSize="9" fontWeight="600">Blockbusters</text>
        <text x={chartWidth - 90} y={chartHeight - 15} fill="#555" fontSize="9">Long Tail</text>
        <text x="0" y={chartHeight + 20} fill="#555" fontSize="10">Rank 1</text>
        <text x={chartWidth - 55} y={chartHeight + 20} fill="#555" fontSize="10">Rank 200+</text>
      </svg>
    );
  };

  return (
    <div className="container">
      {/* Header */}
      <header className="brand-header">
        <h1 className="brand-logo">
          NETFLIX <span>RECS</span>
        </h1>
        <div className="brand-status">5,000,000 Ratings Analyzed</div>
      </header>

      {/* Navigation */}
      <nav className="tabs-list">
        <button className={`tab-trigger ${activeTab === 'dashboard' ? 'active' : ''}`}
          onClick={() => setActiveTab('dashboard')}>
          System Stats
        </button>
        <button className={`tab-trigger ${activeTab === 'recommendations' ? 'active' : ''}`}
          onClick={() => setActiveTab('recommendations')}>
          Recommendations
        </button>
        <button className={`tab-trigger ${activeTab === 'similarity' ? 'active' : ''}`}
          onClick={() => setActiveTab('similarity')}>
          Movie Similarity
        </button>
      </nav>

      {/* Tab 1: Dashboard */}
      {activeTab === 'dashboard' && (
        <div>
          {loadingStats ? (
            <div className="state-message">Loading system parameters…</div>
          ) : !stats ? (
            <div className="state-message">Backend offline — run <code>uvicorn main:app --reload</code></div>
          ) : (
            <>
              <div className="stats-grid">
                <div className="stat-card">
                  <div className="stat-label">Total Ratings</div>
                  <div className="stat-value">{stats.total_ratings.toLocaleString()}</div>
                  <div className="stat-desc">Parsed rows</div>
                </div>
                <div className="stat-card">
                  <div className="stat-label">Unique Users</div>
                  <div className="stat-value">{stats.unique_users.toLocaleString()}</div>
                  <div className="stat-desc">Distinct viewers</div>
                </div>
                <div className="stat-card">
                  <div className="stat-label">Unique Movies</div>
                  <div className="stat-value">{stats.unique_movies.toLocaleString()}</div>
                  <div className="stat-desc">Catalog size</div>
                </div>
                <div className="stat-card">
                  <div className="stat-label">Matrix Sparsity</div>
                  <div className="stat-value">{(stats.sparsity * 100).toFixed(2)}%</div>
                  <div className="stat-desc">Interaction density</div>
                </div>
              </div>

              <div className="panel-row">
                <div className="panel-card">
                  <span className="section-eyebrow">Distribution</span>
                  <h2 className="panel-title">Rating Frequency</h2>
                  <p className="panel-subtitle">Star rating counts across all interactions</p>
                  <div className="svg-chart-container">{renderRatingChart()}</div>
                </div>
                <div className="panel-card">
                  <span className="section-eyebrow">Analysis</span>
                  <h2 className="panel-title">Popularity Curve</h2>
                  <p className="panel-subtitle">Long-tail Zipfian rating density</p>
                  <div className="svg-chart-container">{renderPopularityCurve()}</div>
                </div>
              </div>
            </>
          )}
        </div>
      )}

      {/* Tab 2: Recommendations */}
      {activeTab === 'recommendations' && (
        <div>
          <span className="section-eyebrow">Personalized</span>
          <h2 className="panel-title" style={{ fontSize: '1.6rem', marginBottom: '0.3rem' }}>
            User Recommendation Engine
          </h2>
          <p className="panel-subtitle" style={{ marginBottom: '1.75rem' }}>
            Enter a user ID to retrieve watch history and generate SVD recommendation vectors.
          </p>

          <div className="form-group">
            <input
              type="text"
              className="form-control"
              placeholder="User ID — e.g. 305344"
              value={userIdInput}
              onChange={(e) => setUserIdInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && fetchUserRecommendations(userIdInput)}
            />
            <button className="btn-primary" onClick={() => fetchUserRecommendations(userIdInput)}>
              Analyze
            </button>
          </div>

          <div className="quick-select-wrap">
            <span className="quick-select-label">Sample Users:</span>
            {sampleUsers.map((uid) => (
              <button key={uid} className="btn-quick-select"
                onClick={() => { setUserIdInput(uid); fetchUserRecommendations(uid); }}>
                {uid}
              </button>
            ))}
          </div>

          {loadingUser ? (
            <div className="state-message">Computing SVD latent factors…</div>
          ) : userError ? (
            <div className="state-error">Error: {userError}</div>
          ) : (
            <div className="panel-row">
              <div className="panel-card">
                <span className="section-eyebrow">History</span>
                <h3 className="panel-title">Watch History</h3>
                <p className="panel-subtitle">Previously rated titles for user {queriedUser}</p>
                <div style={{ overflowX: 'auto' }}>
                  {userHistory.length === 0 ? (
                    <p style={{ color: 'var(--text-dim)', fontSize: '0.85rem' }}>No history found.</p>
                  ) : (
                    <table className="custom-table">
                      <thead>
                        <tr>
                          <th>Title</th>
                          <th>Year</th>
                          <th>Rating</th>
                        </tr>
                      </thead>
                      <tbody>
                        {userHistory.slice(0, 10).map((row, idx) => (
                          <tr key={idx}>
                            <td style={{ fontWeight: '500', color: 'var(--white)' }}>{row.title}</td>
                            <td style={{ color: 'var(--text-muted)' }}>{row.year || '—'}</td>
                            <td><span className="rating-text">{row.rating}/5</span></td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  )}
                </div>
              </div>

              <div className="panel-card">
                <span className="section-eyebrow">Predicted</span>
                <h3 className="panel-title">Top Picks For You</h3>
                <p className="panel-subtitle">SVD score projections for user {queriedUser}</p>
                <div style={{ overflowX: 'auto' }}>
                  {userRecs.length === 0 ? (
                    <p style={{ color: 'var(--text-dim)', fontSize: '0.85rem' }}>No recommendations available.</p>
                  ) : (
                    <table className="custom-table">
                      <thead>
                        <tr>
                          <th>#</th>
                          <th>Title</th>
                          <th>Score</th>
                          <th>Basis</th>
                        </tr>
                      </thead>
                      <tbody>
                        {userRecs.map((row) => (
                          <tr key={row.rank}>
                            <td><span className="rank-num">{row.rank}</span></td>
                            <td style={{ fontWeight: '500', color: 'var(--white)' }}>
                              {row.title}
                              <span style={{ color: 'var(--text-dim)', fontWeight: '400' }}> ({row.year || '—'})</span>
                            </td>
                            <td><span className="rating-text">{row.predicted_rating.toFixed(1)}</span></td>
                            <td style={{ fontSize: '0.73rem', color: 'var(--text-dim)' }}>{row.explanation}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Tab 3: Movie Similarity */}
      {activeTab === 'similarity' && (
        <div>
          <span className="section-eyebrow">Explore</span>
          <h2 className="panel-title" style={{ fontSize: '1.6rem', marginBottom: '0.3rem' }}>
            Movie Similarity Explorer
          </h2>
          <p className="panel-subtitle" style={{ marginBottom: '1.75rem' }}>
            Find nearest neighbors in FunkSVD embedding space and ItemCF co-ratings.
          </p>

          <div className="form-group">
            <select className="form-control" value={selectedMovieId}
              onChange={(e) => setSelectedMovieId(e.target.value)}>
              {movies.map((m) => (
                <option key={m.movie_id} value={m.movie_id}>
                  {m.title} ({m.year || '—'})
                </option>
              ))}
            </select>
          </div>

          {loadingSimilar ? (
            <div className="state-message">Calculating distances in embedding space…</div>
          ) : similarError ? (
            <div className="state-error">Error: {similarError}</div>
          ) : !similarData ? (
            <div className="state-message">No movie selected.</div>
          ) : (
            <div>
              <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '1.75rem' }}>
                Showing neighbors for{' '}
                <strong style={{ color: 'var(--white)' }}>{similarData.movie_title}</strong>
              </p>

              <div className="panel-row">
                <div className="panel-card">
                  <span className="section-eyebrow">Embedding Space</span>
                  <h3 className="panel-title">SVD Latent Neighbors</h3>
                  <p className="panel-subtitle">Cosine similarity via movie latent factors</p>
                  <table className="custom-table">
                    <thead>
                      <tr><th>#</th><th>Title</th><th>Year</th><th>Match</th></tr>
                    </thead>
                    <tbody>
                      {similarData.svd_similar.map((row, idx) => (
                        <tr key={idx}>
                          <td><span className="rank-num">{idx + 1}</span></td>
                          <td style={{ fontWeight: '500', color: 'var(--white)' }}>{row.title}</td>
                          <td style={{ color: 'var(--text-muted)' }}>{row.year || '—'}</td>
                          <td><span className="badge-sim">{(row.score * 100).toFixed(1)}%</span></td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                <div className="panel-card">
                  <span className="section-eyebrow">Co-Rating Behavior</span>
                  <h3 className="panel-title">Item-Item CF Neighbors</h3>
                  <p className="panel-subtitle">Proximity via co-rating correlation</p>
                  <table className="custom-table">
                    <thead>
                      <tr><th>#</th><th>Title</th><th>Year</th><th>Match</th></tr>
                    </thead>
                    <tbody>
                      {similarData.cf_similar.map((row, idx) => (
                        <tr key={idx}>
                          <td><span className="rank-num">{idx + 1}</span></td>
                          <td style={{ fontWeight: '500', color: 'var(--white)' }}>{row.title}</td>
                          <td style={{ color: 'var(--text-muted)' }}>{row.year || '—'}</td>
                          <td><span className="badge-sim">{(row.score * 100).toFixed(1)}%</span></td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default App;
