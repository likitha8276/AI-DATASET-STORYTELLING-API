import { useState } from 'react'
import './App.css'

function App() {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [token, setToken] = useState(null)

  const [file, setFile] = useState(null)
  const [result, setResult] = useState(null)

  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  async function handleLogin() {
    setLoading(true)
    setError(null)

    try {
      const formData = new URLSearchParams()

      formData.append('username', email)
      formData.append('password', password)

      const response = await fetch('http://localhost:8000/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded'
        },
        body: formData
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || 'Login failed')
      }

      setToken(data.access_token)
    } catch (error) {
      setError(error.message)
    } finally {
      setLoading(false)
    }
  }

  async function handleAnalyze() {
    if (!file) {
      alert('Please select a CSV file first')
      return
    }

    setLoading(true)
    setError(null)

    try {
      const formData = new FormData()
      formData.append('file', file)

      const response = await fetch('http://localhost:8000/upload', {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${token}`
        },
        body: formData
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(
          data.detail || 'Failed to analyze dataset'
        )
      }

      setResult(data)
    } catch (error) {
      setError(error.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div>
      <h1>DataSense</h1>
      <p>Make sense of your data.</p>

      {!token && (
        <div>
          <h2>Login</h2>

          <input
            type="email"
            placeholder="Email"
            value={email}
            onChange={(event) => setEmail(event.target.value)}
          />

          <input
            type="password"
            placeholder="Password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
          />

          <button onClick={handleLogin} disabled={loading}>
            {loading ? 'Logging in...' : 'Login'}
          </button>
        </div>
      )}

      {token && (
        <div>
          <h2>Upload Dataset</h2>

          <input
            type="file"
            accept=".csv"
            onChange={(event) =>
              setFile(event.target.files[0])
            }
          />

          {file && (
            <p>
              Selected File: {file.name}
            </p>
          )}

          <button
            onClick={handleAnalyze}
            disabled={loading}
          >
            {loading ? 'Analyzing...' : 'Analyze Dataset'}
          </button>
        </div>
      )}

      {error && (
        <p>
          {error}
        </p>
      )}

      {result && (
        <div>
          <h2>Dataset Overview</h2>

          <p>
            Rows: {result.num_rows}
          </p>

          <p>
            Columns: {result.num_columns}
          </p>

          <h2>AI Data Story</h2>

          <p>
            {result.ai_narrative}
          </p>

          <h2>Anomalies</h2>

          <p>
            {result.anomalies?.anomaly_count ?? 0}
            {' '}anomalies detected
          </p>

          <h2>Columns</h2>

          <ul>
            {result.column_names?.map((column) => (
              <li key={column}>
                {column}
              </li>
            ))}
          </ul>

          <h2>Group Analysis</h2>

          <pre>
            {JSON.stringify(
              result.group_analysis,
              null,
              2
            )}
          </pre>
        </div>
      )}
    </div>
  )
}

export default App