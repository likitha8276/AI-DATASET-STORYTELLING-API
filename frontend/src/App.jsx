import { useState } from 'react'
import './App.css'

function App() {
  const[file,setFile]=useState(null)
  function handleAnalyze(){
    console.log(file)
  }
  return (
    <div>
      <h1>DataSense</h1>
      <p>Make sense of your data.</p>

      <input type="file" accept=".csv"
      onChange={(event)=> setFile(event.target.files[0])} />
      {file &&  <p> Selected File : {file.name}</p>}
      <button onClick={handleAnalyze}>
        Analyze Dataset
      </button>
    </div>
  )
}

export default App