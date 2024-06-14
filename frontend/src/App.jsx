import { useState } from 'react'
import './App.css'
import axios from 'axios'

function App() {
  const [message, setMessage] = useState('');
  const [response, setResponse] = useState('');

  const sendMessage = () => {
    axios.post('http://localhost:8000/chat', { message: message })
      .then(response => {
        setResponse(response.data.response);
      })
      .catch(error => {
        console.error('There was an error!', error);
      });
  };

  return (
    <>
      <h1>Chat App</h1>
      <textarea 
        cols="30"
        rows="10"
        value={message} 
        onChange={e => setMessage(e.target.value)} 
      />
      <button onClick={sendMessage}>Generate Answer</button>
      <pre>Response: {response}</pre>
    </>
  )
}

export default App
