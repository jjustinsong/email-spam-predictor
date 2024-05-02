import './App.css';
import {useState} from 'react';
import React from 'react';
import axios from 'axios';

function App() {
  const [spam, setSpam] = useState("N/A");

  const [subject, setSubject] = useState("");
  const [email, setEmail] = useState("");

  const handleSubmit = async (event) => {
    event.preventDefault();

    try {
      const data = {subject, email};
      const response = await axios.post("http://127.0.0.1:5000/predict", data);
      setSpam(response.data.prediction);
    } catch(error) {
      console.error("Error: ", error);
      console.error("Detailed response: ", error.response)
      alert("Failed to get prediction: " + error.message);
    }
  }
  return (
    <>
      <h1 className="mt-10 text-3xl font-bold text-center font-mukta">
        Spam Detection System
      </h1>
      <form onSubmit={handleSubmit} className="m-20">
        <div className="mb-3">
          <label htmlFor="subject" className="font-mukta block text-lg font-medium">
            Subject:
          </label>
          <input
            type="text"
            id="subject"
            value={subject}
            onChange={(e) => setSubject(e.target.value)}
            className="w-full p-2 border rounded font-mukta"
            placeholder="Enter email subject"
            required
          />
        </div>
        <div className="mb-3">
          <label htmlFor="email" className="font-mukta block text-lg font-medium">
            Email Body:
          </label>
          <textarea
            id="emailBody"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="w-full p-2 border rounded font-mukta"
            rows="6"
            placeholder="Enter email body"
            required
          ></textarea>
        </div>
        <button type="submit" className="font-mukta px-4 py-2 text-black bg-slate-400 rounded hover:bg-slate-600 hover:text-white">
          SPAM OR HAM
        </button>
      </form>

      <div className="font-mukta m-20 text-lg">
        Prediction: <strong className="font-mukta">{spam}</strong>
      </div>
    </>
  );
}

export default App;
