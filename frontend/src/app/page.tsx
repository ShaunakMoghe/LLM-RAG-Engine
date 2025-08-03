"use client";
import { useState } from "react";

// Defines the structure for each message in our chat history
type ChatMessage = {
  role: "user" | "model";
  content: string;
};

export default function Home() {
  // State for the upload and train process
  const [file, setFile] = useState<File | null>(null);
  const [uploadStatus, setUploadStatus] = useState("");
  const [trainStatus, setTrainStatus] = useState("");
  const [uploadedFilename, setUploadedFilename] = useState("");

  // State for the new chat interface
  const [message, setMessage] = useState("");
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const [isChatting, setIsChatting] = useState(false);

  // --- Dynamic Backend URL ---
  // Connects to the backend using the browser's current hostname
  const backendHost = typeof window !== 'undefined' ? window.location.hostname : 'localhost';
  const backendUrl = `http://${backendHost}:8000`;


  const handleUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) {
      alert("Please select a file first.");
      return;
    }
    setUploadStatus("Uploading...");
    setTrainStatus(""); // Reset training status on new upload
    const formData = new FormData();
    formData.append("file", file);
    formData.append("base_model", "microsoft/phi-2");
    formData.append("task_type", "causal-lm");

    try {
      const res = await fetch(`${backendUrl}/upload`, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (res.ok) {
        setUploadedFilename(data.filename);
        setUploadStatus(`✅ Upload successful! Ready to train with ${data.filename}.`);
      } else {
        setUploadStatus(`Upload failed: ${data.detail || "Unknown error"}`);
      }
    } catch (err) {
      setUploadStatus("❌ Upload failed. Is the backend server running?");
      console.error(err);
    }
  };

  const handleTrain = async () => {
    if (!uploadedFilename) {
      alert("Please upload a dataset first.");
      return;
    }
    // This message is updated by the backend immediately, but we'll set a local one too
    setTrainStatus("🚀 Training started... Monitor the backend console for progress.");
    
    // The actual training runs in the background on the server
    try {
      await fetch(`${backendUrl}/train`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filename: uploadedFilename }),
      });
      // Note: We don't get a "finished" message here because it's a background task.
      // The user will know it's done by watching the backend console.
      // For a real production app, you'd use WebSockets or polling to get the status.
    } catch (err) {
      setTrainStatus("❌ Failed to start training request.");
      console.error(err);
    }
  };

  const handleChatSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!message || isChatting) return;

    setIsChatting(true);
    const newHistory: ChatMessage[] = [...chatHistory, { role: "user", content: message }];
    setChatHistory(newHistory);
    const userMessage = message;
    setMessage(""); // Clear the input box immediately

    try {
      const res = await fetch(`${backendUrl}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: userMessage }),
      });
      const data = await res.json();
      if (res.ok) {
        setChatHistory([...newHistory, { role: "model", content: data.response }]);
      } else {
        setChatHistory([...newHistory, { role: "model", content: "Error: Could not get a response." }]);
      }
    } catch (err) {
      setChatHistory([...newHistory, { role: "model", content: "Error: Failed to connect to the server." }]);
      console.error(err);
    } finally {
      setIsChatting(false); // Re-enable the send button
    }
  };

  return (
    <main
      className="min-h-screen p-4 md:p-10 flex flex-col items-center justify-start space-y-6"
      style={{ backgroundColor: "var(--background)", color: "var(--foreground)" }}
    >
      <div className="w-full max-w-2xl">
        <h1 className="text-4xl font-bold text-center mb-6">Fine-Tune & Chat</h1>
        
        {/* --- Upload and Train Section --- */}
        <div className="space-y-4 w-full p-6 rounded-lg shadow-md mb-6" style={{ border: "1px solid var(--foreground)" }}>
          <h2 className="text-2xl font-semibold">1. Upload & Train</h2>
          <form onSubmit={handleUpload} className="space-y-3">
            <input
              type="file"
              accept=".json"
              onChange={(e) => setFile(e.target.files?.[0] || null)}
              className="w-full p-2 border rounded" style={{ backgroundColor: "var(--background)", color: "var(--foreground)", borderColor: "var(--foreground)" }}
            />
            <button
              type="submit"
              className="w-full bg-blue-600 hover:bg-blue-700 p-2 rounded text-white font-semibold disabled:bg-gray-500"
              disabled={!file || uploadStatus === "Uploading..."}
            >
              Upload Dataset
            </button>
          </form>
          {uploadStatus && <p className="text-sm text-center font-semibold">{uploadStatus}</p>}
          <button
            onClick={handleTrain}
            className="w-full bg-green-600 hover:bg-green-700 p-3 rounded text-white font-semibold disabled:bg-gray-500"
            disabled={!uploadedFilename || trainStatus.startsWith("🚀")}
          >
            Start Training
          </button>
          {trainStatus && <p className="text-sm text-center font-semibold">{trainStatus}</p>}
        </div>

        {/* --- Chat Section --- */}
        {/* We'll show the chat box once a file is uploaded, so you can chat after training is done */}
        {uploadedFilename && (
          <div className="w-full p-6 rounded-lg shadow-md" style={{ border: "1px solid var(--foreground)" }}>
            <h2 className="text-2xl font-semibold mb-4">2. Chat with your Model</h2>
            <div className="h-80 overflow-y-auto p-4 mb-4 rounded space-y-4 bg-gray-800" style={{ border: "1px solid var(--foreground)" }}>
              {chatHistory.map((msg, index) => (
                <div key={index} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                  <div className={`p-3 rounded-lg max-w-xs md:max-w-md text-white ${msg.role === 'user' ? 'bg-blue-600' : 'bg-gray-600'}`}>
                    {msg.content}
                  </div>
                </div>
              ))}
               {isChatting && (
                <div className="flex justify-start">
                  <div className="p-3 rounded-lg max-w-xs md:max-w-md text-white bg-gray-600">
                    <span className="animate-pulse">...</span>
                  </div>
                </div>
              )}
            </div>
            <form onSubmit={handleChatSubmit} className="flex space-x-2">
              <input
                type="text"
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                placeholder="Ask your model a question..."
                className="w-full p-2 border rounded" style={{ backgroundColor: "var(--background)", color: "var(--foreground)", borderColor: "var(--foreground)" }}
              />
              <button
                type="submit"
                className="bg-purple-600 hover:bg-purple-700 p-2 rounded text-white font-semibold disabled:bg-gray-500"
                disabled={isChatting}
              >
                Send
              </button>
            </form>
          </div>
        )}
      </div>
    </main>
  );
}