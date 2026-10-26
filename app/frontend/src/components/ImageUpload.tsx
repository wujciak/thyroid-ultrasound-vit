import React, { useRef, useState } from "react";
import { predictImage } from "../services/api";

interface HistoryItem {
  id: string;
  fileName: string;
  result: number;
  timestamp: Date;
  imageUrl: string;
}

const ImageUpload = () => {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<number | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      setResult(null);
      setError(null);
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result as string);
      };
      reader.readAsDataURL(selectedFile);
    } else {
      setPreview(null);
      setFile(null);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const res = await predictImage(file);
      setResult(res.prediction);
      
      // Add to history (mock functionality)
      const newHistoryItem: HistoryItem = {
        id: Date.now().toString(),
        fileName: file.name,
        result: res.prediction,
        timestamp: new Date(),
        imageUrl: preview || '',
      };
      setHistory(prev => [newHistoryItem, ...prev].slice(0, 5));
    } catch (err) {
      setError("Prediction failed");
    }
    setLoading(false);
  };

  const handleReset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-100 via-white to-cyan-100">
      {/* Main Content Container */}
      <div className="container mx-auto px-4 py-8">
        <div className="max-w-6xl mx-auto">
          {/* Header */}
          <div className="text-center mb-8">
            <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent mb-2">
              AI Thyroid Diagnosis
            </h1>
            <p className="text-gray-600 text-lg">Advanced ultrasound image classification using machine learning</p>
          </div>

          <div className="grid md:grid-cols-2 gap-8">
            {/* Upload Section */}
            <div className="bg-white/70 backdrop-blur-sm rounded-2xl shadow-xl p-8 border border-white/20">
              <div className="text-center mb-6">
                <div className="inline-flex items-center justify-center w-16 h-16 bg-blue-100 rounded-full mb-4">
                  <svg className="w-8 h-8 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                  </svg>
                </div>
                <h2 className="text-2xl font-bold text-gray-800 mb-2">Upload Image</h2>
                <p className="text-gray-600">Select an ultrasound image for AI analysis</p>
              </div>

              <form onSubmit={handleSubmit} className="space-y-6">
                <div className="border-2 border-dashed border-blue-300 rounded-xl p-8 text-center hover:border-blue-400 transition-colors">
                  <button
                    type="button"
                    className="inline-flex items-center px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white font-semibold rounded-lg shadow-lg hover:from-blue-700 hover:to-purple-700 transition-all disabled:opacity-50"
                    onClick={() => fileInputRef.current?.click()}
                    disabled={loading}
                  >
                    <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                    </svg>
                    Choose Image
                  </button>
                  <input
                    ref={fileInputRef}
                    id="file-input"
                    type="file"
                    accept="image/*"
                    style={{ display: "none" }}
                    onChange={handleFileChange}
                  />
                  <p className="mt-3 text-sm text-gray-500">{file ? file.name : "No file selected"}</p>
                </div>

                {preview && (
                  <div className="relative">
                    <img src={preview} alt="Preview" className="w-full max-h-64 object-contain rounded-xl border border-gray-200 shadow-lg" />
                    <div className="absolute inset-0 bg-gradient-to-t from-black/10 to-transparent rounded-xl"></div>
                  </div>
                )}

                <div className="flex gap-4">
                  <button
                    type="submit"
                    className="flex-1 bg-gradient-to-r from-green-500 to-emerald-600 text-white font-semibold py-3 px-6 rounded-lg shadow-lg hover:from-green-600 hover:to-emerald-700 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                    disabled={!file || loading}
                  >
                    {loading ? (
                      <div className="flex items-center justify-center">
                        <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                        Analyzing...
                      </div>
                    ) : (
                      "Analyze Image"
                    )}
                  </button>
                  <button
                    type="button"
                    className="px-6 py-3 bg-gray-200 text-gray-700 font-semibold rounded-lg shadow hover:bg-gray-300 transition-colors disabled:opacity-50"
                    onClick={handleReset}
                    disabled={loading || (!file && !preview && !result && !error)}
                  >
                    Reset
                  </button>
                </div>
              </form>

              {result !== null && (
                <div className={`mt-6 p-4 rounded-xl text-center ${result === 0 ? "bg-green-50 border-2 border-green-200" : "bg-red-50 border-2 border-red-200"}`}>
                  <div className={`text-2xl font-bold ${result === 0 ? "text-green-700" : "text-red-700"}`}>
                    {result === 0 ? (
                      <div className="flex items-center justify-center">
                        <div className="w-3 h-3 bg-green-500 rounded-full mr-2"></div>
                        Benign
                      </div>
                    ) : (
                      <div className="flex items-center justify-center">
                        <div className="w-3 h-3 bg-red-500 rounded-full mr-2"></div>
                        Malignant
                      </div>
                    )}
                  </div>
                  <p className="text-sm text-gray-600 mt-2">
                    {result === 0 ? "The AI model suggests this nodule appears benign." : "The AI model suggests this nodule may be malignant. Please consult with a medical professional."}
                  </p>
                </div>
              )}

              {error && (
                <div className="mt-6 p-4 bg-red-50 border-2 border-red-200 rounded-xl text-center">
                  <div className="text-red-700 font-semibold">Analysis Failed</div>
                  <p className="text-red-600 text-sm mt-1">{error}</p>
                </div>
              )}
            </div>

            {/* History Section */}
            <div className="bg-white/70 backdrop-blur-sm rounded-2xl shadow-xl p-8 border border-white/20">
              <div className="text-center mb-6">
                <div className="inline-flex items-center justify-center w-16 h-16 bg-purple-100 rounded-full mb-4">
                  <svg className="w-8 h-8 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                </div>
                <h2 className="text-2xl font-bold text-gray-800 mb-2">Recent History</h2>
                <p className="text-gray-600">Your recent analysis results</p>
              </div>

              {history.length > 0 ? (
                <div className="space-y-4 max-h-96 overflow-y-auto">
                  {history.map((item) => (
                    <div key={item.id} className="bg-white/50 rounded-lg p-4 border border-gray-200">
                      <div className="flex items-center space-x-4">
                        <img
                          src={item.imageUrl}
                          alt={item.fileName}
                          className="w-16 h-16 object-cover rounded-lg border border-gray-200"
                        />
                        <div className="flex-1">
                          <div className="font-medium text-gray-800 truncate">{item.fileName}</div>
                          <div className={`text-sm font-semibold ${item.result === 0 ? "text-green-600" : "text-red-600"}`}>
                            {item.result === 0 ? "Benign" : "Malignant"}
                          </div>
                          <div className="text-xs text-gray-500">
                            {item.timestamp.toLocaleDateString()} {item.timestamp.toLocaleTimeString()}
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-12">
                  <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
                    <svg className="w-8 h-8 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                  </div>
                  <p className="text-gray-500">No analysis history yet</p>
                  <p className="text-sm text-gray-400 mt-1">Your recent results will appear here</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ImageUpload;
