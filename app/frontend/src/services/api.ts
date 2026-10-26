const API_BASE = "http://localhost:8000";

export async function checkHealth(): Promise<{ status: string }> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error("API health check failed");
  return res.json();
}

export async function predictImage(file: File): Promise<{ prediction: number }> {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) throw new Error("Prediction failed");
  return res.json();
}
