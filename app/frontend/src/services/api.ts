const API_BASE = process.env.REACT_APP_API_URL ?? "http://localhost:8000";

export interface Prediction {
  label: "benign" | "malignant";
  probability_malignant: number;
  threshold: number;
  model: string;
}

export async function checkHealth(): Promise<{ status: string; model: string }> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error("API health check failed");
  return res.json();
}

export async function predictImage(file: File): Promise<Prediction> {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(body?.detail ?? "Prediction failed");
  }
  return res.json();
}
