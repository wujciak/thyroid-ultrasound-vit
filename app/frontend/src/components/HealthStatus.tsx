import { useEffect, useState } from "react";
import { checkHealth } from "../services/api";

const HealthStatus = () => {
  const [status, setStatus] = useState<string>("checking...");
  const [isLoading, setIsLoading] = useState<boolean>(true);

  useEffect(() => {
    checkHealth()
      .then(() => {
        setStatus("OK");
        setIsLoading(false);
      })
      .catch(() => {
        setStatus("ERROR");
        setIsLoading(false);
      });
  }, []);

  return (
    <div className="fixed top-4 right-4 z-50">
      <div className={`flex items-center space-x-2 px-4 py-2 rounded-full shadow-lg backdrop-blur-sm border ${
        status === "OK" 
          ? "bg-green-100/80 border-green-200 text-green-800" 
          : status === "ERROR"
          ? "bg-red-100/80 border-red-200 text-red-800"
          : "bg-gray-100/80 border-gray-200 text-gray-800"
      }`}>
        <div className={`w-2 h-2 rounded-full ${
          isLoading 
            ? "bg-gray-400 animate-pulse" 
            : status === "OK" 
            ? "bg-green-500" 
            : "bg-red-500"
        }`}></div>
        <span className="text-sm font-medium">
          API: {isLoading ? "Connecting..." : status}
        </span>
      </div>
    </div>
  );
};

export default HealthStatus;
