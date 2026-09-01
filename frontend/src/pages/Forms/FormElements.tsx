import { useState } from "react";
import PageBreadcrumb from "../../components/common/PageBreadCrumb";
import DropzoneComponent from "../../components/form/form-elements/DropZone";
import SelectInputs from "../../components/form/form-elements/SelectInputs";
import PageMeta from "../../components/common/PageMeta";


export default function FormElements() {
  const [selectedType, setSelectedType] = useState("");
  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  const triggerDownload = (blob: Blob, filename: string) => {
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    link.style.display = "none";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const handleSubmit = async () => {
    if (!selectedType) {
      setError("Please select a data type.");
      setSuccess("");
      return;
    }

    if (selectedModels.length === 0) {
      setError("Please select at least one model.");
      setSuccess("");
      return;
    }

    if (!selectedFile) {
      setError("Please upload a CSV file.");
      setSuccess("");
      return;
    }

    const token = localStorage.getItem("token");

    if (!token) {
      setError("You must be signed in to upload data.");
      setSuccess("");
      return;
    }

    setLoading(true);
    setError("");
    setSuccess("");

    try {
      const results = [];

      for (const model of selectedModels) {
        const reportType = `${selectedType}-${model}`;

        const formData = new FormData();
        formData.append("file", selectedFile);

        const response = await fetch(
          `http://127.0.0.1:8000/predict/${reportType}`,
          {
            method: "POST",
            headers: {
              Authorization: `Bearer ${token}`,
            },
            body: formData,
          }
        );

        const contentType = response.headers.get("content-type") || "";

        if (!response.ok) {
          const errorData = await response.json().catch(() => null);
          const detail = errorData?.detail;

          const message =
            typeof detail === "string"
              ? detail
              : detail?.message || `Prediction failed for ${reportType}.`;

          throw new Error(message);
        }

        if (contentType.includes("application/json") || contentType.includes("text/plain")) {
          const data = await response.json();
          const blob = new Blob([JSON.stringify(data, null, 2)], {
            type: "application/json",
          });

          triggerDownload(blob, `predictions_${reportType}.json`);
          results.push({
            model: model,
            reportType: reportType,
            rows: data.rows,
          });
        } else {
          const blob = await response.blob();
          triggerDownload(blob, `predictions_${reportType}.pdf`);
          results.push({
            model: model,
            reportType: reportType,
          });
        }
      }

      setSuccess(
        `Prediction completed successfully for ${results.length} model(s). Files were downloaded automatically.`
      );
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Unable to send file."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <PageMeta
        title="Anomaly Arbitration Estimate Anomalies"
        description="This is React.js Form Elements  Dashboard page for TailAdmin - React.js Tailwind CSS Admin Dashboard Template"
      />
      <PageBreadcrumb pageTitle="Form Elements" />
      <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
        <div className="space-y-6">
          <SelectInputs
            value={selectedType}
            onChange={setSelectedType}
            selectedModels={selectedModels}
            onModelsChange={setSelectedModels}
          />
        </div>
        <div className="space-y-6">
          <DropzoneComponent onFileSelected={setSelectedFile} />
          {selectedFile && (
            <div className="rounded-lg border border-gray-200 bg-gray-50 p-3 text-sm text-gray-700 dark:border-gray-700 dark:bg-gray-900 dark:text-gray-300">
              Selected file: {selectedFile.name}
            </div>
          )}
          <button
            type="button"
            onClick={handleSubmit}
            disabled={loading}
            className="w-full rounded-lg bg-brand-500 px-4 py-3 text-sm font-medium text-white transition hover:bg-brand-600 disabled:cursor-not-allowed disabled:bg-brand-300"
          >
            {loading ? "Sending..." : "Send to API"}
          </button>
          {error && (
            <div className="rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-600">
              {error}
            </div>
          )}
          {success && (
            <div className="rounded-lg border border-green-200 bg-green-50 p-3 text-sm text-green-600">
              {success}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
