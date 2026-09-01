import { useState } from "react";
import PageBreadcrumb from "../../components/common/PageBreadCrumb";
import DropzoneComponent from "../../components/form/form-elements/DropZoneJSON";
import SelectInputs from "../../components/form/form-elements/SelectInputsReports";
import PageMeta from "../../components/common/PageMeta";
import { apiFetch } from "../../services/api";

export default function FormReports() {
  const [selectedType, setSelectedType] = useState("");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  const handleSubmit = async () => {
    if (!selectedType) {
      setError("Please select a data type.");
      return;
    }

    if (!selectedFile) {
      setError("Please upload a JSON predictions file.");
      return;
    }

    setLoading(true);
    setError("");
    setSuccess("");

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      const response = await apiFetch(`/report/${selectedType}`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => null);
        const detail = errorData?.detail;
        throw new Error(
          typeof detail === "string"
            ? detail
            : detail?.message || "Report generation failed."
        );
      }

      const contentDisposition = response.headers.get("content-disposition") || "";
      const filenameMatch = contentDisposition.match(/filename="?([^";]+)"?/i);
      const filename = filenameMatch?.[1] || `report_${selectedType}.pdf`;
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = filename;
      link.click();
      URL.revokeObjectURL(url);

      setSuccess("Report generated, saved, and downloaded successfully.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to generate report.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <PageMeta
        title="Anomaly Arbitration Generate Reports"
        description="Generate and save anomaly analysis reports"
      />
      <PageBreadcrumb pageTitle="Generate Reports" />
      <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
        <div className="space-y-6">
          <SelectInputs value={selectedType} onChange={setSelectedType} />
        </div>
        <div className="space-y-6">
          <DropzoneComponent onFileSelected={setSelectedFile} />
          {selectedFile && (
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Selected file: {selectedFile.name}
            </p>
          )}
          <button
            type="button"
            onClick={handleSubmit}
            disabled={loading}
            className="w-full rounded-lg bg-brand-500 px-4 py-3 text-sm font-medium text-white transition hover:bg-brand-600 disabled:cursor-not-allowed disabled:bg-brand-300"
          >
            {loading ? "Generating..." : "Generate Report"}
          </button>
          {error && <p className="text-sm text-red-600">{error}</p>}
          {success && <p className="text-sm text-green-600">{success}</p>}
        </div>
      </div>
    </div>
  );
}
