import {
  Table,
  TableBody,
  TableCell,
  TableHeader,
  TableRow,
} from "../../ui/table";

import Badge from "../../ui/badge/Badge";
import { useEffect, useState } from "react";
import { apiFetch } from "../../../services/api";

interface Report {
  id: number;
  filename: string;
  model: string;
  algorithm: string;
  created_at: string;
}

export default function BasicTableOne() {
  const [reports, setReports] = useState<Report[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    const loadReports = async () => {
      try {
        const response = await apiFetch("/reports");
        if (!response.ok) {
          const data = await response.json().catch(() => null);
          throw new Error(data?.detail || "Unable to load reports.");
        }
        setReports(await response.json());
      } catch (err) {
        setError(err instanceof Error ? err.message : "Unable to load reports.");
      } finally {
        setLoading(false);
      }
    };

    void loadReports();
  }, []);

  const downloadReport = async (report: Report) => {
    const response = await apiFetch(`/reports/${report.id}`);
    if (!response.ok) {
      throw new Error("Unable to download report.");
    }

    const url = URL.createObjectURL(await response.blob());
    const link = document.createElement("a");
    link.href = url;
    link.download = report.filename;
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="overflow-hidden rounded-xl border border-gray-200 bg-white dark:border-white/[0.05] dark:bg-white/[0.03]">
      <div className="max-w-full overflow-x-auto">
        <Table>
          {/* Table Header */}
          <TableHeader className="border-b border-gray-100 dark:border-white/[0.05]">
            <TableRow>
              <TableCell
                isHeader
                className="px-5 py-3 font-medium text-gray-500 text-start text-theme-xs dark:text-gray-400"
              >
                File
              </TableCell>
              <TableCell
                isHeader
                className="px-5 py-3 font-medium text-gray-500 text-start text-theme-xs dark:text-gray-400"
              >
                Dataset
              </TableCell>
              <TableCell
                isHeader
                className="px-5 py-3 font-medium text-gray-500 text-start text-theme-xs dark:text-gray-400"
              >
                Algorithm
              </TableCell>
              <TableCell
                isHeader
                className="px-5 py-3 font-medium text-gray-500 text-start text-theme-xs dark:text-gray-400"
              >
                Created
              </TableCell>
              <TableCell
                isHeader
                className="px-5 py-3 font-medium text-gray-500 text-start text-theme-xs dark:text-gray-400"
              >
                Download
              </TableCell>
            </TableRow>
          </TableHeader>

          {/* Table Body */}
          <TableBody className="divide-y divide-gray-100 dark:divide-white/[0.05]">
            {loading && <TableRow><TableCell colSpan={5} className="px-5 py-6 text-center">Loading reports...</TableCell></TableRow>}
            {error && <TableRow><TableCell colSpan={5} className="px-5 py-6 text-center text-red-600">{error}</TableCell></TableRow>}
            {!loading && !error && reports.length === 0 && <TableRow><TableCell colSpan={5} className="px-5 py-6 text-center text-gray-500">No reports found.</TableCell></TableRow>}
            {!loading && !error && reports.map((report) => (
              <TableRow key={report.id}>
                <TableCell className="px-5 py-4 text-start text-theme-sm text-gray-800 dark:text-white/90">{report.filename}</TableCell>
                <TableCell className="px-4 py-3 text-start text-theme-sm text-gray-500 dark:text-gray-400">{report.model}</TableCell>
                <TableCell className="px-4 py-3 text-start text-theme-sm text-gray-500 dark:text-gray-400"><Badge size="sm" color="info">{report.algorithm}</Badge></TableCell>
                <TableCell className="px-4 py-3 text-start text-theme-sm text-gray-500 dark:text-gray-400">{new Date(report.created_at).toLocaleString()}</TableCell>
                <TableCell className="px-4 py-3 text-start text-theme-sm">
                  <button
                    type="button"
                    onClick={() => void downloadReport(report)}
                    className="font-medium text-brand-500 hover:text-brand-600"
                  >
                    Download
                  </button>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    </div>
  );
}
