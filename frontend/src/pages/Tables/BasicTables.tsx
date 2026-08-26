import PageBreadcrumb from "../../components/common/PageBreadCrumb";
import ComponentCard from "../../components/common/ComponentCard";
import PageMeta from "../../components/common/PageMeta";
import BasicTableOne from "../../components/tables/BasicTables/BasicTableOne";

export default function BasicTables() {
  return (
    <>
      <PageMeta
        title="Anomaly Arbitration Reports Lists Dashboard"
        description="This is React.js Reports Lists Dashboard page for TailAdmin - React.js Tailwind CSS Admin Dashboard Template"
      />
      <PageBreadcrumb pageTitle="Reports Lists" />
      <div className="space-y-6">
        <ComponentCard title="Reports Lists">
          <BasicTableOne />
        </ComponentCard>
      </div>
    </>
  );
}
