import ComponentCard from "../../common/ComponentCard";
import Label from "../Label";
import Select from "../Select";

interface SelectInputsReportsProps {
  value: string;
  onChange: (value: string) => void;
}

export default function SelectInputsReports({ value, onChange }: SelectInputsReportsProps) {
  const options = [
    { value: "creditcard", label: "Credit Card" },
    { value: "cyber", label: "Cyber Security" },
    { value: "school", label: "Academic Results" },
    { value: "transistor", label: "Transistor" },
  ];
  return (
    <ComponentCard title="Inputs Your Data">
      <div className="space-y-6">
        <div>
          <Label>Select Type Of Data</Label>
          <Select
            options={options}
            placeholder="Select Option"
            onChange={onChange}
            className="dark:bg-dark-900"
            defaultValue={value}
          />
        </div>
      </div>
    </ComponentCard>
  );
}
