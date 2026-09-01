import { useState } from "react";
import ComponentCard from "../../common/ComponentCard";
import Label from "../Label";
import Select from "../Select";
import MultiSelect from "../MultiSelect";

interface SelectInputsProps {
  value: string;
  onChange: (value: string) => void;
  selectedModels: string[];
  onModelsChange: (models: string[]) => void;
}

export default function SelectInputs({ value, onChange, selectedModels, onModelsChange }: SelectInputsProps) {
  const options = [
    { value: "creditcard", label: "Credit Card" },
    { value: "cyber", label: "Cyber Security" },
    { value: "school", label: "Academic Results" },
    { value: "transistor", label: "Transistor" },
  ];

  const handleSelectChange = (selectedValue: string) => {
    onChange(selectedValue);
  };

  const [selectedValues, setSelectedValues] = useState<string[]>([]);

  const models = [
    { value: "if", text: "Isolation Forest" },
    { value: "lof", text: "Local Outlier Factor" },
    { value: "svm", text: "One-Class SVM" },
  ];

  return (
    <ComponentCard title="Inputs Your Data">
      <div className="space-y-6">
        <div>
          <Label>Select Type Of Data</Label>
          <Select
            options={options}
            placeholder="Select Option"
            onChange={handleSelectChange}
            className="dark:bg-dark-900"
            defaultValue={value}
          />
        </div>
        <div>
          <MultiSelect
            label="Select Models"
            options={models}
            defaultSelected={[]}
            onChange={(values) => {
              setSelectedValues(values);
              onModelsChange(values);
            }}
          />
          <p className="sr-only">
            Selected Values: {selectedValues.join(", ")}
          </p>
        </div>
      </div>
    </ComponentCard>
  );
}
