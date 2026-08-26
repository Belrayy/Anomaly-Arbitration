import { useState } from "react";
import ComponentCard from "../../common/ComponentCard";
import Label from "../Label";
import Select from "../Select";
import MultiSelect from "../MultiSelect";

export default function SelectInputs() {
  const options = [
    { value: "creditcard", label: "Credit Card" },
    { value: "cyber", label: "Cyber Security" },
    { value: "school", label: "Academic Results" },
    { value: "transistor", label: "Transistor" },
  ];
  const handleSelectChange = (value: string) => {
    console.log("Selected value:", value);
  };

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
          />
        </div>
      </div>
    </ComponentCard>
  );
}
