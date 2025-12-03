// src/utils/downloadCsv.js

// Utility to download an array of objects as a CSV file
export function downloadCsv(rows, filename = "spottox_results.csv") {
  // If there's no data, tell the user and stop
  if (!rows || rows.length === 0) {
    alert("No data to export.");
    return;
  }

  // Get the column names from the first row (object keys)
  const headers = Object.keys(rows[0]);

  const csvLines = [];

  // First line: the header row
  csvLines.push(headers.join(","));

  // Each following line: the values for each row
  for (const row of rows) {
    const line = headers
      .map((h) => {
        const value = row[h] ?? "";
        // Escape double quotes in the value
        const s = String(value).replace(/"/g, '""');
        // Wrap each value in quotes so commas inside text are safe
        return `"${s}"`;
      })
      .join(",");
    csvLines.push(line);
  }

  // Join everything into one big string with newlines
  const csvContent = csvLines.join("\n");

  // Create a downloadable blob
  const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);

  // Create a hidden link and click it
  const link = document.createElement("a");
  link.href = url;
  link.setAttribute("download", filename);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}
