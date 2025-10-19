export function stripAnsi(input = "") {
  if (!input) return "";
  const pattern = /\u001b\[[0-9;]*[A-Za-z]/g;
  return input.replace(pattern, "").replace(/\n{3,}/g, "\n\n");
}
