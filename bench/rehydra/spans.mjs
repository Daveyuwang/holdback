// rehydra's spans for each text, with NER and secret detection on. Usage: node spans.mjs in.json out.json
import { createAnonymizer } from "rehydra";
import { readFileSync, writeFileSync } from "node:fs";

const texts = JSON.parse(readFileSync(process.argv[2], "utf8"));
const anonymizer = createAnonymizer({ ner: { mode: "quantized" }, secrets: { enabled: true } });
await anonymizer.initialize();
const out = [];
for (const text of texts) {
  const { entities } = await anonymizer.anonymize(text);
  out.push(entities.map((e) => [e.start, e.end, e.type]));
}
writeFileSync(process.argv[3], JSON.stringify(out));
