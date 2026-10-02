// After `vite build`: write dist/artifact.html = dist/index.html without the document wrapper
// (doctype/html/head/body), for hosts that supply their own skeleton (claude.ai artifacts).
import { readFileSync, writeFileSync } from "node:fs";

const html = readFileSync("dist/index.html", "utf8");
const head = html.match(/<head>([\s\S]*)<\/head>/)[1].replace(/<meta [^>]*>\s*/g, "");
const body = html.match(/<body>([\s\S]*)<\/body>/)[1];
writeFileSync("dist/artifact.html", `${head.trim()}\n${body.trim()}\n`);
console.log("dist/artifact.html");
