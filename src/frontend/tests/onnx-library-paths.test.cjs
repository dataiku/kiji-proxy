const fs = require("fs");
const os = require("os");
const path = require("path");
const { appendUniqueVersionedOnnxPaths } = require("../src/electron/onnx-library-paths");

const root = fs.mkdtempSync(path.join(os.tmpdir(), "kiji-onnx-paths-"));
try {
  const paths = ["libonnxruntime.dylib"];
  fs.writeFileSync(
    path.join(root, `libonnxruntime.${["1", "2", "3"].join(".")}.dylib`),
    ""
  );
  appendUniqueVersionedOnnxPaths(fs, paths, [root]);
  if (paths.length !== 2) throw new Error(`expected one fallback, got ${paths}`);

  fs.writeFileSync(
    path.join(root, `libonnxruntime.${["2", "0", "0"].join(".")}.dylib`),
    ""
  );
  const warnings = [];
  const ambiguousPaths = ["libonnxruntime.dylib"];
  appendUniqueVersionedOnnxPaths(fs, ambiguousPaths, [root], (message) => warnings.push(message));
  if (ambiguousPaths.length !== 1) throw new Error("ambiguous libraries were not skipped");
  if (warnings.length !== 1 || !warnings[0].includes("ONNXRUNTIME_SHARED_LIBRARY_PATH")) {
    throw new Error(`missing ambiguity guidance: ${warnings}`);
  }
} finally {
  fs.rmSync(root, { recursive: true, force: true });
}
