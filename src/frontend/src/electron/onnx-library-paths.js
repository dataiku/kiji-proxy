const path = require("path");

function appendUniqueVersionedOnnxPaths(
  fs,
  paths,
  directories,
  warn = console.warn
) {
  for (const directory of directories) {
    if (!fs.existsSync(directory)) continue;

    const matches = fs
      .readdirSync(directory)
      .filter((name) => /^libonnxruntime\.\d+\.\d+\.\d+\.dylib$/.test(name));
    if (matches.length === 1) {
      paths.push(path.join(directory, matches[0]));
    } else if (matches.length > 1) {
      warn(
        `Multiple versioned ONNX Runtime libraries found in ${directory}; create libonnxruntime.dylib or set ONNXRUNTIME_SHARED_LIBRARY_PATH explicitly`
      );
    }
  }
  return paths;
}

module.exports = { appendUniqueVersionedOnnxPaths };
