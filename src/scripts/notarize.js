const { notarize } = require("@electron/notarize");
const fs = require("fs");
const os = require("os");
const path = require("path");

exports.default = async function notarizing(context) {
  const { electronPlatformName, appOutDir } = context;
  if (electronPlatformName !== "darwin") return;

  const apiKeyInput = process.env.APPLE_API_KEY;
  const apiKeyId = process.env.APPLE_API_KEY_ID;
  const apiIssuer = process.env.APPLE_API_ISSUER;

  if (!apiKeyInput || !apiKeyId || !apiIssuer) {
    console.log(
      "Skipping notarization (APPLE_API_KEY, APPLE_API_KEY_ID, and APPLE_API_ISSUER are all required)"
    );
    return;
  }

  const appName = context.packager.appInfo.productName;
  const appPath = `${appOutDir}/${appName}.app`;

  // APPLE_API_KEY can be either a file path (from CI) or raw .p8 content.
  let keyFilePath;
  let tmpKeyPath = null;

  if (fs.existsSync(apiKeyInput)) {
    keyFilePath = apiKeyInput;
  } else {
    // Raw key content — write to a temp file for notarytool.
    tmpKeyPath = path.join(os.tmpdir(), `apple-api-key-${Date.now()}.p8`);
    fs.writeFileSync(tmpKeyPath, apiKeyInput, { mode: 0o600 });
    keyFilePath = tmpKeyPath;
  }

  try {
    console.log(`Notarizing ${appPath}...`);
    await notarize({
      tool: "notarytool",
      appBundleId: "com.kiji.proxy",
      appPath,
      appleApiKey: keyFilePath,
      appleApiKeyId: apiKeyId,
      appleApiIssuer: apiIssuer,
    });
    console.log("Notarization complete.");
  } finally {
    if (tmpKeyPath) {
      try {
        fs.unlinkSync(tmpKeyPath);
      } catch (_) {}
    }
  }
};
