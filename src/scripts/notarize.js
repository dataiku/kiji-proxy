const { notarize } = require("@electron/notarize");
const path = require("path");
const fs = require("fs");
const os = require("os");

exports.default = async function notarizing(context) {
  const { electronPlatformName, appOutDir } = context;
  if (electronPlatformName !== "darwin") return;

  const apiKey = process.env.APPLE_API_KEY;
  const apiKeyId = process.env.APPLE_API_KEY_ID;
  const apiIssuer = process.env.APPLE_API_ISSUER;

  if (!apiKey || !apiKeyId || !apiIssuer) {
    console.log(
      "Skipping notarization (APPLE_API_KEY, APPLE_API_KEY_ID, and APPLE_API_ISSUER are all required)"
    );
    return;
  }

  const appName = context.packager.appInfo.productName;
  const appPath = `${appOutDir}/${appName}.app`;

  // notarytool requires the API key as a .p8 file path, not inline content.
  // Write the secret value to a temp file with restricted permissions.
  const tmpKeyPath = path.join(os.tmpdir(), `apple-api-key-${Date.now()}.p8`);
  try {
    fs.writeFileSync(tmpKeyPath, apiKey, { mode: 0o600 });
    console.log(`Notarizing ${appPath}...`);
    await notarize({
      tool: "notarytool",
      appBundleId: "com.kiji.proxy",
      appPath,
      appleApiKey: tmpKeyPath,
      appleApiKeyId: apiKeyId,
      appleApiIssuer: apiIssuer,
    });
    console.log("Notarization complete.");
  } finally {
    try {
      fs.unlinkSync(tmpKeyPath);
    } catch (_) {}
  }
};
