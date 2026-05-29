cask "kiji-privacy-proxy" do
  version "0.0.0"
  sha256 "0000000000000000000000000000000000000000000000000000000000000000"

  url "https://github.com/dataiku/kiji-proxy/releases/download/v#{version}/Kiji-Privacy-Proxy-#{version}.dmg",
      verified: "github.com/dataiku/kiji-proxy/"
  name "Kiji Privacy Proxy"
  desc "Privacy layer that masks PII in requests to AI APIs"
  homepage "https://github.com/dataiku/kiji-proxy"

  livecheck do
    url :url
    strategy :github_latest
  end

  depends_on macos: ">= :high_sierra"

  app "Kiji Privacy Proxy.app"

  zap trash: [
    "~/.kiji-proxy",
    "~/Library/Application Support/Kiji Privacy Proxy",
    "~/Library/Caches/com.kiji.proxy",
    "~/Library/Caches/com.kiji.proxy.ShipIt",
    "~/Library/Logs/Kiji Privacy Proxy",
    "~/Library/Preferences/com.kiji.proxy.plist",
    "~/Library/Saved Application State/com.kiji.proxy.savedState",
  ]
end
