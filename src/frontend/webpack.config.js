const path = require("path");
const HtmlWebpackPlugin = require("html-webpack-plugin");

const isElectron = process.env.ELECTRON === "true";
const isProduction = process.env.NODE_ENV === "production";

module.exports = {
  entry: "./index.js",
  mode: isProduction ? "production" : "development",
  devtool: isProduction ? false : "source-map", // Disable source maps in production to reduce size
  output: {
    path: path.resolve(__dirname, "dist"),
    filename: isProduction ? "bundle.[contenthash].js" : "bundle.js",
    // Use relative path for Electron (file:// protocol), absolute for web
    publicPath: isElectron ? "./" : "/",
    clean: true, // Clean output directory before emit
  },
  module: {
    rules: [
      {
        test: /\.(tsx|ts|jsx|js)$/,
        exclude: /node_modules/,
        use: {
          loader: "babel-loader",
          options: {
            presets: [
              [
                "@babel/preset-env",
                { targets: { browsers: ["last 2 versions"] } },
              ],
              ["@babel/preset-react", { runtime: "automatic" }],
              "@babel/preset-typescript",
            ],
          },
        },
      },
      {
        test: /\.css$/,
        use: ["style-loader", "css-loader"],
      },
      {
        test: /\.(png|jpg|jpeg|gif|svg|ico|icns)$/i,
        type: "asset/resource",
        generator: {
          filename: "assets/[name][ext]",
        },
      },
    ],
  },
  resolve: {
    extensions: [".tsx", ".ts", ".jsx", ".js"],
  },
  plugins: [
    new HtmlWebpackPlugin({
      template: "./index.html",
      filename: "index.html",
      inject: "body", // Automatically inject script tags
      scriptLoading: "blocking",
    }),
  ],
  devServer: {
    static: {
      directory: path.join(__dirname, "dist"),
    },
    compress: true,
    port: 3000,
    host: "0.0.0.0", // Allow external connections (for Docker)
    hot: true,
    liveReload: true,
    historyApiFallback: true,
    watchFiles: {
      paths: ["src/**/*", "*.html", "*.js", "*.tsx", "*.ts"],
      options: {
        usePolling: true, // Enable polling for Docker
        interval: 1000,
      },
    },
    proxy: {
      "/details": {
        target: "http://localhost:8080", // Use localhost for local development
        secure: false,
        changeOrigin: true,
      },
      "/api": {
        target: "http://localhost:8080", // Use localhost for local development
        secure: false,
        changeOrigin: true,
      },
    },
  },
};
