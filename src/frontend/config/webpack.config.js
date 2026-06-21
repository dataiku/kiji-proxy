const path = require("path");
const HtmlWebpackPlugin = require("html-webpack-plugin");
const MiniCssExtractPlugin = require("mini-css-extract-plugin");

const isElectron = process.env.ELECTRON === "true";

module.exports = (env, argv) => {
  // Key "production" off webpack's resolved mode. The npm scripts pass
  // --mode production (webpack-cli's flag wins for the actual build mode) but do
  // not set NODE_ENV, so argv.mode is what actually flips devtool/filename for
  // production builds. Fall back to NODE_ENV for any caller that sets it instead.
  const isProduction =
    argv.mode === "production" || process.env.NODE_ENV === "production";

  return {
    context: path.resolve(__dirname, ".."),
    entry: "./index.js",
    mode: isProduction ? "production" : "development",
    devtool: isProduction ? false : "source-map", // No source maps in production (keeps the bundle out of the embedded binary)
    cache: isElectron ? false : undefined, // Disable webpack cache for Electron builds
    output: {
      path: path.resolve(__dirname, "../dist"),
      filename: isProduction ? "bundle.[contenthash].js" : "bundle.js",
      // Use absolute path for both Electron and web
      publicPath: "/",
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
          use: [
            isElectron ? MiniCssExtractPlugin.loader : "style-loader",
            "css-loader",
            {
              loader: "postcss-loader",
              options: {
                postcssOptions: {
                  plugins: [
                    require("@tailwindcss/postcss"),
                    require("autoprefixer"),
                  ],
                },
              },
            },
          ],
        },
        {
          test: /\.(png|jpg|jpeg|gif|svg|ico|icns)$/i,
          type: "asset/resource",
          generator: {
            filename: "assets/[name][ext]",
          },
        },
        {
          test: /\.md$/,
          type: "asset/source",
        },
      ],
    },
    resolve: {
      extensions: [".tsx", ".ts", ".jsx", ".js", ".md"],
    },
    plugins: [
      new HtmlWebpackPlugin({
        template: "./index.html",
        filename: "index.html",
        inject: "body", // Automatically inject script tags
        scriptLoading: "blocking",
      }),
      ...(isElectron
        ? [
          new MiniCssExtractPlugin({
            filename: "[name].[contenthash].css",
          }),
        ]
        : []),
    ],
    devServer: {
      static: {
        directory: path.resolve(__dirname, "../dist"),
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
      proxy: [
        {
          context: ["/details", "/api", "/v1", "/version", "/health"],
          target: "http://localhost:8080",
          secure: false,
          changeOrigin: true,
        },
      ],
    },
  };
};
