const path = require('path');
const HtmlWebpackPlugin = require('html-webpack-plugin');

module.exports = {
  entry: './index.js',
  output: {
    path: path.resolve(__dirname, 'dist'),
    filename: 'bundle.js',
    publicPath: '/'
  },
  module: {
    rules: [
      {
        test: /\.(tsx|ts|jsx|js)$/,
        exclude: /node_modules/,
        use: {
          loader: 'babel-loader',
          options: {
            presets: [
              ['@babel/preset-env', { targets: { browsers: ['last 2 versions'] } }],
              ['@babel/preset-react', { runtime: 'automatic' }]
            ]
          }
        }
      },
      {
        test: /\.css$/,
        use: ['style-loader', 'css-loader']
      }
    ]
  },
  resolve: {
    extensions: ['.tsx', '.ts', '.jsx', '.js']
  },
  plugins: [
    new HtmlWebpackPlugin({
      template: './index.html',
      filename: 'index.html',
      inject: false
    })
  ],
  devServer: {
    static: {
      directory: path.join(__dirname, 'dist'),
    },
    compress: true,
    port: 3000,
    host: '0.0.0.0', // Allow external connections (for Docker)
    hot: true,
    liveReload: true,
    historyApiFallback: true,
    watchFiles: {
      paths: ['src/**/*', '*.html', '*.js', '*.tsx', '*.ts'],
      options: {
        usePolling: true, // Enable polling for Docker
        interval: 1000,
      },
    },
    proxy: {
      '/details': {
        target: 'http://yaak-proxy:8080', // Use service name for Docker networking
        secure: false,
        changeOrigin: true
      }
    }
  }
};
