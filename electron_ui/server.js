const express = require('express');
const path = require('path');
const { createProxyMiddleware } = require('http-proxy-middleware');

const app = express();
const PORT = process.env.PORT || 3000;

// Proxy /details requests to the Go proxy server
app.use('/details', createProxyMiddleware({
  target: 'http://yaak-proxy:8080',
  changeOrigin: true,
  pathRewrite: {
    '^/details': '/details'
  }
}));

// Serve static files from the dist directory with no-cache headers
app.use(express.static(path.join(__dirname, 'dist'), {
  setHeaders: (res, path) => {
    // Disable caching for HTML and JS files
    if (path.endsWith('.html') || path.endsWith('.js')) {
      res.setHeader('Cache-Control', 'no-store, no-cache, must-revalidate, proxy-revalidate, max-age=0');
      res.setHeader('Pragma', 'no-cache');
      res.setHeader('Expires', '0');
    }
  }
}));

// Handle React routing, return all requests to React app
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'dist', 'index.html'));
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`Privacy Proxy UI server running on port ${PORT}`);
});
